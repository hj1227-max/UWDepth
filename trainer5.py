from __future__ import absolute_import, division, print_function

import random
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from layers import *

import datasets
import networks
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from ewma import EWMA

torch.backends.cudnn.benchmark = True

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 40

        self.threshold = EWMA(momentum=0.98)
        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        #旋转数据增强比例控制
        assert 0<=self.opt.rotation_aug_proportion<=1, "proportion must be between 0 and 1"
        #旋转蒸馏提前校验
        if self.opt.anomaly_mask == True or self.opt.supervised_loss == True or self.opt.rotation_augment == True:
            assert self.opt.load_teacher_network == True, "to use TGAM or RD, load teacher model first"

        self.num_rotated_imgs = int(self.opt.rotation_aug_proportion * self.opt.batch_size)
        self.num_rotated_imgs = max(1,self.num_rotated_imgs)

        self.num_scales = len(self.opt.scales)
        self.opt.frame_ids = [0,-1,1]
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        self.use_pose_net = True
        #物理增强相关变换
        self.gaussian_blur_transform = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 2.0))

        self.models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                     self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())



        if self.opt.pose_model_type == "separate_resnet":
            self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models_pose["pose_encoder"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

            self.models_pose["pose"] = networks.PoseDecoder(
                self.models_pose["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        elif self.opt.pose_model_type == "shared":
            self.models_pose["pose"] = networks.PoseDecoder(
                self.models["encoder"].num_ch_enc, self.num_pose_frames)

        elif self.opt.pose_model_type == "posecnn":
            self.models_pose["pose"] = networks.PoseCNN(
                self.num_input_frames if self.opt.pose_model_input == "all" else 2)

        self.models_pose["pose"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
                            self.model_optimizer,
                            T_0=int(self.opt.lr[2]),
                            T_mul=1,
                            eta_min=self.opt.lr[1],
                            last_epoch=-1,
                            max_lr=self.opt.lr[0],
                            warmup_steps=0,
                            gamma=0.5
                            )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.5
        )

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"FLSea": datasets.FLseaDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.tiff' if self.opt.tiff else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, 4,
            is_train=True, img_ext=img_ext, load_enh_img=self.opt.load_enhanced_img)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width, 4, is_train=False, img_ext=img_ext,load_depth=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)


        #3D投影层（用于伪标签筛选）
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)  #深度到点云
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)    #点云到2D投影
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_filenames)))

        self.prev_depth = None  # 上一帧深度图 (B, 1, H, W)
        self.B = None
        self.beta = None


        self.best_abs_rel = float('inf')  # 初始值设为无穷大，确保首次验证能保存
        # 最优模型保存路径（固定路径，每次更新覆盖，避免多文件夹冗余）
        self.best_model_dir = os.path.join(self.log_path, "models", "best_model")
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.prev_depth = None
            self.prev_img_id = None
            self.run_epoch()

            # -------------------------- 新增：验证与最优模型判断 --------------------------
            # 1. 执行验证，获取当前epoch的验证集de/abs_rel
            current_abs_rel = self.val()
            # 2. 对比当前指标与历史最优（abs_rel越小模型越优）
            if current_abs_rel < self.best_abs_rel:
                self.best_abs_rel = current_abs_rel  # 更新历史最优指标
                self.save_best_model()  # 保存当前模型为新最优
            # -------------------------- 保留原有：按频率保存普通模型 --------------------------
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()


    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        self.model_pose_lr_scheduler.step()

        loss_of_epoch = {"reprojection_loss": 0.0,
                        "smooth_loss": 0.0,
                        "phys_t_loss": 0.0,
                        "phys_t_depth_loss": 0.0,
                        "phys_B_loss":0.0,

                         }

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            for key in loss_of_epoch.keys():
                loss_of_epoch[key] += losses[key]

            self.model_optimizer.zero_grad()
            self.model_pose_optimizer.zero_grad()

            losses["total_loss"].backward()
            self.model_optimizer.step()
            self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            to_log = batch_idx % self.opt.log_frequency == 0

            if to_log:
                self.log_time(batch_idx, duration, losses["total_loss"].cpu().data)
                # if "depth_gt" in inputs:
                #     self.val()

            self.step += 1

        self.log("train", inputs, outputs, loss_of_epoch)

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        D_prev = self.prev_depth if self.prev_depth is not None else None

        # we only feed the image with frame_id 0 through the depth encoder
        features,beta, B= self.models["encoder"](inputs["color_aug", 0, 0])
        beta = beta.to(self.device)
        B = B.to(self.device)
        outputs = self.models["depth"](features,beta, B)

        # 保存当前帧的最高分辨率深度图（detach避免梯度追踪）
        self.prev_depth = outputs[("disp", 0)].detach()

        #位姿输出
        outputs.update(self.predict_poses(inputs, features))
        #生成重建图像（试图合成）
        self.generate_images_pred(inputs, outputs)

        losses = self.compute_losses(inputs, outputs,beta,B)

        return outputs, losses

    def is_sequential(self, prev_id, current_id):
        """判断上一帧与当前帧是否时序连续（帧号差为1）"""
        return current_id - prev_id == 1

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        self.set_eval()
        current_abs_rel = None
        try:
            inputs = next(self.val_iter)  # 正确写法，使用内置的 next() 函数
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)  # 正确写法，使用内置的 next() 函数

        with torch.no_grad():
            # 验证阶段只进行深度估计，跳过位姿估计和图像重建
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            # 只使用当前帧进行深度估计，不使用prev_depth
            features, beta, B= self.models["encoder"](inputs["color_aug", 0, 0])
            beta = beta.to(self.device)
            B = B.to(self.device)
            outputs = self.models["depth"](features,beta,B)

            # 获取最高分辨率的深度图
            disp = outputs[("disp", 0)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, 0)] = depth

            # 初始化一个空的losses字典用于日志记录
            errorss = {}

            if "depth_gt" in inputs:
                print("验证阶段")
                # 1. 提取地面真实深度和预测深度（需确保尺寸匹配）
                depth_gt = inputs["depth_gt"]  # 形状: (B, 1, H, W)
                depth_pred = outputs[("depth", 0, 0)]  # 预测深度，形状: (B, 1, H, W)

                # 2. 关键：将预测深度上采样到与真实深度相同的尺寸
                # 使用双线性插值（align_corners=True保持边缘对齐）
                depth_pred = torch.nn.functional.interpolate(
                    depth_pred,
                    size=depth_gt.shape[2:],  # 目标尺寸：(H_gt, W_gt)
                    mode='bilinear',
                    align_corners=True
                )

                # 在尺度对齐前添加更严格的过滤
                mask = (depth_gt > 1e-3) & (depth_gt < 80)  # 假设有效深度范围是 0.001~80
                depth_gt_valid = depth_gt[mask]
                depth_pred_valid = depth_pred[mask]

                # 确保过滤后还有有效数据（避免空数组导致后续计算出错）
                if len(depth_gt_valid) == 0 or len(depth_pred_valid) == 0:
                    print("警告：没有有效深度数据用于计算指标")

                # 3. 尺度对齐（关键步骤：用GT中位数校准预测深度）
                pred_median = torch.median(depth_pred_valid)
                gt_median = torch.median(depth_gt_valid)
                # 避免除以 0 或极端值
                if pred_median < 1e-6 or gt_median < 1e-6:
                    print("警告：预测或真实深度中位数异常")
                depth_pred_scaled = depth_pred_valid * (gt_median / pred_median)  # 尺度校准

                # 4. 限制深度范围（避免极端值影响）
                depth_pred_scaled = torch.clamp(depth_pred_scaled, min=1e-3, max=80)

                # 5. 转换为numpy数组（函数支持torch张量，但numpy更稳妥）
                gt_np = depth_gt_valid.cpu().numpy()
                pred_np = depth_pred_scaled.cpu().numpy()

                # 6. 调用compute_depth_errors计算指标
                errors = self.compute_depth_errors(gt_np, pred_np)

                # 7. 输出或使用指标（例如保存到losses字典）
                current_abs_rel = errors["de/abs_rel"]  # 获取绝对相对误差
                print(f"验证集指标: de/abs_rel={current_abs_rel:.6f}, de/rms={errors['de/rms']:.6f}")
                errorss.update(errors)  # 存入losses供后续日志记录

            self.log("val", inputs, outputs, errorss)
            del inputs, outputs, errorss

        self.set_train()
        return current_abs_rel
    #视图合成   重建
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if self.opt.load_enhanced_img:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color_enh", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    #重投影损失   计算重建图像和目标图像之间的相似度
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    #尺度不变对数损失（深度一致性）
    def compute_silog_loss(self, depth_est, depth_gt):
        """Compute scale invariant loss
        """
        d = torch.log(depth_est) - torch.log(depth_gt)
        return torch.sqrt((d ** 2).mean() - 1.0 * (d.mean() ** 2))


    def compute_losses(self, inputs, outputs,beta,B):
        losses = {
            "reprojection_loss": 0.0,
            "smooth_loss": 0.0,
            "phys_t_loss": 0.0,
            "phys_t_depth_loss": 0.0,
            "total_loss": 0.0,
            "phys_B_loss": 0.0,
        }

        # -------------------------- 原视觉损失计算 --------------------------
        for scale in self.opt.scales:
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            if self.opt.load_enhanced_img:
                target = inputs[("color_enh", 0, source_scale)]
            else:
                target = inputs[("color", 0, source_scale)]

            # 重投影损失
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            # 自动掩码（保留原逻辑）
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if self.opt.load_enhanced_img:
                        pred = inputs[("color_enh", frame_id, source_scale)]
                    else:
                        pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss = identity_reprojection_losses
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 1e-5  # 打破平局
                combined = torch.cat((identity_reprojection_loss, reprojection_losses), dim=1)
            else:
                combined = reprojection_losses

            # 选择最小损失（保留原逻辑）
            to_optimise = combined.min(dim=1)[0] if combined.shape[1] > 1 else combined.squeeze(1)

            # 累积重投影损失
            loss_reprojection = to_optimise.mean()
            losses["reprojection_loss"] += loss_reprojection
            

            # 平滑损失（保留原逻辑，修正归一化）
            mean_disp = disp.mean(2, keepdim=True).mean(3, keepdim=True)
            norm_disp = disp / (mean_disp + 1e-7)

            # 使用下采样后的图像计算平滑损失
            smooth_loss = get_smooth_loss(norm_disp, color)
            losses["smooth_loss"] += smooth_loss

            if scale == 0:
                # ---------- 物理约束损失 ----------
                disp = outputs[("disp", scale)]  # [B,1,H,W]
                disp_pred, depth_pred = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                I = inputs[("color", 0, source_scale)]  # [B,3,H,W]
                J = inputs[("color_enh", 0, source_scale)]  # [B,3,H,W]
                
        
                # 计算透射率 t = exp(-β * d)
                beta_expanded = beta.expand(-1, -1, depth_pred.shape[2], depth_pred.shape[3])
                t = torch.exp(-beta_expanded * depth_pred)
        
                # --- (1) 公式 ---
                I_recon = J * t + B * (1 - t)
                L_t = torch.abs(I_recon - I).mean()
                losses["phys_t_loss"] += L_t
        

                
                # --- (3) B约束损失 ---
                # 选取最深的 top 0.1% 像素
                B_batch = []
                k_ratio = 0.0005  # top 0.05%
                _, _, H, W = I.shape
                num_pixels = H * W
                k = max(1, int(num_pixels * k_ratio))
        
                for b in range(I.shape[0]):
                    depth_b = depth_pred[b, 0].flatten()  # [H*W]
                    I_b = I[b]  # [3,H,W]
                    topk_values, topk_idx = torch.topk(depth_b, k)
                    I_b_flat = I_b.view(3, -1)
                    B_pseudo = I_b_flat[:, topk_idx].mean(dim=1, keepdim=True).unsqueeze(-1)  # [3,1,1]
                    B_batch.append(B_pseudo)
        
                B_pseudo = torch.stack(B_batch, dim=0)  # [B,3,1,1]
        
                # L1约束：鼓励预测B与伪真值一致
    
                L_B = torch.abs(B - B_pseudo).mean()
                losses["phys_B_loss"] += L_B
    
            if scale == 0:
            # ---------- 总损失 ----------
                loss_total_scale = (
                        loss_reprojection
                        + self.opt.disparity_smoothness * smooth_loss * (1.0 / (2 ** scale)) 
                        + self.opt.lambda_t * L_t 
                        + self.opt.lambda_B * L_B 
                    )
            else:
                loss_total_scale = (
                      loss_reprojection
                      + self.opt.disparity_smoothness * smooth_loss * (1.0 / (2 ** scale)) 
                  )
            
            losses["total_loss"] +=loss_total_scale
        losses["total_loss"] /= self.num_scales
        return losses



    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [608, 968], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        验证阶段只记录指标，不记录图像；训练阶段保持原图像记录
        """
        writer = self.writers[mode]

        # 无论训练还是验证，都记录指标
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.epoch)

        # 只在训练阶段记录图像，验证阶段跳过
        if mode == "train":
            for j in range(min(4, self.opt.batch_size)):
                for frame_id in self.opt.frame_ids:
                    if ("color", frame_id, 0) in inputs:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, 0, j),
                            inputs[("color", frame_id, 0)][j].data, self.epoch)
                    if frame_id != 0 and ("color", frame_id, 0) in outputs:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, 0, j),
                            outputs[("color", frame_id, 0)][j].data, self.epoch)

                if ("disp", 0) in outputs:
                    writer.add_image(
                        "disp_{}/{}".format(0, j),
                        normalize_image(outputs[("disp", 0)][j]), self.epoch)
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def compute_depth_errors(self, gt, pred):
        """计算标准深度估计误差指标"""
        thresh = np.maximum(gt / pred, pred / gt)
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = np.sqrt(((gt - pred) ** 2).mean())
        rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return {
            "de/abs_rel": abs_rel,
            "de/sq_rel": sq_rel,
            "de/rms": rmse,
            "de/log_rms": rmse_log,
            "da/a1": a1,
            "da/a2": a2,
            "da/a3": a3
        }


    def save_best_model(self):
        """保存当前最优模型到固定路径（覆盖旧最优模型）"""
        print(f"\n当前de/abs_rel {self.best_abs_rel:.6f} 优于历史最优，更新保存最优模型...")

        # 保存深度网络（encoder + depth）
        for model_name, model in self.models.items():
            save_path = os.path.join(self.best_model_dir, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == 'encoder':
                # 保留预测时需要的尺寸信息
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        # 保存姿态网络（pose相关模型）
        for model_name, model in self.models_pose.items():
            save_path = os.path.join(self.best_model_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)

        # 保存优化器状态（可选，如需续训最优模型）
        torch.save(self.model_optimizer.state_dict(),
                   os.path.join(self.best_model_dir, "adam.pth"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(),
                       os.path.join(self.best_model_dir, "adam_pose.pth"))