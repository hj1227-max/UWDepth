import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class LGFI(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x


class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x
        
class BackgroundLightEstimator(nn.Module):
    """
    背景光估计模块 - 基于全局特征学习背景光B
    """
    def __init__(self, feature_dims=[48, 80, 128], hidden_dim=64, topk_ratio=0.1):
        super().__init__()
        self.feature_dims = feature_dims
        self.topk_ratio = topk_ratio
        self.hidden_dim = hidden_dim
        
        # 多尺度特征融合
        self.feature_fusion = nn.ModuleDict({
            'conv_0': nn.Conv2d(feature_dims[0], hidden_dim, 3, padding=1),
            'conv_1': nn.Conv2d(feature_dims[1], hidden_dim, 3, padding=1),
            'conv_2': nn.Conv2d(feature_dims[2], hidden_dim, 3, padding=1),
        })
        
        # 距离注意力模块 - 学习哪些区域是"最远"的
        self.distance_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim//2, 1, 1),
            nn.Sigmoid()
        )
        
        # 方差感知模块
        self.variance_attention = VarianceAwareModule(hidden_dim)
        
        # 背景光回归
        self.bg_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 3),  # 输出RGB三通道背景光
            nn.Sigmoid()  # 限制在[0,1]范围
        )
        
    def forward(self, features, input_img=None):
        """
        features: 编码器输出的多尺度特征列表 [feat0, feat1, feat2]
        input_img: 原始输入图像，用于参考
        """
        # 多尺度特征融合
        fused_features = []
        for i, feat in enumerate(features):
            # 上采样到最大特征图的尺寸
            target_size = features[2].shape[2:]
            if feat.shape[2:] != target_size:
                feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            else:
                feat_resized = feat
                
            # 特征变换
            transformed = self.feature_fusion[f'conv_{i}'](feat_resized)
            fused_features.append(transformed)
        
        # 特征聚合
        fused_feat = torch.stack(fused_features).mean(dim=0)
        
        # 距离注意力权重
        distance_weights = self.distance_attention(fused_feat)
        
        # 方差感知权重
        variance_weights = self.variance_attention(fused_feat)
        
        # 结合距离和方差权重
        combined_weights = distance_weights * variance_weights
        
        # 选择最可靠的区域 (topk)
        B, C, H, W = fused_feat.shape
        k = max(1, int(H * W * self.topk_ratio))
        
        # 重塑权重
        weights_flat = combined_weights.view(B, -1)
        topk_weights, topk_indices = torch.topk(weights_flat, k, dim=1)
        
        # 归一化权重
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # 选择对应的特征
        fused_feat_flat = fused_feat.view(B, C, -1)
        selected_features = torch.gather(fused_feat_flat, 2, 
                                       topk_indices.unsqueeze(1).expand(-1, C, -1))
        
        # 加权平均
        weighted_features = selected_features * topk_weights.unsqueeze(1)
        pooled_features = weighted_features.sum(dim=2)
        
        # 回归背景光
        background_light = self.bg_regressor(pooled_features.unsqueeze(-1).unsqueeze(-1))  # 此时形状为 [B, 3]
        background_light = background_light.unsqueeze(2).unsqueeze(3)  # 增加 (1,1) 空间维度 → [B, 3, 1, 1]
        
        # 返回背景光和注意力图（用于可视化）
        attention_map = F.interpolate(combined_weights, scale_factor=8, mode='bilinear', align_corners=False)
        
        return background_light, attention_map


class VarianceAwareModule(nn.Module):
    """
    方差感知模块 - 寻找方差小的稳定区域
    """
    def __init__(self, feature_dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 局部方差计算
        self.variance_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size, 
                                      padding=self.padding, groups=feature_dim, bias=False)
        self.variance_conv.weight.data.fill_(1.0 / (kernel_size * kernel_size))
        self.variance_conv.weight.requires_grad = False
        
    def forward(self, x):
        # 计算局部均值
        local_mean = self.variance_conv(x)
        
        # 计算局部方差
        local_variance = self.variance_conv(x ** 2) - local_mean ** 2
        
        # 方差越小权重越大（加上epsilon防止除零）
        variance_weights = 1.0 / (local_variance.mean(dim=1, keepdim=True) + 1e-6)
        
        # 归一化
        variance_weights = variance_weights / (variance_weights.sum(dim=(2,3), keepdim=True) + 1e-6)
        
        return variance_weights
class AttenuationEstimator(nn.Module):
    """
    衰减系数β估计模块
    β与深度、图像特征密切相关
    """
    def __init__(self, feature_dims=[48, 80, 128], hidden_dim=64, num_wavelengths=3):
        super().__init__()
        self.num_wavelengths = num_wavelengths  # 通常为RGB三通道
        
        # 多尺度特征编码
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            ) for dim in feature_dims
        ])
        
        # 深度特征与衰减系数的关系建模
        self.beta_regressor = nn.Sequential(
            nn.Linear(hidden_dim * len(feature_dims) + 3, hidden_dim),  # +3 for background light
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_wavelengths),  # β_r, β_g, β_b
            nn.Softplus()  # 确保β > 0
        )
        
        # 物理一致性约束
        self.physical_constraint = PhysicsRefinementModule()
        
    def forward(self, features, background_light, depth_estimate=None):
        # 提取全局特征（形状：[B, C]，C为融合后的特征维度）
        global_features = []
        for i, feat in enumerate(features):
            encoded = self.feature_encoders[i](feat)
            global_features.append(encoded.squeeze(-1).squeeze(-1))  # 压缩为 [B, hidden_dim]
        fused_global = torch.cat(global_features, dim=1)  # [B, hidden_dim * len(features)]
    
        # 结合背景光信息（关键修改：压缩background_light的空间维度）
        if background_light is not None:
            # 将 [B, 3, 1, 1] 压缩为 [B, 3]
            background_light_flat = background_light.squeeze(2).squeeze(2)  # 去掉两个空间维度
            combined_features = torch.cat([fused_global, background_light_flat], dim=1)  # 维度一致：2维
        else:
            combined_features = fused_global
    
        # 回归衰减系数（后续再恢复为 [B, 3, 1, 1]）
        beta = self.beta_regressor(combined_features)  # [B, 3]
        beta = beta.unsqueeze(2).unsqueeze(3)  # 恢复为 [B, 3, 1, 1]
    
        # 物理约束
        beta = self.physical_constraint(beta, background_light)  # background_light仍是[B,3,1,1]
    
        return beta


class PhysicsRefinementModule(nn.Module):
    """
    物理约束模块 - 确保β符合水下光学特性
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, beta, background_light):
        # 约束1: β > 0（形状 [B, 3, 1, 1]）
        beta = torch.clamp(beta, min=1e-6)
        
        # 约束2: 软排序（需将空间维度压缩，处理后恢复）
        B, C = beta.shape[0], beta.shape[1]
        beta_flat = beta.view(B, C)  # 压缩为 [B, 3] 进行排序
        beta_ordered_flat = self.soft_ordering(beta_flat)
        beta_ordered = beta_ordered_flat.view(B, C, 1, 1)  # 恢复为 [B, 3, 1, 1]
        
        # 约束3: 与背景光颜色适配（背景光需压缩为 [B, 3]）
        if background_light is not None:
            bg_flat = background_light.view(B, C)  # [B, 3, 1, 1] → [B, 3]
            bg_intensity = bg_flat / (bg_flat.sum(dim=1, keepdim=True) + 1e-6)
            beta_adjusted_flat = self.adapt_to_background(beta_ordered_flat, bg_intensity)
            beta_adjusted = beta_adjusted_flat.view(B, C, 1, 1)  # 恢复为 [B, 3, 1, 1]
            return beta_adjusted
        
        return beta_ordered
    
    def soft_ordering(self, beta):
        """软排序约束"""
        # 使用可微的排序近似
        beta_soft_sorted = torch.zeros_like(beta)
        for i in range(beta.size(0)):
            # 对每个样本单独处理
            sample_beta = beta[i]
            # 使用softmax实现可微排序
            weights = F.softmax(sample_beta * 10, dim=0)  # 温度参数控制排序硬度
            sorted_values, _ = torch.sort(sample_beta)
            beta_soft_sorted[i] = (weights * sorted_values).sum(dim=0)
        return beta_soft_sorted
    
    def adapt_to_background(self, beta, bg_intensity):
        """根据背景光调整β"""
        # 背景光中蓝色成分多意味着红色衰减强
        red_attenuation_factor = 1.0 + bg_intensity[:, 2]  # 蓝色通道
        green_attenuation_factor = 1.0 + bg_intensity[:, 1] * 0.5
        blue_attenuation_factor = torch.ones_like(red_attenuation_factor) 
        
        attenuation_factors = torch.stack([
            red_attenuation_factor,
            green_attenuation_factor, 
            blue_attenuation_factor
        ], dim=1)
        
        return beta * attenuation_factors
class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 10]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]
            else:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.depth = [4, 4, 10]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0]+3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i]*2+3, self.dims[i+1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]
         # 新增背景光估计器
        self.bg_estimator = BackgroundLightEstimator(
            feature_dims=self.dims,
            hidden_dim=64,
            topk_ratio=0.05  # 选择5%的像素用于背景光估计
        )
        # 衰减系数估计器
        self.beta_estimator = AttenuationEstimator(
            feature_dims=self.dims,
            hidden_dim=64
        )
        
        # 物理模型refinement
        self.physics_refinement = PhysicsRefinementModule()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        x = (x - 0.45) / 0.225

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)

        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)
        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            tmp_x.append(x)

            features.append(x)

        return features

    def forward(self, x):
        features = self.forward_features(x)
        
        background_light, attention_map = self.bg_estimator(features, x)
        beta = self.beta_estimator(features, background_light)


        return features,beta,background_light
