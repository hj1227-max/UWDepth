# def modify_file_content(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     modified_lines = []
#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) >= 2:
#             # 处理路径部分，去掉重复的flatiron
#             path_parts = parts[0].split('/')
#             if len(path_parts) == 3 and path_parts[1] == path_parts[2]:
#                 new_path = '/'.join(path_parts[:2])
#                 # 重新组合行内容
#                 new_line = new_path + ' ' + ' '.join(parts[1:])
#                 modified_lines.append(new_line)
#             else:
#                 modified_lines.append(line.strip())
#         else:
#             modified_lines.append(line.strip())
#
#     # 写回文件
#     with open(filename, 'w', encoding='utf-8') as file:
#         for line in modified_lines:
#             file.write(line + '\n')
#
#
# # 使用示例
# modify_file_content("/mnt/data_sdd/hj/WaterMono-main/splits/OUC/train_files.txt")

'''
import os
from glob import glob

import os
from glob import glob


def main():
    base_dir = "/mnt/data_sdd/hj/datasets/water/"
    categories = ['canyons', 'red_sea']

    # 支持的图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    test_lines = []
    val_lines = []
    train_lines = []

    for category in categories:
        # 获取所有子文件夹路径（只包含目录）
        sub_dirs = [d for d in glob(os.path.join(base_dir, category, '*')) if os.path.isdir(d)]

        print(f"在 {category} 中找到 {len(sub_dirs)} 个子文件夹")

        for sub_dir in sub_dirs:
            img_dir = os.path.join(sub_dir, 'imgs')
            if not os.path.exists(img_dir):
                print(f"警告: {img_dir} 不存在，跳过")
                continue

            # 获取所有图片文件并按名称排序
            all_files = os.listdir(img_dir)
            # 过滤出图片文件
            img_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

            if not img_files:
                print(f"警告: {img_dir} 中没有图片文件，跳过")
                continue

            img_files.sort()  # 按文件名排序
            total_images = len(img_files)

            # 提取相对路径前缀
            rel_path = f"{category}/{os.path.basename(sub_dir)}"

            print(f"处理 {rel_path}: 找到 {total_images} 张图片")

            # 处理前300张 -> test
            for img in img_files[:300]:
                img_name = os.path.splitext(img)[0]  # 移除扩展名
                test_lines.append(f"{rel_path} {img_name}")

            # 处理301-350张 -> val
            if total_images > 300:
                for img in img_files[300:350]:
                    img_name = os.path.splitext(img)[0]
                    val_lines.append(f"{rel_path} {img_name}")

            # 处理剩余张 -> train
            if total_images > 350:
                for img in img_files[350:]:
                    img_name = os.path.splitext(img)[0]
                    train_lines.append(f"{rel_path} {img_name}")

    # 确保输出目录存在
    output_dir = "/mnt/data_sdd/hj/WaterMono-main/splits/OUC"
    os.makedirs(output_dir, exist_ok=True)

    # 写入结果文件
    with open(os.path.join(output_dir, "test_files.txt"), 'w') as f:
        f.write('\n'.join(test_lines))

    with open(os.path.join(output_dir, "val_files.txt"), 'w') as f:
        f.write('\n'.join(val_lines))

    with open(os.path.join(output_dir, "train_files.txt"), 'w') as f:
        f.write('\n'.join(train_lines))

    print("\n文件生成完成！")
    print(f"测试集: {len(test_lines)} 条记录")
    print(f"验证集: {len(val_lines)} 条记录")
    print(f"训练集: {len(train_lines)} 条记录")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
'''
'''
import os


def extract_every_nth_line(input_file, output_file, n=6, start_line=1):
    """
    从TXT文件中提取每n行的第start_line行
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n\r') for line in f]

        # 提取对应的行
        extracted_lines = []
        current_index = start_line - 1  # 转换为0-based索引

        while current_index < len(lines):
            extracted_lines.append(lines[current_index])
            current_index += n

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write('\n'.join(extracted_lines))

        print(f"完成! 从 {input_file} 提取了 {len(extracted_lines)} 行到 {output_file}")
        return True

    except Exception as e:
        print(f"错误: {e}")
        return False


# 直接使用
if __name__ == "__main__":
    input_file = "/mnt/data_sdd/hj/WaterMono-main/splits/OUC/test_files.txt"  # 替换为你的输入文件路径
    output_file = "/mnt/data_sdd/hj/WaterMono-main/splits/OUC/output_combined.txt"  # 输出文件路径
    n = 6  # 间隔行数
    start_line = 1  # 每组中的第几行

    extract_every_nth_line(input_file, output_file, n, start_line)
'''


'''
#numpy==2.0.2,tensorFlow==2.20.0
import os
import io
import matplotlib.pyplot as plt
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator

#====================#
# 1. 设置事件文件路径
#====================#
event_path = "/mnt/data_sdd/hj/WaterMono-main/tmp/mygai/train/events.out.tfevents.1760010962.vip-Precision-7920-Tower"
if not os.path.exists(event_path):
    raise FileNotFoundError(f"未找到事件文件：{event_path}")

#====================#
# 2. 加载事件文件
#====================#
ea = event_accumulator.EventAccumulator(event_path, size_guidance={
    event_accumulator.SCALARS: 0,
    event_accumulator.IMAGES: 0,
    event_accumulator.HISTOGRAMS: 0,
})
ea.Reload()

#====================#
# 3. 打印可用标签
#====================#
print("\n==================== 可用数据标签 ====================")
print("标量 (scalars):", ea.Tags().get('scalars', []))
print("图像 (images):", ea.Tags().get('images', []))
print("直方图 (histograms):", ea.Tags().get('histograms', []))
print("======================================================\n")

#====================#
# 4. 绘制损失曲线
#====================#
if 'train/loss' in ea.Tags().get('scalars', []):
    scalars = ea.Scalars('train/loss')
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]

    plt.figure(figsize=(8,5))
    plt.plot(steps, values, label='train/loss', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_loss_curve.png", dpi=300)
    plt.close()
    print("✅ 已保存损失曲线：train_loss_curve.png")
else:
    print("⚠️ 未找到 'train/loss' 标量标签，请查看上方打印的标签名称并替换。")

#====================#
# 5. 保存事件文件中的图像
#====================#
save_dir = "/mnt/data_sdd/hj/WaterMono-main/tmp/mygai/train/event_images/"
os.makedirs(save_dir, exist_ok=True)

image_tags = ea.Tags().get('images', [])
if image_tags:
    print(f"\n事件文件中包含 {len(image_tags)} 个图像标签：{image_tags}")

    for tag in image_tags:
        images = ea.Images(tag)
        print(f"\n🔹 保存图像标签: {tag}（共 {len(images)} 张）")

        for i, img_event in enumerate(images):
            image_data = img_event.encoded_image_string
            image = Image.open(io.BytesIO(image_data))

            # 构造保存路径
            filename = f"{tag.replace('/', '_')}_step{img_event.step:06d}.png"
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)
        print(f"✅ {tag}: 已保存 {len(images)} 张图像到 {save_dir}")
else:
    print("⚠️ 未在事件文件中发现图像数据。")

print("\n✅ 所有图像已保存至：", os.path.abspath(save_dir))

'''

#批量将tif图转为png图
import os
import numpy as np
from PIL import Image

# ============ 参数设置 ============
input_folder = "/mnt/data_sdd/hj/datasets/water/canyons/tiny_canyon/depth/"   # 输入文件夹
output_folder = "/mnt/data_sdd/hj/datasets/water/canyons/tiny_canyon/depth1/"  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

# ============ 函数定义 ============
def normalize_to_uint8(arr):
    """将浮点或整型图像归一化到 [0,255]"""
    arr = arr.astype(np.float32)
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - min_val) / (max_val - min_val)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr

def convert_tif_to_png(input_path, output_path):
    """单张 TIF 转 PNG"""
    try:
        with Image.open(input_path) as img:
            arr = np.array(img)

            # 如果是浮点图或16位图 -> 归一化
            if arr.dtype in [np.float32, np.float64, np.uint16, np.int16]:
                arr = normalize_to_uint8(arr)

            # 保存为 PNG
            Image.fromarray(arr).save(output_path)
            print(f"✅ 已转换: {input_path} → {output_path}")

    except Exception as e:
        print(f"❌ 转换失败: {input_path}")
        print(f"   错误信息: {e}")

# ============ 主程序 ============
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.tif', '.tiff')):
        in_path = os.path.join(input_folder, file_name)
        out_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
        convert_tif_to_png(in_path, out_path)

print("🎉 全部转换完成！")

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#====================#
# 1. 设置路径
#====================#
pred_path = "/mnt/data_sdd/hj/datasets/water/canyons/flatiron/imgs/16233053670626626.jpg"    # 预测深度图路径
gt_path   = "/mnt/data_sdd/hj/datasets/water/canyons/flatiron/imgs/16233053670626626_disp.jpeg"     # 真实深度图路径
save_path = "/mnt/data_sdd/hj/datasets/water/canyons/flatiron/rmse_map.png"       # 保存误差图路径

#====================#
# 2. 读取图像
#====================#
pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

if pred is None or gt is None:
    raise FileNotFoundError("❌ 图像路径错误，请检查 pred_path 和 gt_path")

#====================#
# 3. 转换为灰度图（若为三通道）
#====================#
if len(pred.shape) == 3:
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
if len(gt.shape) == 3:
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

#====================#
# 4. 尺寸对齐
#====================#
if pred.shape != gt.shape:
    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)

#====================#
# 5. 转 float32 并归一化到 [0,1]
#====================#
pred = pred.astype(np.float32)
gt = gt.astype(np.float32)
pred /= (pred.max() + 1e-8)
gt /= (gt.max() + 1e-8)

#====================#
# 6. 计算 RMSE map
#====================#
error_map = (pred - gt) ** 2
rmse_value = np.sqrt(np.mean(error_map))
rmse_map = np.sqrt(error_map)

print(f"✅ 图像 RMSE: {rmse_value:.6f}")

#====================#
# 7. 可视化并保存
#====================#
plt.figure(figsize=(6,5))
plt.imshow(rmse_map, cmap='inferno')
plt.colorbar(label='RMSE per pixel')
plt.title(f"RMSE Map (RMSE={rmse_value:.4f})")
plt.axis('off')

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"💾 已保存误差图到: {os.path.abspath(save_path)}")
'''






