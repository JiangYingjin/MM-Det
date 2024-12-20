# %%
import torch
from PIL import Image
from models import MMEncoder
from options.base_options import BaseOption
from LLaVA.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import numpy as np

# 全局变量存储模型实例
_mm_model = None


def get_mm_model():
    """
    获取多模态模型的单例实例
    Returns:
        MMEncoder: 多模态编码器模型实例
    """
    global _mm_model
    if _mm_model is None:
        # 配置
        opt = BaseOption()
        config = opt.parse().__dict__

        # 加载模型
        config["lmm_ckpt"] = "sparklexfantasy/llava-7b-1.5-rfrd"
        config["load_4bit"] = False
        _mm_model = MMEncoder(config)
        _mm_model.eval()
    return _mm_model


def extract_mm_features(image):
    """
    提取单张图片的多模态特征
    Args:
        image: 输入图像路径或PIL Image对象
    Returns:
        dict: 包含visual和textual特征的字典
    """
    # 获取模型实例
    model = get_mm_model()

    # 处理图像
    with torch.inference_mode():
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (torch.Tensor, list, tuple, np.ndarray)):
            # 处理从get_video_frames返回的numpy数组
            img = Image.fromarray(np.uint8(image)).convert("RGB")
        else:
            raise TypeError("输入图像必须是路径字符串、PIL Image对象或numpy数组")

        visual_features, mm_features = model(img)

        # 提取特征
        mm_layer_features = {}
        for idx, layer in enumerate(model.selected_layers):
            mm_layer_features[str(layer)] = mm_features[idx].cpu()

        # # 打印特征信息
        # print("视觉特征 shape:", visual_features.squeeze(0).cpu().shape)
        # print("文本特征信息:")
        # for layer, feat in mm_layer_features.items():
        #     print(f"- Layer {layer} shape:", feat.shape)

        features = {
            "visual": visual_features.squeeze(0).cpu(),
            "textual": mm_layer_features,
        }

        return features


def get_video_frames(video_path):
    """
    获取视频的第一帧和第八帧
    Args:
        video_path: 视频文件路径
    Returns:
        tuple: (第一帧图像, 第八帧图像), 如果视频帧数不足则返回 (None, None)
    """
    import cv2

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return None, None

    frames = []
    frame_count = 0

    # 读取前8帧
    while frame_count < 8:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count == 1 or frame_count == 8:
            # 将BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    # 释放视频对象
    cap.release()

    # 如果帧数不足8帧
    if frame_count < 8:
        print(f"视频帧数不足8帧，实际帧数: {frame_count}")
        return None, None

    return frames[0], frames[1]  # 返回第1帧和第8帧


def process_video_directory(input_dir, output_dir):
    """
    处理视频目录下的所有视频文件，提取特征并保存
    Args:
        input_dir: 输入视频目录
        output_dir: 输出特征目录
    """
    import os
    import datetime
    from tqdm import tqdm

    print(f"\n处理视频目录:")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}\n")

    # 创建日志目录和文件
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(log_dir, f"{current_time}.log")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有视频文件
    print("正在扫描视频文件...")
    video_files = []
    for root, dirs, files in tqdm(list(os.walk(input_dir)), desc="扫描目录"):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                rel_path = os.path.relpath(root, input_dir)
                video_files.append((os.path.join(root, file), rel_path))

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"找到 {len(video_files)} 个视频文件\n")
    print(f"\n找到 {len(video_files)} 个视频文件")

    # 处理每个视频
    print("\n开始处理视频...")
    for video_path, rel_path in tqdm(video_files, desc="处理进度"):
        try:
            # 构建保存路径
            save_dir = os.path.join(output_dir, rel_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir, os.path.splitext(os.path.basename(video_path))[0] + ".pt"
            )

            # 检查特征文件是否已存在且有效
            if os.path.exists(save_path):
                try:
                    existing_features = torch.load(save_path)
                    if len(existing_features) > 0 and all(
                        k in existing_features for k in [1, 8]
                    ):
                        continue  # 跳过已存在且有效的特征文件
                except:
                    pass  # 如果加载失败，则重新提取特征

            # 获取第1帧和第8帧
            frame1, frame8 = get_video_frames(video_path)
            if frame1 is None or frame8 is None:
                error_msg = f"跳过视频 {video_path}: 帧数不足"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(error_msg + "\n")
                print(error_msg)
                continue

            # 提取特征
            features1 = extract_mm_features(frame1)
            features8 = extract_mm_features(frame8)

            # 组织特征
            video_mm_features = {1: features1, 8: features8}

            # 保存特征
            torch.save(video_mm_features, save_path)

        except Exception as e:
            error_msg = f"处理视频 {video_path} 时出错: {str(e)}"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(error_msg + "\n")
            print(error_msg)
            continue

    complete_msg = "\n处理完成!"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(complete_msg + "\n")
    print(complete_msg)


def list_pt_files(base_dir):
    """列出指定目录下所有的.pt文件并按自然顺序排序

    Args:
        base_dir (str): 基础目录路径

    Returns:
        list: 排序后的.pt文件列表
    """
    import re
    from pathlib import Path

    # 构建完整路径
    target_dir = Path(base_dir)
    # 获取所有.pt文件
    pt_files = list(target_dir.glob("*.pt"))

    # 自然排序
    def natural_sort_key(path):
        # 提取文件名中的数字部分用于排序
        numbers = re.findall(r"\d+", path.name)
        if numbers:
            return int(numbers[0])
        return path.name

    # 按自然顺序排序
    sorted_files = sorted(pt_files, key=natural_sort_key)

    return [str(f) for f in sorted_files]


if __name__ == "__main__":
    input_dir = "/root/proj/MM-Det/data/genvideo/val"
    output_dir = "/root/proj/MM-Det/features/mm/genvideo/val"
    process_video_directory(input_dir, output_dir)

    # base_dir = (
    #     "/U_20240905_ZSH_SMIL/jyj/proj/MM-Det/features/mm/genvideo/val/fake/Crafter"
    # )
    # pt_files = list_pt_files(base_dir)
    # print('\n'.join(pt_files))
