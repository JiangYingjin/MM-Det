import torch
from PIL import Image
from models import MMEncoder
from options.base_options import BaseOption
from LLaVA.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def extract_mm_features(image_path):
    """
    提取单张图片的多模态特征
    Args:
        image_path: 输入图像路径
    Returns:
        dict: 包含visual和textual特征的字典
    """
    # 配置
    opt = BaseOption()
    config = opt.parse().__dict__

    # 加载模型
    config["lmm_ckpt"] = "sparklexfantasy/llava-7b-1.5-rfrd"
    config["load_4bit"] = False
    model = MMEncoder(config)
    model.eval()

    # 处理图像
    with torch.inference_mode():
        img = Image.open(image_path).convert("RGB")
        visual_features, mm_features = model(img)

        # 提取特征
        mm_layer_features = {}
        for idx, layer in enumerate(model.selected_layers):
            mm_layer_features[str(layer)] = mm_features[idx].cpu()

        # 打印特征信息
        print("视觉特征 shape:", visual_features.squeeze(0).cpu().shape)
        print("文本特征信息:")
        for layer, feat in mm_layer_features.items():
            print(f"- Layer {layer} shape:", feat.shape)

        features = {
            "visual": visual_features.squeeze(0).cpu(),
            "textual": mm_layer_features,
        }

        return features


if __name__ == "__main__":
    # 测试用例
    image_path = "/root/proj/MM-Det/data/WallStreet_01.jpg"
    features = extract_mm_features(image_path)
