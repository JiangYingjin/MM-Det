# %%
import torch

# 从文件中读取特征
features = torch.load(
    "/U_20240905_ZSH_SMIL/jyj/proj/MM-Det/features/mm/genvideo/val/real/MSR-VTT/MSRVTT_400.pt"
)

# 打印特征尺寸
print("特征信息:")
print("第1帧:")
print("- 视觉特征 shape:", features[1]["visual"].shape)
print("- 文本特征 shape:")
for layer, feat in features[1]["textual"].items():
    print(f"  Layer {layer}:", feat.shape)

print("\n第8帧:")
print("- 视觉特征 shape:", features[8]["visual"].shape)
print("- 文本特征 shape:")
for layer, feat in features[8]["textual"].items():
    print(f"  Layer {layer}:", feat.shape)
