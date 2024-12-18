# %%
import torch

# 从文件中读取特征
features = torch.load("out/mm_feat1.pt")

# 打印特征信息
print("视觉特征 shape:", features["visual"].shape)
print("\n文本特征信息:")
for layer, feat in features["textual"].items():
    print(f"- Layer {layer} shape:", feat.shape)

print(features)
