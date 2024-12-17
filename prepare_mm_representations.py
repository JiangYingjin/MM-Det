import torch
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoProcessor
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage
from models import MMEncoder
from options.base_options import BaseOption

from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from LLaVA.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    get_anyres_image_grid_shape,
)


def parse_args(opt):
    parser = opt.parser
    parser.add_argument(
        "--cached-data-root",
        type=str,
        default="./data/dvf_recons",
        help="the data root for frames in the reconstruction structure",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/mm_representations",
        help="the output directory",
    )
    parser.add_argument(
        "--output-fn",
        type=str,
        default="mm_representation.pth",
        help="the output file name",
    )
    return parser.parse_args()


def get_dataset_meta(dataset_path):
    fns = sorted(os.listdir(dataset_path))
    meta = {}
    for fn in fns:
        data_id = fn.rsplit("_", maxsplit=1)[0]
        if data_id not in meta:
            meta[data_id] = 1
        else:
            meta[data_id] += 1
    return meta


def sample_by_interval(frame_count, interval=200):
    sampled_index = []
    count = 1
    while count <= frame_count:
        sampled_index.append(count)
        count += interval
    return sampled_index


if __name__ == "__main__":
    opt = BaseOption()
    config = parse_args(opt).__dict__
    output_dir = config["output_dir"]
    output_fn = config["output_fn"]
    config["lmm_ckpt"] = "sparklexfantasy/llava-1.5-7b-rfrd"
    config["load_4bit"] = False
    model = MMEncoder(config)
    visual_model = model.model.get_vision_tower().vision_tower
    input_data_root = config["cached_data_root"]
    cls_folder = sorted(os.listdir(input_data_root))  # each dataset cls
    cls_folder = list(
        filter(lambda x: os.path.isdir(os.path.join(input_data_root, x)), cls_folder)
    )
    print(f"Find {len(cls_folder)} classes: {cls_folder}")
    with torch.inference_mode():
        for cls_idx, sub_cls in enumerate(cls_folder, 1):
            labels = sorted(
                os.listdir(os.path.join(input_data_root, sub_cls))
            )  # [0_real, 1_fake]
            for label in labels:
                if not os.path.isdir(os.path.join(input_data_root, sub_cls, label)):
                    continue
                os.makedirs(os.path.join(output_dir, sub_cls, label), exist_ok=True)
                dataset_meta = get_dataset_meta(
                    os.path.join(input_data_root, sub_cls, label, "original")
                )
                mm_representations = {}
                for data_id, count in tqdm(dataset_meta.items()):
                    sampled_index = sample_by_interval(count, config["interval"])
                    for index in sampled_index:
                        img = Image.open(
                            os.path.join(
                                input_data_root,
                                sub_cls,
                                label,
                                "original",
                                f"{data_id}_{index}.jpg",
                            )
                        ).convert("RGB")
                        visual_features, mm_features = model(img)
                        mm_layer_features = {}
                        for idx, layer in enumerate(model.selected_layers):
                            mm_layer_features[str(layer)] = mm_features[idx].cpu()
                        mm_representations[f"{data_id}_{index}.jpg"] = {
                            "visual": visual_features.squeeze(0).cpu(),
                            "textual": mm_layer_features,
                        }
                torch.save(
                    mm_representations,
                    os.path.join(output_dir, sub_cls, label, output_fn),
                )
            print(f"Finished {cls_idx}/{len(cls_folder)}")
