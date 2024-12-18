import argparse
import cv2
import numpy as np
import os
import torch
from einops import rearrange
from tqdm import tqdm
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image

from models import VectorQuantizedVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-root", type=str, default="data", help="data root for videos"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/reconstruction_dataset/",
        help="output path",
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=["mp4", "avi", "mpeg", "wmv", "mov", "flv"],
        help="target extensions",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="weights/vqvae/model.pth",
        help="checkpoint path for vqvae",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device for reconstruction model"
    )
    return parser.parse_args()


def denormalize_batch_t(img_t, mean, std):
    try:
        assert len(mean) == len(std)
        assert len(mean) == img_t.shape[1]
    except:
        print(
            f"Unmatched channels between image tensors and normalization mean and std. Got {img_t.shape[1]}, {len(mean)}, {len(std)}."
        )
    img_denorm = torch.empty_like(img_t)
    for t in range(img_t.shape[1]):
        img_denorm[:, t, :, :] = (img_t[:, t, :, :].clone() * std[t]) + mean[t]
    return img_denorm


if __name__ == "__main__":
    args = parse_args()
    cls_folders = list(
        filter(
            lambda p: os.path.isdir(os.path.join(args.data_root, p)),
            sorted(os.listdir(args.data_root)),
        )
    )
    recons_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    print(f"Find {len(cls_folders)} classes: {cls_folders}")
    model = VectorQuantizedVAE(3, 256, 512)
    state_dict = torch.load("weights/vqvae/model.pt", map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    for idx, cls_folder in enumerate(cls_folders, 1):
        os.makedirs(os.path.join(args.output, cls_folder, "original"), exist_ok=True)
        video_list = list(
            filter(
                lambda fn: os.path.splitext(fn)[-1].lower()[1:] in args.ext,
                sorted(os.listdir(os.path.join(args.data_root, cls_folder))),
            )
        )
        for video in tqdm(video_list, desc="Extracting original frames"):
            video_path = os.path.join(args.data_root, cls_folder, video)
            vc = cv2.VideoCapture(video_path)
            if vc.isOpened():
                rval, frame = vc.read()
            else:
                rval = False
            c = 1
            while rval:
                cv2.imwrite(
                    os.path.join(
                        args.output,
                        cls_folder,
                        "original",
                        f"{os.path.splitext(os.path.basename(video))[0]}_{c}.jpg",
                    ),
                    frame,
                )
                c = c + 1
                rval, frame = vc.read()
            vc.release()
        os.makedirs(os.path.join(args.output, cls_folder, "recons"), exist_ok=True)
        for image_fn in tqdm(
            sorted(os.listdir(os.path.join(args.output, cls_folder, "original"))),
            desc="Reconstruction",
        ):
            img = Image.open(
                os.path.join(args.output, cls_folder, "original", image_fn)
            ).convert("RGB")
            img = recons_transform(img).unsqueeze(0)
            with torch.no_grad():
                img = img.to(args.device)
                recons, _, _ = model(img)
                if recons.shape != img.shape:
                    recons = F.interpolate(
                        recons, (img.shape[-2], img.shape[-1]), mode="nearest"
                    )
                denorm_recons = denormalize_batch_t(
                    recons, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                )
                denorm_recons = rearrange(
                    denorm_recons.squeeze(0).cpu().numpy(), "c h w -> h w c"
                )
                recons_img = np.uint8(denorm_recons * 255.0)
                Image.fromarray(recons_img).save(
                    os.path.join(args.output, cls_folder, "recons", image_fn)
                )
        print(f"Finished {idx}/{len(cls_folders)}")
    print(f"Finished. Results saved at {args.output}")
