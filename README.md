# [NeurIPS 2024] On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection

This repository is the official implementation of [MM-Det](https://arxiv.org/abs/2410.23623) [NeurIPS 2024 Poster].

[![arxiv](https://img.shields.io/badge/arXiv-2310.23623-b31b1b.svg)](https://arxiv.org/abs/2410.23623)

<table class="center">
    <tr>
    <td><img src="assets/overview_of_main_method.png" width="600">></td>
    </tr>
</table>


## Contents

- [\[NeurIPS 2024\] On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection](#neurips-2024-on-learning-multi-modal-forgery-representation-for-diffusion-generated-video-detection)
  - [Contents](#contents)
  - [Environment](#environment)
  - [Diffusion Video Forensics Dataset](#diffusion-video-forensics-dataset)
  - [Preparation](#preparation)
    - [Pre-trained Weights](#pre-trained-weights)
    - [Reconstruction Dataset](#reconstruction-dataset)
    - [Multi-Modal Forgery Representation](#multi-modal-forgery-representation)
  - [Evaluation](#evaluation)
  - [Training](#training)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Environment

1. Install basic packages
```bash
conda create -n MM_Det python=3.10
conda activate 
pip install -r requirements.txt
cd LLaVA
pip install -e .
```
2. For training cases, install additional packages
```bash
cd LLaVA
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn==2.5.8 --no-build-isolation
```

## Diffusion Video Forensics Dataset
We release Diffusion Video Forensics (DVF) as the benchmark for forgery video detection, which can be downloaded via links: [BaiduNetDisk](https://pan.baidu.com/s/14d-_jLB_yUwKzOosrMHMvg?pwd=296c)(Code: 296c). [google driver](https://drive.google.com/drive/folders/1NxCvJVPSxV2Mib5NaNj5Cf2WnnjrqpMb?usp=drive_link)

<table class="center">
    <tr>
    <td><img src="assets/dvf_dataset_statistics.png"></td>
    </tr>
</table>

## Preparation

### Pre-trained Weights
We provide the [weights](https://huggingface.co/sparklexfantasy/llava-7b-1.5-rfrd) for our fine-tuned large multi-modal model, which is based on llava-v1.5-Vicuna-7b from [LLaVA](https://github.com/haotian-liu/LLaVA). The overall weights for MM-Det without the LMM can be obtained via [weights](https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing) at `MM-Det/current_model.pth`. Please download and put the weights at `./weights/`.

### Reconstruction Dataset

For all videos in DVF, we provide a ready reconstruction dataset at [BaiduNetDisk](https://pan.baidu.com/s/1a0sWzGXfkBfblV1wZ70qsQ?pwd=l8h4) (Code: l8h4).

### Multi-Modal Forgery Representation

For all videos in DVF, we provide a ready dataset for cached MMFR at [BaiduNetDisk](https://pan.baidu.com/s/1kuybGikCfxs8CnTnxWI-gQ?pwd=m6uy) (Code: m6uy). Since the representation is fixed during training and inference, it is recommended to cache the representation before the overall training to reduce time cost. 

## Evaluation

Set `$RECONSTRUCTION_DATASET_ROOT` as `DVF_recons` and `$MM_REPRESENTATION_ROOT` as `mm_representations`. Set `$RECONSTRUCTION_DATASET_ROOT` and `$MM_REPRESENTATION_ROOT` to both data roots. `--cache-mm` is also recommended for save the computational and memory cost of LMM branch.
Make sure the pre-trained weights are organized at `./weights`. Then, run the following script for testing on 7 datasets respectively. 
```bash
python test.py \
--classes videocrafter1 zeroscope opensora sora pika stablediffusion stablevideo \
--ckpt ./weights/MM-Det/current_model.pth \
--data-root $RECONSTRUCTION_DATASET_ROOT \
--cache-mm \
--mm-root $MM_REPRESENTATION_ROOT\
# when sample-size > 0, only [sample-size] videos are evaluated for each dataset.
--sample-size -1
```

Since the entire evaluation is time-costing, `sample-size` can be specified (e.g., 1,000) to reduce time by conducting inference only on limited (1,000) videos. To finish the entire evaluation, please set `sample-size` as `-1`.

## Training

Known Issues:

- From the feedback we’ve received, we noticed a deviation in the training process when fine-tuning the large language model, making it difficult to reproduce our reported results fully in some cases. We are now resolving this issue and will share the updated training scripts soon. Currently, we provide the inference interface first.


## Acknowledgement

We express our sincere appreciation to the following projects.

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [pytorch-vqvae](https://github.com/ritheshkumar95/pytorch-vqvae)
- [Stable Diffusion](https://github.com/comfyanonymous/ComfyUI)
- [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter)
- [Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w)
- [OpenSora](https://github.com/hpcaitech/Open-Sora)
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).

## Citation

```
@misc{song2024learningmultimodalforgeryrepresentation,
      title={On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection}, 
      author={Xiufeng Song and Xiao Guo and Jiache Zhang and Qirui Li and Lei Bai and Xiaoming Liu and Guangtao Zhai and Xiaohong Liu},
      year={2024},
      eprint={2410.23623},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.23623}, 
}
```