


#  We discovered a bug and have temporarily closed the open source code. We will fix it soon















# LF-ViT: Reducing Spatial Redundancy in Vision Transformer for Efficient Image Recognition

This is Pytorch implementation of our paper "LF-ViT: Reducing Spatial Redundancy in Vision Transformer for Efficient Image Recognition".

## Pre-trained Models


- What are contained in the checkpoints:

```
**.pth
├── model: state dictionaries of the model
├── flops: a list containing the GFLOPs corresponding to exiting at each stage
├── anytime_classification: Top-1 accuracy of each stage
├── budgeted_batch_classification: results of budgeted batch classification (a two-item list, [0] and [1] correspond to the two coordinates of a curve)

```

## Requirements
- python 3.9.7
- pytorch 1.10.1
- torchvision 0.11.2

## Data Preparation
- The ImageNet dataset should be prepared as follows:
```
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...

```
## Evaluate Pre-trained Models
- Get accuracy of each stage
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 0 --data_url PATH_TO_IMAGENET  --batch_size 64 --model lf_deit_small --checkpoint_path PATH_TO_CHECKPOINT  --location-stage-size {7,9} 

```

- Infer the model on the validation set with various threshold([0.01:1:0.01])
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 1 --data_url PATH_TO_IMAGENET  --batch_size 64 --model lf_deit_small --checkpoint_path PATH_TO_CHECKPOINT  --location-stage-size {7,9} 

```

- Infer the model on the validation set with one threshold and meature the throughput

```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 2 --data_url PATH_TO_IMAGENET  --batch_size 1024 --model lf_deit_small --checkpoint_path PATH_TO_CHECKPOINT  --location-stage-size {7,9} --threshold THRESHOLD

```

- Read the evaluation results saved in pre-trained models
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 3 --data_url PATH_TO_IMAGENET  --batch_size 64 --model lf_deit_small --checkpoint_path PATH_TO_CHECKPOINT  --location-stage-size {7,9} 

```

## Train
- Train LF-ViT on ImageNet 
```
python -m torch.distributed.launch --use_env --nproc_per_node=4 main_deit.py  --model lf_deit_small --batch-size 256 --data-path PATH_TO_IMAGENET --location-stage-size {7,9} --dist-eval --output PATH_TO_LOG

```



## Visualization
- Visualization of images correctly classified at location stage and focus stage.

```
python visualize.py --model lf_deit_small --resume  PATH_TO_CHECKPOINT --output_dir PATH_TP_SAVE --data-path PATH_TO_IMAGENET --batch-size 64 

```


## Acknowledgment
Our code of LFVisionTransformer is from [here](https://github.com/JER-ry/CF-ViT). Our code of DeiT is from [here](https://github.com/facebookresearch/deitzhe). The visualization code is modified from [Evo-ViT](https://github.com/YifanXu74/Evo-ViT). The dynamic inference with early-exit code is modified from [DVT](https://github.com/blackfeather-wang/Dynamic-Vision-Transformer). Thanks to these authors. 

