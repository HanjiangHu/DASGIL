# DASGIL: Domain Adaptation for Semantic and Geometric-aware Image-based Localization

This is our Pytorch implementation for DASGIL ([arxiv](https://arxiv.org/pdf/2010.00573.pdf)) by [Hanjiang Hu](https://github.com/HanjiangHu) and [Ming Cheng](https://mingcheng991129.github.io/).


<img src='img/overview.png' align="center" width=666 alt="Text alternative when image is not available">


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install requisite Python libraries.
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/HanjiangHu/DASGIL.git
```

### Training

[KITTI](http://www.cvlibs.net/datasets/kitti/index.php) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-2/) dataset are used to train the model, while [Extended CMU-Seasons](https://www.visuallocalization.net/datasets) dataset is used to test.
The datasets involved in this paper are well organized [HERE](). Please uncompress it under the root path. Our pretrained models are found [HERE](). Please uncompress it under the root path.

- Training on KITTI and Virtual KITTI Dataset:
```
python train.py --name DASGIL
```
- Fine-tune the pretrained model:
```
python train.py --name DASGIL --continue_train --which_epoch 200
```
### Testing
- Testing on the Extended CMU-Seasons Dataset:
```
python test.py --name DASGIL --which_epoch 200
```
### Results
The test results will be saved to `./output`. The txt results should be merged into a single txt file and submitted to [the official benchmark website](https://www.visuallocalization.net/submission/).

Our [DASGIL results](https://www.visuallocalization.net/details/3479/) on Extended CMU-Seasons Dataset could be found on the benchmark website.


## Other Details
- See `./options/train_options.py` for training-specific flags, `./options/test_options.py` for test-specific flags, and `./options/base_options.py` for all common flags.
- CPU/GPU (default `--gpu_ids 0`): set`--gpu_ids -1` to use CPU mode (NOT recommended); set `--gpu_ids 0,1,2` for multi-GPU mode.

If you use this code in your own work, please cite:

H. Hu, M. Cheng, Z. Liu and H. Wang
”[DASGIL: Domain Adaptation for Semantic and Geometric-aware Image-based Localization](https://arxiv.org/pdf/2010.00573.pdf)”,  

```
@misc{hu2020dasgil,
      title={DASGIL: Domain Adaptation for Semantic and Geometric-aware Image-based Localization}, 
      author={Hanjiang Hu and Ming Cheng and Zhe Liu and Hesheng Wang},
      year={2020},
      eprint={2010.00573},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
