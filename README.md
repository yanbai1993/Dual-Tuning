Dual-Tuning: Joint Prototype Transfer and Structure Regularization for Compatible Feature Learning

Our paper See [arxiv](https://arxiv.org/abs/2108.02959).

The complete code will be released soon.

# Installation

Our code is based on FastReID. See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).
For apex used in FastReID, we install it based on its source code. 
```shell script
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
```

# Quick Start
## Prepare pretrained model and dataset

You can download pre-train models manually and put it in `~/.cache/torch/checkpoints`




## Compile with cython to accelerate evalution

```bash
cd fastreid/evaluation/rank_cylib; make all
```

## Training 

First, you need to get an old model, 

```bash
python tools/train_net.py --config-file ./configs/Market1501/part-bagtricks_R18-softmax.yml MODEL.DEVICE "cuda:0"
```

Note that, only a part of training data (e.g. 312 classes for Market1501) will be used for training an old model. We provide partMarket1501 function (data/datasets/partmarket1501.py), and you can define the classes number used for old model.


Second, if you want to perform our center-based prototype loss, you need to obtain the class centers in old embedding space,
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_center.py --config-file ./configs/Market1501/center-part-bagtricks_R18-softmax.yml
```

Third, train the new compatible model, for resnet18 backbone:
```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py --config-file ./configs/Market1501/center-bagtricks_R18-softmax.yml
```

For resnet50 backbone:
```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py --config-file ./configs/Market1501/center-bagtricks_R50-softmax.yml
```

We do not restrict the new architecture to be the same as the old model, and you can choose any other architectures like Resnet, Osnet, ResNeSt, et al. 
Besides, the losses of new and old model can also be different.

We also provide several baselines to achieve feature compatibility for person ReID. 
Our loss include two parts, i.e. metric loss for target task and compatible loss. 
The metric loss can be one loss or a combination of multiple losses (e.g., "CrossEntropyLoss", "TripletLoss", "CircleLoss") 
The compatible loss can also  be one loss or a combination of multiple losses (e.g., "OldCenterLoss", "loss_O2Nfc", "loss_N2Ofc", "MixCenterLoss", "N2OTripletLoss", "loss_KL", "loss_triplet_center","L2Loss").
More details see ./fastreid/modeling/meta_arch/dualtuning.py

## Test
We calculate the self-test results and cross-test results by
```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py --config-file ./configs/Market1501/center-bagtricks_R18-softmax.yml MODEL.WEIGHTS "./logs_final/market1501/R18-softmax_n512_o512_dualtuning/model_final.pth" TEST.OLDRATE 100
```
You need to change the config-file and weights path for your model testing. Besides, the TEST.OLDRATE can change the ratio of the old and new features in the gallery. If OLDTATE is 100, all the gallery features are from old model, if OLDRATE is 50, 50% of the features are from old model. 

