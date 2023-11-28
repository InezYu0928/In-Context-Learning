# On the Robustness of In-Context Learning with Noisy Labels: Train, Inference, and Beyond

This is the repository for our paper **On the Robustness of In-Context Learning with Noisy Labels: Train, Inference, and Beyond**, 

with Cheng Chen, Haodong Wen, Xinzhi Yu, Zeming Wei.

Paper link: TBD

## Train transformers with noisy labels (Section 4)
To train a transformer with demonstrations of $\sigma$=``std``, use
```
python train.py --config conf/NL_train_{std}.yaml
```
Note that you should modify the ``wandb`` configurations in the corresponding config file.

The trained model checkpoints are available at https://drive.google.com/drive/folders/1-Z2-lJMQ8QjQIVaV0eJdDPlQtBxUpRec?usp=drive_link.

## In-context inference with noisy labels (Section 5)

[pretrained model](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip)


## Noisy inference with Noisy training (Section 6)

## Extrapolating beyond *i.i.d.* noises (Section 7)


## Acknowledgement

This repository is forked from [this link](https://github.com/dtsip/in-context-learning).
