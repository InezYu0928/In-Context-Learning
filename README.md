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

The trained model checkpoints are available at [this file](https://drive.google.com/drive/folders/1-Z2-lJMQ8QjQIVaV0eJdDPlQtBxUpRec?usp=drive_link).

## In-context inference with noisy labels (Section 5)

To explore how different noise levels and types influence the [pretrained ICL model](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip)'s performance in inference stage, you can manipulate `eval_inference.ipynb` to load the models, adjust the noise, plot their accuracy performance, and evaluate them on new data.


## Noisy inference with Noisy training (Section 6)
To test the performance of inference under different noise levels, first download all trained models [here](https://drive.google.com/drive/folders/1-Z2-lJMQ8QjQIVaV0eJdDPlQtBxUpRec?usp=drive_link) into /src/models.
Then you can run `eval_inference_noise.ipynb` to see the results.

## Extrapolating beyond *i.i.d.* noises (Section 7)
To test the performance of inference with outliers in prompts, first download all trained models [here](https://drive.google.com/drive/folders/1-Z2-lJMQ8QjQIVaV0eJdDPlQtBxUpRec?usp=drive_link) into /src/models. Run ``python eval_outliers.py`` to generate metrics then run `eval_outliers.ipynb` to draw the plots with computed metrics.

## Acknowledgement

This repository is forked from [this link](https://github.com/dtsip/in-context-learning).
