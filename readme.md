# SSM-Crossformer: Attention Mechanisms based on Spatial-Spectral-Masking in Crossformer for Multidimensional Time Series Forcasting

This is the origin Pytorch implementation of [SSMCrossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting](https://openreview.net/forum?id=vSVLM2j9eie).

## Point of difference between the architecture of SSM-Crossformer and Crossformer








**. Spatial-Spectral-Masking-Attention (SSMA)**
<p align="center">
<img src=".\pic\ssmAttention.PNG" height = "200" alt="" align=center />

<b>Figure 3.</b> SSSMA uses the principle of Spectral Spatial Maskage (SSM) to pay more attention to strong dependencies between variables, based on the averaging of these elements.  All variables with a value greater than or equal to the mean will have a higher attention score..
</p>  

## Requirements

- Python 3.7.10
- numpy==1.20.3
- pandas==1.3.2
- torch==1.8.1
- einops==0.4.1

## Reproducibility
1. Put datasets to conduct experiments into folder `datasets/`. We have already put `ILI` into it.  `ILI` and  can be downloaded from https://github.com/thuml/Autoformer. Note that the `WTH` we used in the paper is the one with 12 dimensions from Informer, not the one with 21 dimensions from Autoformer.

2. To get results of SSM-Crossformer with $T=168, \tau = 24, L_{seg} = 6$ on ETTh1 dataset, run:
```
python main_ssmcrossformer.py --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --itr 1
```
The model will be automatically trained and tested. The trained model will be saved in folder `checkpoints/` and evaluated metrics will be saved in folder `results/`.

3. You can also evaluate a trained model by running:
```
python eval_ssmcrossformer.py --checkpoint_root ./checkpoints --setting_name SSM-Crossformer_ETTh1_il168_ol24_sl6_win2_fa10_dm256_nh4_el3_itr0
```

4. To reproduce all results in the paper, run following scripts to get corresponding results:
```
bash scripts/ETTh1.sh
bash scripts/ETTm1.sh
bash scripts/WTH.sh
bash scripts/ECL.sh
bash scripts/ILI.sh
bash scripts/Traffic.sh
```







`main_ssmcrossformer` is the entry point of our model and there are other parameters that can be tuned. Here we describe them in detail:
| Parameter name | Description of parameter |
| --- | --- |
| data           | The dataset name                                             |
| root_path      | The root path of the data file (defaults to `./datasets/`)    |
| data_path      | The data file name (defaults to `ETTh1.csv`)                  |
| data_split | Train/Val/Test split, can be ratio (e.g. `0.7,0.1,0.2`) or number (e.g. `16800,2880,2880`), (defaults to `0.7,0.1,0.2`) 
| checkpoints    | Location to store the trained model (defaults to `./checkpoints/`)  |
| in_len | Length of input/history sequence, i.e. $T$ in the paper (defaults to 96) |
| out_len | Length of output/future sequence, i.e. $\tau$ in the paper (defaults to 24) |
| seg_len | Length of each segment in DSW embedding, i.e. $L_{seg}$ in the paper (defaults to 6) |
| win_size | How many adjacent segments to be merged into one in segment merging of HED  (defaults to 2) |
| factor | Number of routers in Cross-Dimension Stage of TSA, i.e. $c$ in the paper (defaults to 10) |
| data_dim | Number of dimensions of the MTS data, i.e. $D$ in the paper (defaults to 7 for ETTh and ETTm) |
| d_model | Dimension of hidden states, i.e. $d_{model}$ in the paper (defaults to 256) |
| d_ff | Dimension of MLP in MSA (defaults to 512) |
| n_heads | Num of heads in MSA (defaults to 4) |
| e_layers | Num of encoder layers, i.e. $N$ in the paper (defaults to 3) |
| dropout | The probability of dropout (defaults to 0.2) |
| num_workers | The num_works of Data loader (defaults to 0) |
| batch_size | The batch size for training and testing (defaults to 32) |
| train_epochs | Train epochs (defaults to 20) |
| patience | Early stopping patience (defaults to 3) |
| learning_rate | The initial learning rate for the optimizer (defaults to 1e-4) |
| lradj | Ways to adjust the learning rate (defaults to `type1`) |
| itr | Experiments times (defaults to 1) |
| save_pred | Whether to save the predicted results. If True, the predicted results will be saved in folder `results` in numpy array form. This will cost a lot time and memory for datasets with large $D$. (defaults to `False`). |
| use_gpu | Whether to use gpu (defaults to `True`) |
| gpu | The gpu no, used for training and inference (defaults to 0) |
| use_multi_gpu | Whether to use multiple gpus (defaults to `False`) |
| devices | Device ids of multile gpus (defaults to `0,1,2,3`) |

## Citation
If you find this repository useful in your research, please cite:
```

```


## Acknowledgement
We appreciate the following works for their valuable code and data for time series forecasting:

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/alipay/Pyraformer

https://github.com/MAZiqing/FEDformer

https://github.com/google-research/vision_transformer

https://github.com/microsoft/Swin-Transformer

We would like to thank the authors of this work for the availability of their code. And also for allowing us to improve the attention mechanism used in this model, in order to optimize forcasting performance.

https://github.com/Thinklab-SJTU/Crossformer

