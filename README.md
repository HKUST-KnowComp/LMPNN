# Logical Message Passing Networks with One-hop Inference on Atomic Formula

Implementation for ICLR 2023 paper:
> Logical Message Passing Networks with One-hop Inference on Atomic Formula

see the [arXiv version](https://arxiv.org/abs/2301.08859) and the [OpenReview version](https://openreview.net/forum?id=SoyOsp7i_l).

In this documentation, we detail how to reproduce the results in the paper based on existing checkpoints released by other researchers.

## Requirement of this repository
- pytorch
- jupyter
- tqdm

Requirement of other submodules will be discussed accordingly.

My conda environment could be found at `ENV1.yaml`

## Version
** This repo is under construction for usability. The key results for the paper can already be reproduced. **

Todo features:
- [x] Implement CQD CO
- [ ] Implement CQD BEAM
- [ ] Introduce several ways to train KGE checkpoints with released repositories, which can be backbones for LMPNN.

## Preparation

### (A) Prepare the dataset
Please download the dataset from [snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning).

Specifically, one can run:
```bash
mkdir data
cd data
wget http://snap.stanford.edu/betae/KG_data.zip # a zip file of 1.3G
unzip KG_data.zip
```

Then the `data` folder will contain the following folders and files:
```
FB15k-237-betae
FB15k-237-q2b
FB15k-betae
FB15k-q2b
KG_data.zip
NELL-betae
NELL-q2b
```

We rearange them into different subfolders:
```
mkdir betae-dataset
mv *betae betae-dataset
mkdir q2b-dataset
mv *q2b q2b-dataset
```

Finally, run `convert_beta_dataset.py` to convert the data into the graph forms for LMPNN. One can find the new dataset folders in `./data`.

An example converted dataset format is
```
data/FB15k-237-betae
  - kgindex.json
  - train_kg.tsv
  - valid_kg.tsv
  - test_kg.tsv
  - train-qaa.json
  - valid-qaa.json
  - test-qaa.json
```
where
- `kgindex.json` file stores the relation/entity names and their ids,
- `{train/valid/test}_kg.tsv` stores the triples in three knowledge graphs (triples in `train_kg.tsv` is the subset of those in `valid_kg.tsv`, and triples `valid_kg.tsv` is also the subset of those in `test_kg.tsv`)
- `{train/valid/test}-qaa.json` stores the Query, easy Answers and hard Answers for train, valid, and test set.

### (B1) Pretrain KGE checkpoints with external submodules

We consider two different repositories to pretrain the KGE checkpoints.
Including
1. [uma-pi1/kge](https://github.com/uma-pi1/kge)
2. [facebookresearch/ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction)

To initialize these modules, please run
```bash
git submodule update
```

How to train the checkpoints with these submodules is discussed in this section.

Generally, there are two steps, once the KG is prepared:
1. Convert the KG triples into the format that can be used in each submodule.
2. Train the checkpoints.

#### Choice 1: Pretrain with [uma-pi1/kge](https://github.com/uma-pi1/kge)

##### Step 0: Prepare the environment and config

To run `libkge` submodule, one need editable installation.

```bash
cd kge
pip install -e .
```

##### Step 1: Prepare the dataset

Running the following code to convert BetaE dataset into `./kge/data`.

```sh
python convert_kg_data_for_kge.py
```

##### Step 2: Train the checkpoint

We provide a config in `config/kge/fb15k-237-complex.yaml`. Tailor the config to train checkpoints with `libkge`.
```sh
kge start config/kge/fb15k-237-complex.yaml --job.device cuda:0
```

The obtained checkpoints can be found at `kge/local`.


#### Choice 2: Pretrain with [facebookresearch/ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction)


##### Step 0: Prepare the environment
```
pip install ogb networkx wandb
```

##### Step 1: Prepare the dataset

Running the following code to convert BetaE dataset into `./ssl-relation-prediction/data`.

```sh
python convert_kg_data_for_ssl.py
```

Notably, in the `ssl-relation-prediction/src/main.py` file, the command arg parser prohibits external sources of datasets in line 47-50.
```python
parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)
```

Let's change it into
```python
parser.add_argument(
    '--dataset'
)
```

##### Step 2: Train the checkpoint
The training process can be initialized by running
```sh
cd ssl-relation-prediction
python src/main.py --dataset FB15k-237-betae \
               --score_rel True \
               --model ComplEx \
               --rank 1000 \
               --learning_rate 0.1 \
               --batch_size 1000 \
               --lmbda 0.05 \
               --w_rel 4 \
               --max_epochs 100 \
               --model_cache_path ./ckpts/FB15k-237-complex/
```

The obtained checkpoints can be found at `ssl-relation-prediction/ckpts/FB15k-237-complex`.

### (B2) Convert pretrained KGE checkpoints into the acceptable format

We convert external KGE checkpoints into the format that can be loaded by LMPNN. We consider three sources of external checkpoints

1. Checkpoints released by [uclnlp/cqd](https://github.com/uclnlp/cqd).
2. Checkpoints released / pretrained by [uma-pi1/kge](https://github.com/uma-pi1/kge)
2. Checkpoints released by [facebookresearch/ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction)


The pretrained checkpoints are managed in the folder `pretrain`.
```sh
mkdir pretrain
```

#### Sources 1. [uclnlp/cqd](https://github.com/uclnlp/cqd)

This source of checkpoints is used to repreduced the results shown in the paper.

```bash
cd pretrain
wget http://data.neuralnoise.com/cqd-models.tgz # a .tgz file of 4.8G
tar xvf cqd-models.tgz
mv models raw_cqd_pretrain_models
```

Then we can convert the checkpoints into the format used in this repo.
```bash
python convert_cqd_pretrain_ckpts.py
```

## Train LMPNN


Sample usage at FB15k-237
```bash
python3 train_lmpnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/FB15k-237/lmpnn-complex1k-default \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --embedding_dim 1000 \
  --device cuda:0
```

Sample usage at FB15k
```bash
python3 train_lmpnn.py \
  --task_folder data/FB15k-betae \
  --checkpoint_path pretrain/cqd/FB15k-model-rank-1000-epoch-100-1602520745.pt \
  --device cuda:1 \
  --hidden_dim 8192 \
  --output_dir log/FB15k/lmpnn-complex1k-default
```

Sample usage at NELL
```bash
python3 train_lmpnn.py \
  --task_folder data/NELL-betae \
  --checkpoint_path pretrain/cqd/NELL-model-rank-1000-epoch-100-1602499096.pt \
  --device cuda:2 \
  --hidden_dim 8192 \
  --temp 0.1 \
  --batch_size 512 \
  --batch_size_eval_dataloader 8 \
  --batch_size_eval_truth_value 1 \
  --output_dir log/NELL/lmpnn-complex1k-default
```

## Answering Existential First Order (EFO) queries

In this repository, the capability of answering EFO-1 queries is implemented by the `reasoner`s.

- CQD-CO is implemented as `GradientEFOReasoner`, which is refered as CQD(E) in the paper.
- LMPNN is implemented as `GNNEFOReasoner` with `LogicalGNNLayer`

```bash
python train_lmpnn.py \
  --reasoner gradient \
  --device cuda:1 \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt
```

## Summarize the results from log files

We use script `read_eval_from_log.py` to summarize the results from the log file.

For example, the results on FB15k-237 in file `log/FB15k-237/pretrain_complex1000-default/output.log` can be summarized by the following piece of code.
```bash
python3 read_eval_from_log.py --log_file log/FB15k-237/pretrain_complex1000-default/output.log
```


Then the code will output the markdown table of the trajectory over valid and test set.

For FB15k-237, the output trajectory could be like

|              |      1p |       2p |       3p |      2i |      3i |      pi |      ip |      2u |      up |     2in |      3in |     inp |     pin |     pni |   epfo mean |   Neg mean |
|:-------------|--------:|---------:|---------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|--------:|--------:|--------:|------------:|-----------:|
| (5, 'mrr')   | 43.3374 |  9.3963  |  7.36107 | 28.8171 | 43.2848 | 12.7592 | 11.338  | 10.9825 | 8.06806 | 4.89454 |  9.03825 | 6.84014 | 3.57948 | 3.43009 |     19.4827 |    5.5565  |
| (10, 'mrr')  | 43.5847 | 10.2061  |  8.38958 | 30.1158 | 44.8668 | 13.7348 | 12.4519 | 11.1331 | 8.40302 | 5.50171 |  9.92851 | 7.19045 | 3.68003 | 3.68664 |     20.3207 |    5.99747 |
| (15, 'mrr')  | 43.1136 |  9.93098 |  8.41377 | 30.7293 | 45.7028 | 14.1373 | 12.4096 | 11.17   | 8.48811 | 5.90808 |  9.97395 | 7.14056 | 3.78556 | 3.88563 |     20.455  |    6.13876 |
| (20, 'mrr')  | 43.6212 | 10.5731  |  8.92229 | 30.8057 | 45.9869 | 15.2658 | 13.1256 | 11.0792 | 8.84129 | 5.74387 | 10.4465  | 7.32727 | 3.87902 | 3.76653 |     20.9135 |    6.23263 |
| (25, 'mrr')  | 43.5993 | 10.6286  |  9.14751 | 30.7887 | 46.2739 | 16.2113 | 13.8994 | 11.2555 | 8.81353 | 5.98244 | 10.6118  | 7.51721 | 3.7699  | 3.55735 |     21.1798 |    6.28774 |
| (30, 'mrr')  | 43.6251 | 10.8747  |  9.03938 | 30.9388 | 46.5563 | 16.4211 | 13.543  | 11.2324 | 9.15598 | 5.83169 | 10.5604  | 7.63332 | 3.9878  | 3.64097 |     21.2652 |    6.33083 |
| (35, 'mrr')  | 43.4778 | 11.0762  |  9.3528  | 31.3629 | 47.202  | 17.1742 | 13.701  | 11.2211 | 9.16062 | 5.97082 | 10.7614  | 7.62482 | 3.90409 | 3.7512  |     21.5254 |    6.40247 |
| (40, 'mrr')  | 43.4053 | 11.0612  |  9.56793 | 31.3726 | 47.2989 | 16.9117 | 13.6576 | 10.8811 | 9.21064 | 5.86364 | 10.6366  | 7.91589 | 3.90675 | 3.52537 |     21.4852 |    6.36965 |
| (45, 'mrr')  | 43.2611 | 11.0465  |  9.57849 | 31.2661 | 47.44   | 17.705  | 13.9036 | 11.0584 | 9.11347 | 5.98817 | 10.6902  | 7.86165 | 3.93147 | 3.75511 |     21.5969 |    6.44531 |
| (50, 'mrr')  | 42.9117 | 11.4134  |  9.45842 | 31.4618 | 47.5948 | 17.8229 | 13.9    | 11.0091 | 9.40493 | 5.89662 | 10.722   | 7.65385 | 4.07289 | 3.60841 |     21.6641 |    6.39074 |
| (55, 'mrr')  | 43.2503 | 11.6637  |  9.83592 | 31.7507 | 48.095  | 19.3408 | 14.4881 | 11.0594 | 9.40362 | 6.04338 | 10.8817  | 8.20831 | 4.18475 | 3.58321 |     22.0986 |    6.58026 |
| (60, 'mrr')  | 43.3788 | 11.7293  | 10.0087  | 31.8454 | 48.1824 | 19.6966 | 14.5867 | 11.0317 | 9.52452 | 6.06797 | 10.8322  | 8.18048 | 4.23077 | 3.65369 |     22.2205 |    6.59303 |
| (65, 'mrr')  | 43.265  | 11.6121  | 10.0383  | 32.0212 | 48.3501 | 19.8917 | 14.6462 | 11.0356 | 9.44041 | 6.03938 | 10.9391  | 8.08408 | 4.3106  | 3.65148 |     22.2556 |    6.60492 |
| (70, 'mrr')  | 43.2776 | 11.6904  |  9.98802 | 32.0637 | 48.5154 | 20.0188 | 14.7324 | 11.0485 | 9.54078 | 5.98941 | 11.0414  | 8.21996 | 4.22696 | 3.7204  |     22.3195 |    6.63963 |
| (75, 'mrr')  | 43.3091 | 11.7581  | 10.0377  | 32.1079 | 48.6767 | 20.1973 | 14.8317 | 11.0657 | 9.52549 | 6.03904 | 11.0798  | 8.28582 | 4.34577 | 3.63607 |     22.39   |    6.6773  |
| (80, 'mrr')  | 43.3248 | 11.7603  |  9.98525 | 32.2027 | 48.5901 | 20.2498 | 14.8366 | 11.0489 | 9.5386  | 5.98985 | 11.0667  | 8.26597 | 4.35415 | 3.66858 |     22.393  |    6.66906 |
| (85, 'mrr')  | 43.2465 | 11.7877  | 10.0583  | 32.1016 | 48.7363 | 20.4814 | 14.9262 | 10.9962 | 9.51707 | 6.02979 | 11.2297  | 8.3184  | 4.39278 | 3.72155 |     22.4279 |    6.73846 |
| (90, 'mrr')  | 43.376  | 11.6823  | 10.0461  | 32.241  | 48.6458 | 20.4281 | 14.8565 | 11.0369 | 9.5371  | 6.03664 | 11.1348  | 8.34831 | 4.32066 | 3.62985 |     22.4278 |    6.69404 |
| (95, 'mrr')  | 43.291  | 11.7701  |  9.97534 | 32.3046 | 48.8179 | 20.5445 | 14.8694 | 11.1149 | 9.53749 | 6.02634 | 11.2056  | 8.25653 | 4.34473 | 3.6933  |     22.4695 |    6.70531 |
| (100, 'mrr') | 43.2896 | 11.8108  |  9.95793 | 32.3719 | 48.9575 | 20.4634 | 14.8929 | 11.0351 | 9.56184 | 6.06656 | 11.1147  | 8.34275 | 4.35363 | 3.69552 |     22.4823 |    6.71463 |


|              |      1p |      2p |       3p |      2i |      3i |      pi |      ip |      2u |       up |     2in |     3in |     inp |     pin |     pni |   epfo mean |   Neg mean |
|:-------------|--------:|--------:|---------:|--------:|--------:|--------:|--------:|--------:|---------:|--------:|--------:|--------:|--------:|--------:|------------:|-----------:|
| (5, 'mrr')   | 44.9465 | 10.4843 |  8.6229  | 31.7221 | 44.4341 | 14.2661 | 14.1401 | 13.6675 |  9.80541 | 6.86755 | 11.2436 | 6.54754 | 3.84052 | 4.55932 |     21.3432 |    6.61171 |
| (10, 'mrr')  | 45.1145 | 11.2116 |  9.19404 | 32.7899 | 46.5547 | 15.7508 | 15.1701 | 13.5092 |  9.95767 | 7.41947 | 11.6497 | 6.58225 | 3.93812 | 4.52801 |     22.1392 |    6.8235  |
| (15, 'mrr')  | 44.7565 | 11.6183 |  9.65737 | 33.0242 | 46.9324 | 16.0322 | 15.6126 | 13.9755 | 10.2088  | 7.83579 | 12.0131 | 6.88489 | 4.00784 | 4.6131  |     22.4242 |    7.07094 |
| (20, 'mrr')  | 45.3177 | 12.0233 |  9.57721 | 33.2067 | 47.472  | 17.4125 | 16.2467 | 13.6488 | 10.4001  | 8.05987 | 12.2843 | 7.14386 | 4.14163 | 4.45509 |     22.8117 |    7.21695 |
| (25, 'mrr')  | 45.1459 | 12.1309 |  9.8724  | 33.4177 | 47.7942 | 17.7643 | 16.4459 | 13.4371 | 10.4981  | 7.80966 | 12.3091 | 7.22651 | 4.23409 | 4.35564 |     22.9452 |    7.187   |
| (30, 'mrr')  | 45.2453 | 12.516  | 10.2946  | 33.8225 | 47.6455 | 18.1683 | 16.3805 | 13.5795 | 10.5048  | 7.82047 | 12.0024 | 7.65289 | 4.37242 | 4.55625 |     23.1285 |    7.28089 |
| (35, 'mrr')  | 44.9832 | 12.6714 | 10.3581  | 33.9382 | 48.1022 | 19.0562 | 16.7391 | 13.4728 | 10.3644  | 7.80266 | 12.4172 | 7.50164 | 4.27899 | 4.18954 |     23.2984 |    7.23801 |
| (40, 'mrr')  | 45.0497 | 12.9713 | 10.5534  | 33.9162 | 48.2494 | 19.4182 | 17.0646 | 13.3876 | 10.7647  | 7.76955 | 12.0863 | 7.76417 | 4.58433 | 4.18861 |     23.4861 |    7.27859 |
| (45, 'mrr')  | 44.8709 | 12.7598 | 10.6207  | 34.0123 | 48.4076 | 20.1168 | 16.7293 | 13.4826 | 10.7217  | 7.7207  | 12.2151 | 7.79834 | 4.38069 | 4.2472  |     23.5246 |    7.27242 |
| (50, 'mrr')  | 44.4331 | 12.9428 | 10.3695  | 33.9687 | 48.3719 | 20.2204 | 17.002  | 13.2265 | 10.7709  | 7.51472 | 12.3501 | 7.86527 | 4.52629 | 4.30918 |     23.4784 |    7.31311 |
| (55, 'mrr')  | 44.8656 | 13.2094 | 10.7748  | 34.5006 | 48.9649 | 21.6445 | 17.5028 | 13.3423 | 10.9163  | 7.63899 | 12.6289 | 8.30786 | 4.61518 | 4.30892 |     23.969  |    7.49997 |
| (60, 'mrr')  | 44.9011 | 13.3243 | 10.8495  | 34.6471 | 49.0874 | 22.1415 | 17.4582 | 13.3771 | 10.8835  | 7.62007 | 12.5587 | 8.2811  | 4.68471 | 4.34016 |     24.0744 |    7.49695 |
| (65, 'mrr')  | 44.882  | 13.4375 | 10.8931  | 34.7594 | 49.126  | 22.3102 | 17.6696 | 13.408  | 10.9255  | 7.63332 | 12.6301 | 8.24105 | 4.70683 | 4.2976  |     24.1568 |    7.50179 |
| (70, 'mrr')  | 44.8828 | 13.651  | 11.0718  | 34.7735 | 49.2482 | 22.703  | 17.7788 | 13.4791 | 11.0766  | 7.68911 | 12.7907 | 8.35583 | 4.76243 | 4.40152 |     24.2961 |    7.59992 |
| (75, 'mrr')  | 44.9174 | 13.555  | 11.012   | 34.906  | 49.3564 | 22.7873 | 17.6738 | 13.396  | 11.0057  | 7.63671 | 12.7132 | 8.41619 | 4.61401 | 4.29353 |     24.2899 |    7.53473 |
| (80, 'mrr')  | 44.9689 | 13.694  | 11.0733  | 34.9882 | 49.5035 | 22.9782 | 17.7274 | 13.3799 | 11.0732  | 7.6816  | 12.7935 | 8.37103 | 4.75101 | 4.35969 |     24.3763 |    7.59136 |
| (85, 'mrr')  | 44.8746 | 13.5736 | 11.0516  | 35.0654 | 49.612  | 23.1731 | 17.7409 | 13.4019 | 11.0125  | 7.63303 | 12.8699 | 8.32122 | 4.76494 | 4.31249 |     24.3895 |    7.58032 |
| (90, 'mrr')  | 44.963  | 13.6078 | 11.0434  | 35.0212 | 49.539  | 22.9025 | 17.6933 | 13.3906 | 10.9815  | 7.68771 | 12.8578 | 8.27871 | 4.70053 | 4.26779 |     24.3491 |    7.5585  |
| (95, 'mrr')  | 44.8846 | 13.665  | 11.0756  | 35.1892 | 49.8396 | 23.0492 | 17.7944 | 13.3872 | 11.0303  | 7.66467 | 12.8382 | 8.39552 | 4.71225 | 4.3251  |     24.435  |    7.58714 |
| (100, 'mrr') | 44.9048 | 13.6668 | 11.1112  | 35.2322 | 49.6774 | 22.859  | 17.8386 | 13.3444 | 11.0184  | 7.70513 | 12.857  | 8.45513 | 4.74044 | 4.26359 |     24.4059 |    7.60425 |

## Citing this paper

```bibtex
@inproceedings{LMPNN,
  author    = {Zihao Wang and
               Yangqiu Song and
               Ginny Y. Wong and
               Simon See},
  title     = {Logical Message Passing Networks with One-hop Inference on Atomic Formulas},
  booktitle = {The Eleventh International Conference on Learning Representations, {ICLR} 2023, Kigali Rwanda, May 1-5, 2023},
  publisher = {OpenReview.net},
  year      = {2023},
  url       = {https://openreview.net/forum?id=SoyOsp7i_l},
}
```
