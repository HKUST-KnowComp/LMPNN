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

## Version
** This repo is under construction for usability. The key results for the paper can already be reproduced. **

Todo features:
- [ ] Introduce several ways to train KGE checkpoints with released repositories, which can be backbones for LMPNN.
- [x] Run CQD CO

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
  --output_dir log/FB15k/lmpnn-complex1k-default
```

Sample usage at NELL
```bash
python3 train_lmpnn.py \
  --task_folder data/NELL-betae \
  --checkpoint_path pretrain/cqd/NELL-model-rank-1000-epoch-100-1602499096.pt \
  --device cuda:2 \
  --batch_size 512 \
  --batch_size_eval_dataloader 64 \
  --batch_size_eval_truth_value 8 \
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
