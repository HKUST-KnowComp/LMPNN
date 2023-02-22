# Logical Message Passing Networks with One-hop Inference on Atomic Formula

Codes for ICLR 2023 paper Logical Message Passing Networks with One-hop Inference on Atomic Formula, see the [arXiv version](https://arxiv.org/abs/2301.08859) and the [OpenReview version](https://openreview.net/forum?id=SoyOsp7i_l).

In this documentation, we demonstrate to run the code on FB15k-237

## Requirement
- pytorch
- jupyter
- tqdm

## Prepare the dataset

Please download the dataset from [snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning)

Specifically,
```bash
mkdir data
cd data
wget http://snap.stanford.edu/betae/KG_data.zip # a zip file of 1.3G
unzip KG_data.zip
```

Then the `data` folder will contain the following folders
```
FB15k-237-betae
FB15k-237-q2b
FB15k-betae
FB15k-q2b
KG_data.zip
NELL-betae
NELL-q2b
```
Then, we rearange them into different subfolders
```
mkdir betae-dataset
mv *betae betae-dataset
mkdir q2b-dataset
mv *q2b q2b-dataset
```

Then we run `convert_beta_dataset.py` to convert the data into the graph forms

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

## Pretrain KGE checkpoints with external submodules

We consider two different repositories to pretrain the KGE checkpoints.
Including
1. [uma-pi1/kge](https://github.com/uma-pi1/kge)
2. [facebookresearch/ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction)

To initialize these modules, please run
```bash
git submodule update
```

How to train the checkpoints with these submodules is discussed in this section.

Generally, there are three steps:
1. Convert the KG triples into the format that can be used in each submodule.
2. Train the checkpoints.
3. Convert the obtained checkpoints into the format that can be used in LMPNN.

In this part, we provide examples on training ComplEx embeddings at FB15k-237, other KGs can be reproduced accordingly.

### Example usage of [uma-pi1/kge](https://github.com/uma-pi1/kge)

#### Step 0: Prepare the environment and config

To run `libkge` submodule, one need editable installation.

```bash
cd kge
pip install -e .
```

Then download the configs

```
cd kge/data
sh download_all.sh

mkdir -p config/kge
wget http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.yaml -o config/kge/fb15k-237-complex.yaml
```

#### Step 1: Prepare the dataset

1. Run the script to convert the knowledge graph triples.
2. Run kge with the customized config.
3. Convert KGE checkpoints back into the.

### Example usage of [facebookresearch/ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction)

1. Run the script to convert the knowledge graph triples.
2. Run kge with the customized config.
3. Convert KGE checkpoints back into the.

## Train LMPNN

Sample usage

```bash
python lifted_embedding_estimation_with_truth_value.py \
  --task_folder data/FB15k-237-betae \
  --checkpoint_path pretrain/complex/FB15k-model-rank-1000-epoch-100-1602520745.pt
  --embedding_dim 1000 \
  --device cuda:0 \
  --output_dir log/fb15k/pretrain_complex1000-default \
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
