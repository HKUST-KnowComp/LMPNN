echo Download the data
mkdir data
cd data
wget http://snap.stanford.edu/betae/KG_data.zip # a zip file of 1.3G
unzip KG_data.zip
mkdir betae-dataset
mv *betae betae-dataset
mkdir q2b-dataset
mv *q2b q2b-dataset
cd ..

echo Convert the Data
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 convert_beta_dataset.py

echo Download the pretrain KGE
mkdir pretrain
cd pretrain
wget http://data.neuralnoise.com/cqd-models.tgz
tar xvf cqd-models.tgz
mv models raw_cqd_pretrain_models
cd ..

echo Convert the pretrain KGE
python3 convert_cqd_pretrain_ckpts.py
