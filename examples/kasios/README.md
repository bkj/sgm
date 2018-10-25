### kasios

#### Setup

```bash

# Download data
PROJECT_ROOT=$(pwd)
mkdir -p data/orig
cd data/orig
wget http://vacommunity.org/tiki-download_file.php?fileId=577
mv tiki-download_file.php?fileId=577 tiki.zip
unzip tiki.zip && rm tiki.zip
cd $PROJECT_ROOT

# Dataset is large, so sample by doing random walks
mkdir -p data
python rw-sample.py --inpath data/orig/calls.csv > data/calls.rw


```

#### Experiments

See `run.sh` for performance comparisons using different SGM backends.