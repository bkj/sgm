### sgm

Python port of seeded graph matching (SGM)

#### Installation

```
conda create -n sgm_env python=3.6 -y
source activate sgm_env

conda install pytorch -c pytorch
pip install -r requirements.txt

git clone https://github.com/bkj/auction-lap auction_lap
```

#### Usage
```

# Generate synthetic data
mkdir -p _data/synthetic/{sparse,dense}
Rscript utils/make-correlated-gnp.R

INPATH="_data/synthetic/sparse/0.05/5000/"
python main-scipy.py \
    --A-path $INPATH/A.edgelist \
    --B-path $INPATH/B.edgelist \
    --P-path $INPATH/P.edgelist \
    --mode exact \
    --symmetric

```