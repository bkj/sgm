### sgm

Python port of [seeded graph matching (SGM)](https://arxiv.org/pdf/1209.0367.pdf)

#### Quickstart

```
# make fresh conda env
conda create -n sgm_env python=3.6 pip -y
source activate sgm_env

# install requirements
cat requirements.txt | xargs -I {} pip install {}

# install sgm
pip install -e .
```

Tested on Ubuntu 16.04.  Installation on OSX may fail.

#### Example Usage

Simple example:
```
cd examples/synthetic
python main.py
```

More complex example:
```
cd examples/kasios/
python kasios.py --num-nodes 1000 --num-seeds 32 --backend scipy.classic.jv
```
