### sgm

Incomplete python port of seeded graph matching (SGM)

#### Installation

1) Install `pytorch` per instructions at pytorch.org
2) `pip install -r requirements.txt` -- might have trouble on OSX, I only tested on Linux, and you have to compile something

#### Usage

Untar `data.tar.gz` in the project directory, and look at `./run.sh`

Alternatively, if you have access to the original SGM R code, you can use `./utils/ta1-pipeline-prep.R` to fill `./data` directory.

Note -- this yields the same results as the original package, but it is not the same at the _iteration_ level.  Not totally sure what the source of this is -- could be some numerical precision thing, or differences in fallbacks in the LAP solver, or something else.  It doesn't seem like it's causing big problems though (yet).

#### Docker

If you want to run this in the official `pytorch` docker:

  - Start terminal inside docker container:
  
    `sudo docker run -v $(pwd)/data:/data -it pytorch/pytorch /bin/bash`
     
  - Then do:
    ```
    git clone https://github.com/bkj/sgm
    cd sgm

    pip install -r requirements.txt

    python sgm.py \
        --A-path /data/A1.ordered \
        --B-path /data/A2.ordered \
        --P-path /data/S
    ```
