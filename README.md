### sgm

Incomplete python port of seeded graph matching (SGM)

#### Installation

1) Install `pytorch` per instructions at pytorch.org
2) `pip install -r requirements.txt` -- might have trouble on OSX, I only tested on Linux, and you have to compile something

#### Usage

Untar `data.tar.gz` in the project directory, and look at `./run.sh`

Alternatively, if you have access to the original SGM R code, you can use `./utils/ta1-pipeline-prep.R` to fill `./data` directory.

Note -- this yields the same results as the original package, but it is not the same at the _iteration_ level.  Not totally sure what the source of this is -- could be some numerical precision thing, or differences in fallbacks in the LAP solver, or something else.  It doesn't seem like it's causing big problems though (yet).
