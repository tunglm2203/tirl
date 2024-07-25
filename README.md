# On the Perturbed States for Transformed Input-robust Reinforcement Learning

##  Description:

This repository implements the paper "On the Perturbed States for Transformed Input-robust Reinforcement Learning".

## Installation

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions:
* Training scripts can be found in `scripts/run`. for example, to run `SAC-TIRL-VQ`: 

      `bash scripts/run/run_sac_tirl_vq.sh`

The checkpoint will be stored in `d3rlpy_logs/ENV_NAME/EXP_NAME`. 

* Evaluation scripts can be found in `scripts/evals`
