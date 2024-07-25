#!/bin/bash
ROOT_DIR="d3rlpy_logs"

ENV_NAME="Ant-v4"

N_STEPS=3000000

BDR_STEP=0.05
TRANSFORMATION_TYPE="bdr"

RL_ALGO='SAC'
EXP_NAME="sac_tilr_bdr_bW${BDR_STEP}"
LOG_DIR="${ROOT_DIR}/${ENV_NAME}/${EXP_NAME}"

CUDA_VISIBLE_DEVICES=0 python run_rl_online.py --dataset ${ENV_NAME} --logdir ${LOG_DIR} --exp ${EXP_NAME} \
  --algo ${RL_ALGO} --n_steps ${N_STEPS} --standardization --no_replacement \
  --use_input_transformation --bdr_step ${BDR_STEP} \
  --transformation_type ${TRANSFORMATION_TYPE} \
  --seed 1 --wandb
