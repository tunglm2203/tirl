#!/bin/bash
ROOT_DIR="d3rlpy_logs"

ENV_NAME="Ant-v4"

N_STEPS=3000000

N_EMBEDDINGS=500000
TRANSFORMATION_TYPE="vanilla_sgd"

RL_ALGO='SAC'
EXP_NAME="sac_tirl_vq_K${N_EMBEDDINGS}"
LOG_DIR="${ROOT_DIR}/${ENV_NAME}/${EXP_NAME}"

CUDA_VISIBLE_DEVICES=0 python run_rl_online.py --dataset ${ENV_NAME} --logdir ${LOG_DIR} --exp ${EXP_NAME} \
  --algo ${RL_ALGO} --n_steps ${N_STEPS} --standardization --no_replacement \
  --use_input_transformation --n_embeddings ${N_EMBEDDINGS} \
  --transformation_type ${TRANSFORMATION_TYPE} --autoscale_vq_loss \
  --seed 1 --wandb
