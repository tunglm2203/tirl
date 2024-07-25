#!/bin/bash
ROOT_DIR="d3rlpy_logs"

ENV_NAME="Ant-v4"

EXP_NAME="AEDenoiser_ant_eps0.15"
LOG_DIR="${ROOT_DIR}/${ENV_NAME_DIR}/denoiser/${EXP_NAME}"

CKPT_STEPS="model_3000000.pt"
CKPT_DIR="${ROOT_DIR}/${ENV_NAME}/sac_baseline"

CUDA_VISIBLE_DEVICES=${GPU} python run_denoiser.py --dataset ${ENV_NAME} --logdir ${LOG_DIR} --exp ${EXP_NAME} \
  --ckpt ${CKPT_DIR} --ckpt_steps ${CKPT_STEPS} --load_buffer --standardization \
  --seed 1 --ckpt_id 0

