#!/bin/bash
ENV_NAME="Ant-v4"


CKPT_STEPS="model_3000000.pt"
N_SEEDS_WANT_TO_TEST=1
N_EVAL_EPISODES=50

TRANSFORMATION_TYPE="bdr"
BDR_STEP=0.05

CKPT="d3rlpy_logs/${ENV_NAME}/sac_tirl_bdr_bW${BDR_STEP}"
CUDA_VISIBLE_DEVICES=0 python -W ignore evaluate_online_learned_policy.py --dataset ${ENV_NAME} --gpu 0 --ckpt ${CKPT} \
  --n_seeds_want_to_test ${N_SEEDS_WANT_TO_TEST} --n_eval_episodes ${N_EVAL_EPISODES} \
  --ckpt_steps ${CKPT_STEPS} \
  --use_input_transformation --bdr_step ${BDR_STEP} --transformation_type ${TRANSFORMATION_TYPE} \
  --standardization
