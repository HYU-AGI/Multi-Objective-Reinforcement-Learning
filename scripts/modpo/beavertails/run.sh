#!/usr/bin/env bash
set -euo pipefail

# 캐시/토크나이저 병렬 경고 억제 (선택)
export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TOKENIZERS_PARALLELISM=false

# GPU 고정: 기본값 제공, 스크립트 실행 시 GPUS 환경변수로 커스터마이징 가능
# Usage: GPUS="0,1" ./scripts/modpo/beavertails/run.sh
GPUS="${GPUS:-4,7}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=2"

SFT_MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT_TEMPLATE="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
DATA_ROOT="./data/sample"            # dataset_prepare.py가 생성한 루트
OUTPUT_ROOT="./output/myset"

MAX_LENGTH=512
BS=2
GAS=1
LR=5e-4
SANITY=False   

# 1단계: DPO로 '마진' 정책 학습 (예: 창의성 정책을 마진으로 쓰는 경우)
RM_RUN_DIR="${OUTPUT_ROOT}/rm/creative"
PYTHONPATH=. $LAUNCH scripts/examples/dpo/dpo.py \
  --sft_model_name "${SFT_MODEL_NAME}" \
  --dataset_dir "${DATA_ROOT}/creative" \
  --prompt_template "${PROMPT_TEMPLATE}" \
  --sanity_check ${SANITY} \
  --max_length ${MAX_LENGTH} \
  --training_args.output_dir "${RM_RUN_DIR}" \
  --training_args.run_name "${RM_RUN_DIR}" \
  --training_args.per_device_train_batch_size ${BS} \
  --training_args.per_device_eval_batch_size ${BS} \
  --training_args.gradient_accumulation_steps ${GAS} \
  --training_args.learning_rate ${LR} \
  --peft_config.r 64 \
  --peft_config.target_modules q_proj k_proj v_proj o_proj \
  --peft_config.lora_alpha 1 \
  --peft_config.lora_dropout 0

# 2단계: MODPO로 주목적 데이터 학습 (예: 신뢰도를 주목적, 창의성을 마진으로)
for w in 0.1 0.5 0.9; do
  LM_RUN_DIR="${OUTPUT_ROOT}/lm/trust_with_creative_margin_w${w}"
  PYTHONPATH=. $LAUNCH scripts/modpo/beavertails/modpo.py \
    --sft_model_name "${SFT_MODEL_NAME}" \
    --margin_reward_model_name "${RM_RUN_DIR}/best_checkpoint" \
    --dataset_dir "${DATA_ROOT}/trust" \
    --prompt_template "${PROMPT_TEMPLATE}" \
    --sanity_check ${SANITY} \
    --w ${w} \
    --max_length ${MAX_LENGTH} \
    --training_args.output_dir "${LM_RUN_DIR}" \
    --training_args.run_name "${LM_RUN_DIR}" \
    --training_args.per_device_train_batch_size ${BS} \
    --training_args.per_device_eval_batch_size ${BS} \
    --training_args.gradient_accumulation_steps ${GAS} \
    --training_args.learning_rate ${LR} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0
done
