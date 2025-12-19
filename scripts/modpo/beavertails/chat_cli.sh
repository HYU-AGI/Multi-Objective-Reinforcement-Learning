#!/bin/bash
# GPU 설정 (환경변수가 없으면 기본값 사용)
# 사용법: GPUS=0,1 ./scripts/modpo/beavertails/chat_cli.sh
GPUS=${GPUS:-"4,7"}

python scripts/modpo/beavertails/chat_cli.py \
  --gpus ${GPUS} \
  --sft_model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter_path ./output/myset/lm/trust_with_creative_margin_w0.5/best_checkpoint \
  --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" \
  --max_new_tokens 256 --temperature 0.7 --top_p 0.9 --stream
