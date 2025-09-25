#!/bin/bash
set -euo pipefail
set -x

pretrained_model_name_or_path=$1
run_type=${2:-"llm"} # "ocr": only ocr, "llm": ocr + qa
if [ ! -d $pretrained_model_name_or_path ]; then
    echo "Pretrained model not found: $pretrained_model_name_or_path"
    exit 1
fi

source .venv/bin/activate

python src/ui/tabq.py \
   --run_type "$run_type" \
   --vis_model_name_or_path "$pretrained_model_name_or_path/vis" \
   --tokenizer_name_or_path "$pretrained_model_name_or_path" \
   --pretrained_model_name_or_path "$pretrained_model_name_or_path" \
   --port "5120"