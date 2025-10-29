#!/bin/bash
source .venv/bin/activate
which python
python -V

nvidia-smi

export WANDB_MODE="offline"

echo "Started: $(date)"
accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-base \
    --instance_data_dir dreambooth/dataset/dog6 \
    --output_dir experiments/dreambooth/00019-e75e-dog6 \
    --instance_prompt="a photo of dog6" \
    --resolution=512 \
    --train_text_encoder \
    --max_train_steps=600 \
    --train_batch_size=1 \
    --learning_rate=2e-6 \
    --gradient_accumulation_steps=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --checkpointing_steps=100 \
    --seed=0 \
    --report_to wandb
echo "Finished: $(date)"



# python collect_pairs.py \
#     --meta_path experiments/dreambooth/$EXP/eval.json \
#     --img_root experiments/dreambooth/$EXP/checkpoint-400/samples/ns50_gs7.5/version_0 \
#     --prompts_path data/$CLASS/$NPROMPTS-2-parts/0.json \
#     --concept "sks $CLASS" \
#     --out_path data/$CLASS/$NPROMPTS-$CONCEPT-a20_70-2-parts/0.json \
#     --exp_key "('$EXP', ('400', '50', '7.5'))" \
#     --score_type text \
#     --threshold 0.0 \
#     --atan_min 20.0 \
#     --atan_max 70.0
# echo "Finished (pairs: 20, 70): $(date)"