# Dreambooth-DPO-Instructions-to-run

# Installation
# Clone this repo:
git clone https://github.com/ControlGenAI/DreamBoothDPO.git
cd DreamBoothDPO

# Create Conda environment:
conda create -n dbdpo python=3.11
conda activate dbdpo

# Install the dependencies in your environment:
pip install -r requirements.txt

# t resolve error from the torch and python mismatch version:
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm torch torchvision cython pycocotools

# Prompts preparation
# 0.0 Download and extract COCO annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

# 0.1 Collect prompts from COCO.
python gen_prompts_from_coco.py <args...>

# 0.2 Merge COCO prompts with ChatGPT prompts.
python data/merge.py <args...>

#Remove error 
python gen_prompts_from_coco.py \
  --coco_path ./ \
  --out_path data/coco_prompts.jsonl \
  --split train


# Move them to the current folder
mv annotations/* .
rm -r annotations

# Before merging clone dreambooth dataset via
git clone https://github.com/google/dreambooth.git

# 0.2 Merge COCO prompts with ChatGPT prompts.
cd data/dog
cp prompts-4k.json coco-train.json
cp prompts-4k.json coco-val.json

cd/data
python merge.py --object dog --n_coco_prompts 200
#(There should be two files in dog folder coco-train.json and coco-val.json.)

#Login
pip install --upgrade huggingface-hub
huggingface-cli login
#then enter the token

#pips
pip install --upgrade "peft>=0.17.0"
pip install --upgrade pip setuptools wheel
pip install accelerate diffusers transformers datasets safetensors
pip install wandb
# uninstall current release
pip uninstall -y diffusers || true

# clone and install editable
git clone https://github.com/huggingface/diffusers.git /workspace/diffusers-src
cd /workspace/diffusers-src
pip install -e .
cd /workspace/DreamBoothDPO

cd /workspace/diffusers-src

# install the example-specific requirements (this file exists in the diffusers repo)
pip install -r examples/dreambooth/requirements.txt

# Clone source repo
git clone https://github.com/huggingface/diffusers.git
cd diffusers
# Install source version
pip install -e .
cd ..


accelerate config

# Activate your environment if not already
source /venv/dbdpo/bin/activate   # or your correct path

# Install wandb
pip install wandb
 #And then do some changes in script/train_dreambooth.sh file :
1. source .vev/bin/activate --> source /venv/dbdpo/bin/activate
2. --instance_data_dir dreambooth/dataset/dog6 \
    --output_dir experiments/dreambooth/00019-e75e-dog6 \
    --instance_prompt="a photo of dog6" \

After these changes it should work 
bash scripts/train_dreambooth.sh

After this do evaluation:

pip uninstall clip -y
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib natsort ipywidgets ipyfilechooser seaborn 

Then do some changes in generate.py file

def generate_and_save(pipeline: StableDiffusionPipeline, prompts: List[str], samples_dir: Path, args):
    images = generate_batch(pipeline, prompts, args)

    assert len(images) == len(prompts) * args.samples_per_prompt

    #for prompt, images_for_one_prompt in zip(prompts, chunks(images, args.samples_per_prompt), strict=True):
    #    out_dir = samples_dir / prompt
    #    for i, img in enumerate(images_for_one_prompt):
    #       img.save(out_dir / f'{i}.png')
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(samples_dir / f'{i:03d}.png')
Make the following script file with name generate_val1.sh in scripts
python generate.py \
  --ckpt_path /workspace/DreamBoothDPO/experiments/dreambooth/00019-e75e-dog \
  --out_dir /workspace/DreamBoothDPO/experiments/ddpo/00115-0001-dog-s1k-a20_70-step1of2 \
  --concept 'sks dog' \
  --prompt_source eval_set \
  --eval_set live_long \
  --samples_per_prompt 10 \
  --prompts_per_batch 5

  
python generate.py \
  --ckpt_path /workspace/DreamBoothDPO/experiments/dreambooth/00019-e75e-dog \
  --out_dir /workspace/DreamBoothDPO/experiments/ddpo/00115-0001-dog-s1k-a20_70-step1of2 \
  --concept 'sks dog' \
  --prompt_source eval_set \
  --eval_set live_long \
  --samples_per_prompt 10 \
  --prompts_per_batch 5
echo "Finished (genval): $(date)"

bash scripts/generate_val1.sh
