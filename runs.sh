#! /bin/bash
echo "Activating conda"
source ~/anaconda3/bin/activate
conda activate penguin 

python main.py --backbone wavlm-large --precision 16 --batch-size 4 --accumulate-grad-batches 16 --max-epochs 16 --grad-clip-val 0.5 --learning-rate 1e-4 --eng-other --final
python main.py --backbone wavlm-large --precision 16 --batch-size 4 --accumulate-grad-batches 16 --max-epochs 16 --grad-clip-val 0.5 --learning-rate 1e-4 --lang-fams --pretrained-eng-other --no-mono-eng --final
python main.py --backbone wavlm-large --precision 16 --batch-size 4 --accumulate-grad-batches 16 --max-epochs 16 --grad-clip-val 0.5 --learning-rate 1e-4 --pretrained-lang-fams --no-mono-eng --final