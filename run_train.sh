CUDA_VISIBLE_DEVICES='3' \
python train.py \
    --mimic_root /PATH/TO/MIMIC/DIRECTORY \
    --save_dir outputs/test \
    --batch_size 100 \
    --log_every -1 \
    --notes "test"A


CUDA_VISIBLE_DEVICES=1 \
    torchrun --nproc_per_node=1 \
    train_sparc.py \
    --mimic_root /data/wonjun/mimic-cxr-jpg-resized512/ \
    --save_dir /data/jaayeon/checkpoint-xclip/sparc-norm-lw \
    --batch_size 144 \
    --log_every 50 \
    --notes "test"A \
    --model sparc \
    --clip_grad 3 \
    --multi_gpu \
    --use_fp16 \
    --epochs 100