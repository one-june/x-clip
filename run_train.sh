CUDA_VISIBLE_DEVICES='3' \
python train.py \
    --mimic_root /PATH/TO/MIMIC/DIRECTORY \
    --save_dir outputs/test \
    --batch_size 100 \
    --log_every -1 \
    --notes "test"A


CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --mimic_root /data/wonjun/mimic-cxr-jpg-resized512/ \
    --save_dir /data/jaayeon/ \
    --batch_size 128 \
    --log_every -1 \
    --notes "test"A \
    --model sparc