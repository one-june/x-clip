CUDA_VISIBLE_DEVICES='3' \
python train.py \
    --mimic_root /PATH/TO/MIMIC/DIRECTORY \
    --save_dir outputs/test \
    --batch_size 100 \
    --log_every -1 \
    --notes "test"