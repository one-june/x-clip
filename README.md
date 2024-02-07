Train
```
bash run_train.sh
```

or 

```
python train.py \
    --mimic_root /PATH/TO/MIMIC/DIRECTORY \
    --save_dir outputs/test \
    --batch_size 100 \
    --log_every -1 \
    --notes "test"

    --visual_patch_dropout 0.0 # set to 0.5 for randomly masking 50% of patches (FLIP)
    --use_all_token_embeds false # set to true for FILIP training
    --decoupled_contrastive_learning # set to true for DCL training (ie, positive pair not in denominator of InfoNCE loss)
    --extra_latent_projection
    --use_visual_ssl
    --use_mlm
    --text_ssl_loss_weight
    --image_ssl_loss_weight
```