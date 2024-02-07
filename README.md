Train
```
bash run_train.sh
```

or 

```
python train.py \
    --mimic_root /PATH/TO/MIMIC/DIRECTORY \
    --save_dir /PATH/TO/OUTPUT/DIRECTORY \
    --batch_size 100 \
    --log_every -1 \ # wandb logging interval (set to -1 for no logging)

    --visual_patch_dropout 0.0 \ # set to 0.5 for randomly masking 50% of patches (FLIP)
    --use_all_token_embeds false \ # set to true for FILIP training
    --decoupled_contrastive_learning false \ # set to true for DCL training (ie, positive pair not in denominator of InfoNCE loss)
    --extra_latent_projection false \
    --use_visual_ssl false \
    --use_mlm false \
    --text_ssl_loss_weight 0.0 \
    --image_ssl_loss_weight 0.0 \
```