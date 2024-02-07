#%%
import os; os.environ['CUDA_VISIBLE_DEVICES']='3'
import torch
from x_clip import CLIP, TextTransformer
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

#%%
clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 10000,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8,
    visual_patch_dropout = 0.5,             # patch dropout probability, used in Kaiming He's FLIP to save compute and improve end results - 0.5 is good value, 0.75 on high end is tolerable
    use_all_token_embeds = False,           # whether to use fine-grained contrastive learning (FILIP)
    decoupled_contrastive_learning = False,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_visual_ssl = False,                  # whether to do self supervised learning on images
    use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
    text_ssl_loss_weight = 0.05,            # weight for text MLM loss
    image_ssl_loss_weight = 0.05            # weight for image self-supervised learning loss
).to('cuda')
# %%
text = torch.randint(0, 10000, (4, 256))
images = torch.randn(4, 3, 256, 256)
text = text.to('cuda')
images = images.to('cuda')
# %%
loss = clip(
    text,
    images,
    freeze_image_encoder = False,   # whether to freeze image encoder if using a pretrained image net, proposed by LiT paper
    return_loss = True              # needs to be set to True to return contrastive loss
)
#%%
loss

#%%
print(text.shape)
print(images.shape)

# %%
base_vit = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 512,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

image_encoder = Extractor(
    base_vit,
    return_embeddings_only = True
)

text_encoder = TextTransformer(
    dim = 512,
    num_tokens = 10000,
    max_seq_len = 256,
    depth = 6,
    heads = 8,
    dim_head=64
)

clip = CLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 512, # must be set as the same dimensions as the vision transformer above
    dim_text = 512,
    dim_latent = 512,
)

text = torch.randint(0, 10000, (4, 256))
images = torch.randn(4, 3, 256, 256)

loss = clip(text, images, return_loss = True)
loss.backward()
# %%
image_encoder(images).shape
# %%
text_encoder(text).shape