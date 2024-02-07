#%%
import os; os.environ['CUDA_VISIBLE_DEVICES']='3'
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer

from x_clip import CLIP, TextTransformer, VisionTransformer
# from vit_pytorch import ViT
# from vit_pytorch.extractor import Extractor
# import vision_transformer as vits

from Q_utils import extract_findings, extract_impression, get_parameter_count

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
    
BATCH_SIZE = 4


#%%
class MIMIC_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 mimic_root,
                 df_path,
                 text_encoder_name = "emilyalsentzer/Bio_ClinicalBERT",
                 text_max_length = 256):
        self.mimic_root = mimic_root
        
        # metadata_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-metadata.csv')
        # self.metadata = pd.read_csv(metadata_path)
        self.df = pd.read_csv(df_path)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        if self.text_tokenizer.bos_token_id is None:
            self.text_tokenizer.bos_token_id = self.text_tokenizer.cls_token_id

        self.text_max_length = text_max_length
        
        self.image_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((256,256)),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # load text
        txt_path = os.path.join(self.mimic_root, 'reports', 'files',
                            f"p{str(int(row['subject_id']))[:2]}",
                            f"p{str(int(row['subject_id']))}",
                            f"s{str(int(row['study_id']))}.txt")
        with open(txt_path, 'r') as handle:
            report = handle.read()
        # findings = extract_findings(report)
        impression = extract_impression(report)
        
        text = impression
        
        # load image
        jpg_path = os.path.join(self.mimic_root, 'files',
                                f"p{str(int(row['subject_id']))[:2]}",
                                f"p{str(int(row['subject_id']))}",
                                f"s{str(int(row['study_id']))}",
                                row['dicom_id']+'.jpg')
        image = Image.open(jpg_path).convert('RGB')
        image = self.image_transforms(image)
        
        out = {"image": image, "text": text}
        
        return out
        
    def collate_fn(self, instances):
        images = torch.stack([ins['image'] for ins in instances], dim=0)
        
        texts = [ins["text"] for ins in instances]
        text_tokens = self.text_tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=self.text_max_length)

        batch = {
            "images": images,
            "text_tokens": text_tokens
        }
        return batch        
        
#%%
dataset = MIMIC_Dataset(
    mimic_root='/home/wonjun/data/mimic-cxr-jpg-resized512',
    df_path='mimic_multilabel_all.csv',
)

dataloader = DataLoader(
    dataset,
    collate_fn=getattr(dataset, "collate_fn"),
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    num_workers=1,
    prefetch_factor=16,
    batch_size=BATCH_SIZE
)

#%%
# clip = CLIP(
#     dim_text = 256,
#     dim_image = 256,
#     dim_latent = 256,
#     num_text_tokens = dataset.text_tokenizer.vocab_size,
#     text_enc_depth = 6,
#     text_seq_len = 256,
#     text_heads = 8,
#     visual_enc_depth = 6,
#     visual_image_size = 256,
#     visual_patch_size = 32,
#     visual_heads = 8,
#     visual_patch_dropout = 0,             # patch dropout probability, used in Kaiming He's FLIP to save compute and improve end results - 0.5 is good value, 0.75 on high end is tolerable
#     use_all_token_embeds = False,           # whether to use fine-grained contrastive learning (FILIP)
#     decoupled_contrastive_learning = False,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
#     extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
#     use_visual_ssl = False,                  # whether to do self supervised learning on images
#     use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
#     text_ssl_loss_weight = 0.05,            # weight for text MLM loss
#     image_ssl_loss_weight = 0.05            # weight for image self-supervised learning loss
# ).to('cuda')



# for i, batch in enumerate(tqdm(dataloader, desc='dataloader')):
#     if i >= 10:
#         break
    
#     i = batch['images'].to('cuda')
#     t = batch['text_tokens']['input_ids'].to('cuda')
#     loss = clip(
#         t, i,
#         freeze_image_encoder = False,
#         return_loss = True
#     )

#%%
image_encoder = VisionTransformer(
          dim = 256,
          image_size = 256,
          patch_size = 16,
          channels = 3,
          depth = 12,
          heads = 6,
          dim_head = 384,
          patch_dropout = 0,
          checkpoint_during_training = False,
      )

text_encoder = TextTransformer(
    dim = 256,
    num_tokens = dataset.text_tokenizer.vocab_size,
    max_seq_len = 256,
    depth = 12,
    heads = 6,
    dim_head=384
)

#%%
clip = CLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 256, # must be set as the same dimensions as the vision transformer above
    dim_text = 256,
    dim_latent = 256,
).to('cuda')

# text = torch.randint(0, 10000, (4, 256)).to('cuda')
# images = torch.randn(4, 3, 256, 256).to('cuda')

# loss = clip(text, images, return_loss = True)
# loss


# %%
params = get_params_groups(clip)
optimizer = torch.optim.AdamW(params)

for i, batch in enumerate(tqdm(dataloader, desc='dataloader')):
    
    
    i = batch['images'].to('cuda')
    t = batch['text_tokens']['input_ids'].to('cuda')
    loss = clip(
        t, i,
        freeze_image_encoder = False,
        return_loss = True
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
# %%
i.shape