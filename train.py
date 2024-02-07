#%%
import warnings
warnings.filterwarnings("ignore")

import os#; os.environ['CUDA_VISIBLE_DEVICES']='3'
import cv2
import json
# import h5py
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from  torch.cuda.amp import autocast
import torchvision.transforms as T

from Q_utils import extract_impression, get_parameter_count
from simple_tokenizer import SimpleTokenizer
from openai_model import build_model
from x_clip import CLIP, TextTransformer, VisionTransformer
from dataloaders import get_mimic_dataloader, get_chexpert_test_loader_and_gt, get_mimic_test_loader_and_gt
from eval import eval

import wandb


#%%
def preprocess_text(texts):
#     if model.context_length is None: 
#         model = model.module
    LEN_TXT_TOKENS = 77 # context length (77 for CLIP)
    
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), LEN_TXT_TOKENS, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > LEN_TXT_TOKENS:
            tokens = tokens[:LEN_TXT_TOKENS]
            tokens[LEN_TXT_TOKENS - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

#%%
def main(args):
    train_loader = get_mimic_dataloader(mimic_root=args.mimic_root,
                                    split='train',
                                    shuffle=True,
                                    batch_size=args.batch_size)
    
    vision_transformer_config = {
        "dim":512,
        "image_size":256,
        "patch_size":32,
        "channels":3,
        "depth":12,
        "heads":12,
        "dim_head":64,
        "patch_dropout":args.visual_patch_dropout,
        "checkpoint_during_training":False,
    }
    
    text_transformer_config = {
        "dim":512,
        "num_tokens":len(SimpleTokenizer().encoder),
        "max_seq_len":77,
        "depth":12,
        "heads":8,
        "dim_head":64
    }
    
    # save encoder configs
    vision_transformer_config_path = os.path.join(args.save_dir, 'vision_transformer_config.json')
    with open(vision_transformer_config_path, 'w') as f:
        json.dump(vision_transformer_config, f, indent=2)
    text_transformer_config_path = os.path.join(args.save_dir, 'text_transformer_config.json')
    with open(text_transformer_config_path, 'w') as f:
        json.dump(text_transformer_config, f, indent=2)
    
    image_encoder = VisionTransformer(**vision_transformer_config)
    text_encoder = TextTransformer(**text_transformer_config)

    clip = CLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image = 512,
        dim_text = 512,
        dim_latent= 512,
        use_all_token_embeds = args.use_all_token_embeds, # whether to use fine-grained contrastive learning (FILIP)
        decoupled_contrastive_learning = args.decoupled_contrastive_learning, # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
        extra_latent_projection = args.extra_latent_projection, # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_visual_ssl = args.use_visual_ssl, # whether to do self supervised learning on images
        use_mlm = args.use_mlm, # use masked language learning (MLM) on text (DeCLIP)
        text_ssl_loss_weight = args.text_ssl_loss_weight, # weight for text MLM loss
        image_ssl_loss_weight = args.image_ssl_loss_weight # weight for image self-supervised learning loss
    ).to('cuda')

    optimizer = torch.optim.SGD(clip.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(args.epochs):
        image_encoder.train()
        text_encoder.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch} training', colour='green')):
            
            images = batch['img'].to('cuda')
            texts = batch['txt']
            texts = preprocess_text(texts).to('cuda')
                    
            loss = clip(
                text=texts,
                image=images,
                freeze_image_encoder=False,
                freeze_text_encoder=False,
                return_loss=True,
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.log_every > 0:
                if i % args.log_every == 0:
                    example_ct = args.batch_size * i + epoch * len(train_loader) * args.batch_size
                    wandb.log({'loss': loss}, step=example_ct)
        
        # Eval every epoch
        texts = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        chexpert_test_loader, chexpert_test_gt = get_chexpert_test_loader_and_gt()
        mean_auroc, aurocs = eval(image_encoder, text_encoder, chexpert_test_loader, texts, chexpert_test_gt)
        aurocs = {f"CheXpert_{k}":v for k,v in aurocs.items()}
        # print(mean_auroc)
        # pprint(aurocs)
        if args.log_every > 0:
            example_ct = args.batch_size * i + epoch * len(train_loader) * args.batch_size
            wandb.log({'CheXpert_mean_auroc': mean_auroc, **aurocs}, step=example_ct)
            
        mimic_test_loader, mimic_test_gt = get_mimic_test_loader_and_gt()
        mean_auroc, aurocs = eval(image_encoder, text_encoder, mimic_test_loader, texts, mimic_test_gt)
        aurocs = {f"MIMIC_{k}":v for k,v in aurocs.items()}
        # print(mean_auroc)
        # pprint(aurocs)
        if args.log_every > 0:
            example_ct = args.batch_size * i + epoch * len(train_loader) * args.batch_size
            wandb.log({"MIMIC_mean_auroc": mean_auroc, **aurocs}, step=example_ct)
        
        
        # Save checkpoint every epoch
        sd = clip.visual_transformer.state_dict()
        save_path = os.path.join(args.save_dir, f"image_encoder_epoch{epoch}.pt")
        torch.save(sd, save_path)
        sd = clip.text_transformer.state_dict()
        save_path = os.path.join(args.save_dir, f"text_encoder_epoch{epoch}.pt")
        torch.save(sd, save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='x-clip')
    parser.add_argument('--mimic_root', type=str, default='/home/wonjun/data/mimic-cxr-jpg-resized512')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='outputs')
    
    # x-clip options
    parser.add_argument('--visual_patch_dropout', type=float, default=0) # flip
    parser.add_argument('--use_all_token_embeds', type=bool, default=False)
    parser.add_argument('--decoupled_contrastive_learning', type=bool, default=False)
    parser.add_argument('--extra_latent_projection', type=bool, default=False)
    parser.add_argument('--use_visual_ssl', type=bool, default=False)
    parser.add_argument('--use_mlm', type=bool, default=False)
    parser.add_argument('--text_ssl_loss_weight', type=float, default=0.05)
    parser.add_argument('--image_ssl_loss_weight', type=float, default=0.05)
    parser.add_argument('--notes', type=str, default="")
    
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    args = parse_args()
    
    train_args_path = os.path.join(args.save_dir, 'train_args.json')
    with open(train_args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    if args.log_every > 0:
        run = wandb.init(project=args.project_name, config=vars(args))
    main(args)