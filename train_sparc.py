#%%
import warnings
warnings.filterwarnings("ignore")

import os#; os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from Q_utils import extract_impression, get_parameter_count
from simple_tokenizer import SimpleTokenizer
from openai_model import build_model
from x_clip import CLIP, TextTransformer, VisionTransformer, SPARC, ChexZero, TextEncoder, ImageEncoder
from dataloaders import get_mimic_dataloader, get_chexpert_test_loader_and_gt, get_mimic_test_loader_and_gt
from eval import eval

# multi gpu
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoConfig
from transformers import CLIPProcessor, CLIPModel, CLIPVisionConfig, CLIPVisionModel, AutoProcessor
import utils
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
    vision_transformer_config = {
        "dim":512,
        "image_size":args.img_size,
        "patch_size":32, #32
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


    # encoder configs
    vision_transformer_config_path = os.path.join(args.save_dir, 'vision_transformer_config.json')
    text_transformer_config_path = os.path.join(args.save_dir, 'text_transformer_config.json')


    # processor=None
    if args.model=='clip':
        # initialize encoders
        image_encoder = VisionTransformer(**vision_transformer_config)
        text_encoder = TextTransformer(**text_transformer_config)

        # save encoders configs
        with open(vision_transformer_config_path, 'w') as f:
            json.dump(vision_transformer_config, f, indent=2)
        with open(text_transformer_config_path, 'w') as f:
            json.dump(text_transformer_config, f, indent=2)

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
    elif args.model=='sparc':
        # initialize encoders
        url = "microsoft/BiomedVLP-CXR-BERT-general" # "microsoft/BiomedVLP-CXR-BERT-specialized"
        tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        text_encoder = AutoModel.from_pretrained(url, trust_remote_code=True)
        
        # processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch16')
        image_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16')
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        # text_transformer_config = AutoConfig.from_pretrained(url)
        # vision_transformer_config = AutoConfig.from_pretrained('facebook/dinov2-base')
        # image_encoder = AutoModel.from_pretrained('facebook/dinov2-base')

        # save encoders configs
        image_encoder.config.to_json_file(vision_transformer_config_path)
        text_encoder.config.to_json_file(text_transformer_config_path)
        args.img_size = 224
        local_weights = [0.1, 0.5, 1, 5, 10]

        clip = SPARC(
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            dim_image = 768, #512
            dim_text = 768, #512
            dim_latent= 768, #512
            decoupled_contrastive_learning = args.decoupled_contrastive_learning, # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
            extra_latent_projection = args.extra_latent_projection, # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_visual_ssl = args.use_visual_ssl, # whether to do self supervised learning on images
            use_mlm = args.use_mlm, # use masked language learning (MLM) on text (DeCLIP)
            text_ssl_loss_weight = args.text_ssl_loss_weight, # weight for text MLM loss
            image_ssl_loss_weight = args.image_ssl_loss_weight # weight for image self-supervised learning loss
        ).to('cuda')

    # use multi-GPU
    if args.multi_gpu:
        # utils.init_distributed_mode(args)
        utils.ddp_setup()
        args.gpu_id = int(os.environ['LOCAL_RANK'])
        clip = DistributedDataParallel(clip, device_ids=[args.gpu_id], find_unused_parameters=True) # device_ids=typically single list that model lives on

    # data loader
    train_loader = get_mimic_dataloader(mimic_root=args.mimic_root,
                                    split='train',
                                    shuffle=True,
                                    # transform=processor,
                                    batch_size=args.batch_size,
                                    img_size = args.img_size,
                                    multi_gpu=args.multi_gpu)

    optimizer = torch.optim.SGD(clip.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.AdamW(clip.parameters(), lr=0.004, weight_decay=0.001) #original clip: weight_decay 0.1

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1.1e-4)

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # resume the training code if output dir already exists
    save_path = os.path.join(args.save_dir, f"clip_last_checkpoint.pt")
    num_epoch = 0
    if os.path.exists(save_path):
        # import ipdb; ipdb.set_trace()
        saved_dict = torch.load(save_path)
        clip.module.load_state_dict(saved_dict['model']) if args.multi_gpu else clip.load_state_dict(saved_dict['model'])
        optimizer.load_state_dict(saved_dict['optimizer'])
        if args.use_fp16 is not None and saved_dict['scaler'] is not None:
            fp16_scaler.load_state_dict(saved_dict['scaler']) 
        scheduler.load_state_dict(saved_dict['scheduler'])
        num_epoch = saved_dict['epoch']
        print('Resuming training from snapshot at Epoch: {} from {}'.format(num_epoch, save_path))

    # train
    for epoch in range(num_epoch, args.epochs):
        image_encoder.train()
        text_encoder.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch} training', colour='green')):
            # import ipdb; ipdb.set_trace()
            images = batch['img']
            texts = batch['txt']
            if args.model=='sparc':
                texts = tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                                    add_special_tokens=True,
                                                    padding='max_length', #'longest'
                                                    max_length=77,
                                                    truncation=True,
                                                    return_tensors='pt')
                texts = texts['input_ids'].to('cuda')
                # images = images['pixel_values'][0].cuda() #dinov2
                # images = images['pixel_values'].squeeze(1).cuda() #openai-clip
                images = batch['img'].to('cuda')
                # linearly increased loss weight
                lw_idx = int(epoch / args.epochs * len(local_weights))
                local_weight = local_weights[lw_idx]

            else:
                texts = preprocess_text(texts).to('cuda')
                images = batch['img'].to('cuda')
                local_weight = 0.5

            with torch.cuda.amp.autocast(fp16_scaler is not None):    
                loss = clip(
                    text=texts,
                    image=images,
                    freeze_image_encoder=False,
                    freeze_text_encoder=False,
                    return_loss=True,
                    local_weight=local_weight
                )
            
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss['loss'].backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss['loss']).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(clip, args.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            scheduler.step()
            
            if args.log_every > 0:
                if i % args.log_every == 0:
                    example_ct = args.batch_size * i + epoch * len(train_loader) * args.batch_size
                    # wandb.log({'loss': loss, 'global_loss': gloss, 'local_loss': closs}, step=example_ct)
                    wandb.log(loss, step=example_ct)
            
            if torch.isnan(loss['loss']).sum()>0:
                raise ValueError('there is Nan value in loss')
            # if i>10:
            #     break
        
        # Eval every epoch
        ''' 
        texts = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        
        chexpert_test_loader, chexpert_test_gt = get_chexpert_test_loader_and_gt()
        mean_auroc, aurocs = eval(image_encoder, text_encoder, chexpert_test_loader, texts, chexpert_test_gt)
        aurocs = {f"CheXpert_{k}":v for k,v in aurocs.items()}
        # print(mean_auroc)
        # pprint(aurocs)
        if args.log_every > 0:
            example_ct = args.batch_size * i + epoch * len(train_loader) * args.batch_size
            wandb.log({'CheXpert_mean_auroc': mean_auroc, **aurocs}, step=example_ct)
            
        mimic_test_loader, mimic_test_gt = get_mimic_test_loader_and_gt(mimic_root=args.mimic_root)
        mean_auroc, aurocs = eval(image_encoder, text_encoder, mimic_test_loader, texts, mimic_test_gt)
        aurocs = {f"MIMIC_{k}":v for k,v in aurocs.items()}
        # print(mean_auroc)
        # pprint(aurocs)
        if args.log_every > 0:
            example_ct = args.batch_size * i + epoch * len(train_loader) * args.batch_size
            wandb.log({"MIMIC_mean_auroc": mean_auroc, **aurocs}, step=example_ct)
        '''
        
        # Save checkpoint every epoch
        save_dict = {
                    'model': clip.module.state_dict() if args.multi_gpu else clip.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(), 
                    'scaler': fp16_scaler.state_dict() if args.use_fp16 else None, 
                    'epoch': epoch+1
                    }
        save_path = os.path.join(args.save_dir, f"clip_last_checkpoint.pt")
        torch.save(save_dict, save_path)
        # sd = clip.visual_transformer.state_dict()
        # save_path = os.path.join(args.save_dir, f"image_encoder_epoch{epoch}.pt")
        # torch.save(sd, save_path)
        # sd = clip.text_transformer.state_dict()
        # save_path = os.path.join(args.save_dir, f"text_encoder_epoch{epoch}.pt")
        # torch.save(sd, save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='x-clip')
    parser.add_argument('--mimic_root', type=str, default='/home/wonjun/data/mimic-cxr-jpg-resized512')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=256, help='224 for sparc')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--model', type=str, choices=['clip', 'sparc'], default='clip')
    parser.add_argument('--use_fp16', action='store_true', default=False, help='whether or not to use half precision for training')
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter gradient norm if using gradient clipping
                            Clipping with norm .3~1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--multi_gpu', action='store_true', default=False, help='use multi_gpu')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", '--local-rank', default=0, type=int, help="Please ignore and do not set this argument.")
    
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
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    train_args_path = os.path.join(args.save_dir, 'train_args.json')
    with open(train_args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    if args.log_every > 0:
        run = wandb.init(project=args.project_name, config=vars(args))
    main(args)