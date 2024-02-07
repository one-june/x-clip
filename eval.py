#%%
import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt

import torch

from sklearn.metrics import roc_auc_score

from x_clip import CLIP, TextTransformer, VisionTransformer

from dataloaders import (get_mimic_test_loader_and_gt,
                         get_chexpert_test_loader_and_gt)
from simple_tokenizer import SimpleTokenizer, preprocess_text

from Q_utils import get_parameter_count


#%%
def get_img_txt_similarity(loader, texts, image_encoder, text_encoder):
    """
    loader: torch.utils.data.DataLoader
        DataLoader that loads dicts that include 'img' key
        e.g. {'img': ..., 'txt': ...}
        The value for 'img' key are torch.Tensors with
        shape [batch_size, c, h, w] (e.g. [32, 3, 256, 256])
    
    labels: list
        List of strings to assess
    """
    text_encoder = text_encoder.to('cpu')
    with torch.no_grad():
        zeroshot_weights = []
        for text in texts:
            t = preprocess_text([text], context_length=77).to('cpu')
            class_embeddings = text_encoder(t)[:, 0, :]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    
    image_encoder = image_encoder.to('cpu')
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc='getting similarities', colour='cyan')):
            images = data['img'].to('cpu')
            
            image_features = image_encoder(images)[:,0,:] # CLS embedding
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            logits = image_features @ zeroshot_weights
            logits = np.squeeze(logits.numpy(), axis=0)
            
            y_pred.append(logits)

    y_pred = np.array(y_pred)
    
    return y_pred


def eval(image_encoder, text_encoder, loader, texts, gt_labels):
    image_encoder.eval()
    text_encoder.eval()
    with torch.no_grad():
        pos_texts = texts
        pos_pred = get_img_txt_similarity(loader=loader,
                                        texts=pos_texts,
                                        image_encoder=image_encoder,
                                        text_encoder=text_encoder)
        
        neg_texts = [f"No {t}" for t in texts]
        neg_pred = get_img_txt_similarity(loader=loader,
                                        texts=neg_texts,
                                        image_encoder=image_encoder,
                                        text_encoder=text_encoder)
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    preds = np.exp(pos_pred) / sum_pred
    
    mean_auroc = roc_auc_score(gt_labels, preds, average='weighted')
    aurocs = roc_auc_score(gt_labels, preds, average=None)
    aurocs = {label:auroc for label, auroc in zip(texts, aurocs)}
    return mean_auroc, aurocs    



# %%
if __name__ == '__main__':
    image_encoder_config_path = 'outputs/clip_bsz32_run1/vision_transformer_config.json'
    text_encoder_config_path = 'outputs/clip_bsz32_run1/text_transformer_config.json'
    # image_encoder_config_path = 'outputs/flip0.5_bsz32_run0/vision_transformer_config.json'
    # text_encoder_config_path = 'outputs/flip0.5_bsz32_run0/text_transformer_config.json'
    with open(image_encoder_config_path, 'r') as f:
        image_encoder_config = json.load(f)
    with open(text_encoder_config_path, 'r') as f:
        text_encoder_config = json.load(f)
    
    image_encoder = VisionTransformer(**image_encoder_config)
    text_encoder = TextTransformer(**text_encoder_config)
    
    image_encoder_ckpt_path = 'outputs/clip_bsz32_run1/image_encoder_epoch0.pt'
    text_encoder_ckpt_path = 'outputs/clip_bsz32_run1/text_encoder_epoch0.pt'
    # image_encoder_ckpt_path = 'outputs/flip0.5_bsz32_run0/image_encoder_epoch6.pt'
    # text_encoder_ckpt_path = 'outputs/flip0.5_bsz32_run0/text_encoder_epoch6.pt'    
    
    print(f"Loading image encoder ckpt from\n {image_encoder_ckpt_path}")
    sd = torch.load(image_encoder_ckpt_path)
    msg = image_encoder.load_state_dict(sd)
    print(msg)
    print(f"Loading text encoder ckpt from\n {text_encoder_ckpt_path}")
    sd = torch.load(text_encoder_ckpt_path)
    msg = text_encoder.load_state_dict(sd)
    print(msg)
    
    
    texts = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    mimic_test_loader, mimic_gt = get_mimic_test_loader_and_gt(label_columns=texts)
    chexpert_test_loader, chexpert_gt = get_chexpert_test_loader_and_gt(
        testset_h5_filepath='/home/wonjun/data/chexzero-format/chexpert/chexpert_test.h5',
        testset_gt_csv_path='/home/wonjun/data/chexzero-format/chexpert/chexpert-test-groundtruth.csv'
    )

    mean_auroc, aurocs = eval(image_encoder, text_encoder, chexpert_test_loader, texts, chexpert_gt)
    print("Weighted mean AUROC: ", mean_auroc)
    pprint(aurocs)
    mean_auroc, aurocs = eval(image_encoder, text_encoder, mimic_test_loader, texts, mimic_gt)
    print("Weighted mean AUROC: ", mean_auroc)
    pprint(aurocs)
# %%
