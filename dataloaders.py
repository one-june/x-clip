#%%
import os
import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from simple_tokenizer import SimpleTokenizer
from Q_utils import extract_impression, extract_findings

chexzero_transforms = T.Compose(
        [
            T.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        ]
    )

class MIMICDataset(Dataset):
    def __init__(self, mimic_root, df, transform=None):
        self.mimic_root = mimic_root
        # self.df = pd.read_csv(df_path)
        self.df = df
        
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        
        txt_path = os.path.join(self.mimic_root, 'reports', 'files',
                                f"p{str(int(row['subject_id']))[:2]}",
                                f"p{str(int(row['subject_id']))}",
                                f"s{str(int(row['study_id']))}.txt")
        with open(txt_path, 'r') as handle:
            report = handle.read()
        # findings = extract_findings(report)
        impression = extract_impression(report)
        # txt = preprocess_text(impression) # tokenize
        txt = impression
        
        jpg_path = os.path.join(self.mimic_root, 'files',
                                f"p{str(int(row['subject_id']))[:2]}",
                                f"p{str(int(row['subject_id']))}",
                                f"s{str(int(row['study_id']))}",
                                row['dicom_id']+'.jpg')
        # CheXzero's data_process.img_to_hdf5
        img = cv2.imread(jpg_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # data_process.preprocess
        desired_size = 320
        old_size = img.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size, Image.LANCZOS)
        new_img = Image.new('L', (desired_size, desired_size))
        new_img.paste(img, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        img = new_img
        # CheXzero's train.CXRDataset.__getitem__
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        if self.transform:
            img = self.transform(img)
        
        out = {"img": img, "txt": txt}
        return out
    

def get_mimic_dataloader(mimic_root='/home/wonjun/data/mimic-cxr-jpg-resized512',
                            split='train',
                            shuffle=True,
                            transform=chexzero_transforms,
                            batch_size=1):
    
    split_df_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-split.csv')
    metadata_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-metadata.csv')
    metadata = pd.read_csv(metadata_path)
    df = pd.read_csv(split_df_path)
    df = df[df['split']==split]
    df = pd.merge(df, metadata, how='left', on=['subject_id', 'study_id', 'dicom_id'])
    df = df[df['ViewPosition'].isin(['AP', 'PA'])]
    # df = df.sample(500)
    
    dataset = MIMICDataset(mimic_root = mimic_root,
                           df = df,
                           transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=os.cpu_count(),
                        drop_last=False)
    return loader

def get_mimic_test_loader_and_gt(mimic_root='/home/wonjun/data/mimic-cxr-jpg-resized512',
                                transform=chexzero_transforms,
                                label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']):
    
    split_df_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-split.csv')
    metadata_df_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-metadata.csv')
    metadata_df = pd.read_csv(metadata_df_path)
    labels_df_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-negbio.csv')
    labels_df = pd.read_csv(labels_df_path)
    
    df = pd.read_csv(split_df_path)
    df = df[df['split']=='test']
    df = pd.merge(df, metadata_df, how='left', on=['subject_id', 'study_id', 'dicom_id'])
    df = df[df['ViewPosition'].isin(['AP', 'PA'])]
    
    # Get test set dataloader
    dataset = MIMICDataset(mimic_root=mimic_root,
                           df=df,
                           transform=transform)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=os.cpu_count(),
                        drop_last=True)
    
    # Ground-truth labels
    df = pd.merge(df, labels_df, how='left', on=['subject_id', 'study_id'])
    df = df.fillna(0)
    df = df.replace(-1, 0)
    gt_labels = df[label_columns].values
    
    return loader, gt_labels
    
class ChexpertTestDataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample


def get_chexpert_test_loader_and_gt(testset_h5_filepath='/home/wonjun/data/chexzero-format/chexpert/chexpert_test.h5',
                                    testset_gt_csv_path='/home/wonjun/data/chexzero-format/chexpert/chexpert-test-groundtruth.csv',
                                    transform=chexzero_transforms,
                                    label_columns=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']):
    ds = ChexpertTestDataset(img_path=testset_h5_filepath,
                             transform=transform)
    loader = DataLoader(ds, shuffle=False)
    
    df = pd.read_csv(testset_gt_csv_path)
    df = df.fillna(0)
    df = df.replace(-1, 0)
    df = df.loc[:, label_columns]
    gt_labels = df.to_numpy()
    
    return loader, gt_labels