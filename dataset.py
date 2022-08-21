# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class SkinDataset():
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.df.loc[self.df.index[idx], 'hasher'] + ".jpg")
        image = Image.open(img_name)

        label = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick'] - 1
        if self.transform:
            image = self.transform(image)

        return image, label, fitzpatrick


def get_fitz_dataloaders(root, holdout_mode, batch_size, shuffle, partial_skin_types=[], partial_ratio=1.0):
    all_domains = [1, 2, 3, 4, 5, 6]

    train_dir = root + 'fitz17k_train_' + holdout_mode + '.csv'
    val_dir = root + 'fitz17k_val_' + holdout_mode + '.csv'
    test_dir = root + 'fitz17k_test_' + holdout_mode + '.csv'

    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    for s in all_domains:
        print("\ttrain: skin type", s, ":", len(train[train['fitzpatrick'] == s]))


    train = train.loc[train['fitzpatrick'] != -1]
    val = val.loc[val['fitzpatrick'] != -1]
    test = test.loc[test['fitzpatrick'] != -1]

    if len(partial_skin_types) > 0:
        train_1 = train.loc[~train['fitzpatrick'].isin(partial_skin_types)]
        train_2 = train.loc[train['fitzpatrick'].isin(partial_skin_types)]
        
        if partial_ratio > 0:
            try:
                train_2_partial, _, _, _ = train_test_split(
                    train_2,
                    train_2.low,
                    train_size=partial_ratio,
                    random_state=None, #4242
                    stratify=train_2.low)
            except:
                print("Unable to stratify -> skipped the stratification")
                train_2_partial, _, _, _ = train_test_split(
                    train_2,
                    train_2.low,
                    train_size=partial_ratio,
                    random_state=None, #4242
                    )
            train = pd.concat([train_1, train_2_partial])
            train.drop_duplicates(subset=['hasher'])
            train.reset_index(drop=True, inplace=True)
        else:
            train = train_1

        print("After partial skin type edit:")
        for s in all_domains:
            print("\ttrain: skin type", s, ":", len(train[train['fitzpatrick'] == s]))

    print("train size:", len(train))
    print("val size:", len(val))
    print("train skin types:", train.fitzpatrick.unique())
    print("val skin types:", val.fitzpatrick.unique())
    label_codes = sorted(list(train['label'].unique()))
    print("train skin conditions:", len(label_codes))
    label_codes1 = sorted(list(val['label'].unique()))
    print("val skin conditions:", len(label_codes1))

    transformed_train = SkinDataset(
        df=train,
        root_dir=root,
        transform=transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )

    transformed_val = SkinDataset(
        df=val,
        root_dir=root,
        transform=transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    transformed_test = SkinDataset(
        df=test,
        root_dir=root,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        transformed_train,
        batch_size=batch_size,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        transformed_val,
        batch_size=batch_size,
        shuffle=shuffle, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        transformed_test,
        batch_size=batch_size,
        shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader

