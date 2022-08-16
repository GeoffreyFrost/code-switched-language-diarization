import os
import pandas as pd
import numpy as np
import random
import torch
import torchaudio
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

DS_FACTOR, ALPHA = 335, 0.49
melspec_extractor = torchaudio.transforms.MelSpectrogram(n_fft=400, hop_length=160, n_mels=23)

def down_sample_labels_fn(batch):

    tgts = torch.tensor(batch['tgts'], dtype=torch.float32).reshape(1, 1, 1, -1)
    ds_size = int((tgts.size(-1)/DS_FACTOR)*ALPHA)
    interp_legnths = (1,  ds_size)
    interp_tgts = F.interpolate(tgts, interp_legnths).squeeze()
    batch['ds_tgts'] = interp_tgts

    return batch

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio_fpath"])
    batch["speech"] = speech_array.to(torch.float16)
    batch["sampling_rate"] = sampling_rate

    return batch

def get_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    return df

def dataset_dict(df_trn, df_dev, df_tst):
    dataset = DatasetDict({'train': Dataset.from_pandas(df_trn),
                            'dev': Dataset.from_pandas(df_dev),
                            'test': Dataset.from_pandas(df_tst)})
    return dataset

class CSDataset(Dataset):
    def __init__(self, df, melspecs=False):
        self.df = df
        self.melspecs = melspecs
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if not self.melspecs: 
            audio, sr = torchaudio.load(self.df.audio_fpath.iloc[idx])
            return audio[0], torch.tensor(self.df.tgts.iloc[idx], dtype=torch.long)
        
        else:
            path = self.df.audio_fpath.iloc[idx].split('.')[0] + '.pt'
            if not os.path.exists(path):
                audio, sr = torchaudio.load(self.df.audio_fpath.iloc[idx])
                melspec = torch.log10(melspec_extractor(audio))
                # torch.save(melspec, path)

                return melspec[0].T, torch.tensor(self.df.tgts.iloc[idx], dtype=torch.long)

def collator(batch):
    
    xx = [s[0] for s in batch]
    xx_ll = [len(s[0]) for s in batch]
    yy = [s[1] for s in batch]
    yy_ll = [len(s[1]) for s in batch]

    xx_ll = torch.tensor(xx_ll, dtype=torch.float)
    yy_ll = torch.tensor(yy_ll, dtype=torch.float)

    return pad_sequence(xx, batch_first=True), xx_ll, pad_sequence(yy, batch_first=True, padding_value=0), yy_ll

def norm_binary_labels_func(x):
    x = norm_labels_func(x)
    x[x > 0] = 1.0
    return x

def norm_labels_func(x):
    x = x - 1.0
    return x

def read_pickle_df(path, norm_labels=True, binary_labels=True):
    df = pd.read_pickle(path)
    if norm_labels: 
        if binary_labels: df.tgts = df.tgts.map(norm_binary_labels_func)
        else: df.tgts = df.tgts.map(norm_labels_func)       
    return df

def load_dfs(data_df_root_dir, cs_pair, all_cs_pairs=False):

    if all_cs_pairs: 
        df_trn = read_pickle_df(os.path.join(data_df_root_dir, f"cs_{cs_pair}_trn.pkl"), binary_labels=False)
        df_dev = read_pickle_df(os.path.join(data_df_root_dir, f"cs_{cs_pair}_dev.pkl"), binary_labels=False)
    
    else:
        df_trn = read_pickle_df(os.path.join(data_df_root_dir, f"cs_{cs_pair}_trn.pkl"))
        df_dev = read_pickle_df(os.path.join(data_df_root_dir, f"cs_{cs_pair}_dev.pkl"))
    
    return df_trn, df_dev

def filter_mono_eng(df):
    indexs = []
    for tgts in tqdm(df.tgts): 
        indexs.append(np.mean(tgts) > 0.0)
    return df.loc[indexs]

def filter_code_for_switched_only(df):
    indexs = []
    for tgts in tqdm(df.tgts): 
        indexs.append(np.mean(np.absolute(np.gradient(tgts))) > 0.0)
    return df.loc[indexs]


def create_dataloaders(df_trn, df_dev, df_tst=None, melspecs=False, bs=1, num_workers=6):

    dataset_trn = CSDataset(df_trn, melspecs=melspecs)
    dataset_dev = CSDataset(df_dev, melspecs=melspecs)
    
    train_dataloader = DataLoader(dataset_trn, batch_size=bs, shuffle=True, collate_fn=collator, num_workers=num_workers)
    dev_dataloader = DataLoader(dataset_dev, batch_size=bs, collate_fn=collator, num_workers=num_workers)

    if df_tst != None:
        dataset_test = CSDataset(df_dev, melspecs=melspecs)
        test_dataloader = DataLoader(dataset_test, batch_size=bs, collate_fn=collator, num_workers=num_workers)
        return  train_dataloader, dev_dataloader, test_dataloader
    
    return train_dataloader, dev_dataloader