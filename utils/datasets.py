import pandas as pd
import random
import torch
import torchaudio
from datasets import Dataset, DatasetDict
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

DS_FACTOR, ALPHA = 335, 0.49

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
    def __init__(self, df, transform=False):
        self.transform = transform
        self.df = df

        #self.audio_path = dataset['audio_fpath']
        #self.speech = dataset['speech']
        #self.target = dataset['ds_tgts']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.df.audio_fpath.iloc[idx])
        #audio = self.speech[idx][0]
        if self.transform:
            pass
        return audio[0], torch.tensor(self.df.tgts.iloc[idx], dtype=torch.long)

def collator(batch):
    
    xx = [s[0] for s in batch]
    xx_ll = [len(s[0]) for s in batch]
    yy = [s[1] for s in batch]
    yy_ll = [len(s[1]) for s in batch]

    xx_ll = torch.tensor(xx_ll, dtype=torch.float)
    yy_ll = torch.tensor(yy_ll, dtype=torch.float)

    return pad_sequence(xx, batch_first=True), xx_ll, pad_sequence(yy, batch_first=True, padding_value=1), yy_ll