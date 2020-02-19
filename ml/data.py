import os
import unicodedata
import json
import csv
from io import StringIO
import time
import string

import numpy as np

import torch
from torch.utils.data import Dataset

import pandas as pd


class TextDataset(Dataset):
    def __init__(self, data_path,
                 seq_len=140,
                 all_chars='abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        self.data = pd.read_csv(data_path, quotechar='"',
                                usecols=["comment", "label"])
        self.seq_len = seq_len
        self.all_chars = all_chars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return to_feature_vector(row['comment'], self.all_chars, self.seq_len), \
            rating_to_id(row['label'])


class LazyTextDataset(Dataset):
    def __init__(self, metadata_path,
                 seq_len=140,
                 all_chars='abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        with open(metadata_path, 'r') as f:
            d = json.load(f)
        self.data_path = d['data_path']
        self.length = d['length']
        self.offset = d['offset']

        self.seq_len = seq_len
        self.all_chars = all_chars

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = time.time()
        with open(self.data_path, 'r') as f:
            f.seek(self.offset[str(idx + 1)])  # First line is for col names
            buffer = StringIO(f.readline())
            reader = csv.reader(buffer, quotechar='"')
            row = next(reader)
#             print(f"Getting item took {time.time() - start}")
        return to_feature_vector(row[0], self.all_chars, self.seq_len), \
            rating_to_id(row[-1])


def id_to_rating(id):
    d = {0: 'good', 1: 'average', 2: 'bad'}
    return d[id]


def rating_to_id(label):
    d = {
        "good": 0,
        "average": 1,
        "bad": 2,
    }
    return d[label]


def to_feature_vector(sentence, all_chars, seq_len):
    s = preprocess_text(sentence)
    return onehot_encode(s, all_chars, seq_len)


def preprocess_text(s):
    s = unicodeToAscii(s.lower())
    return s


def onehot_encode(s, all_chars, seq_len):
    start = time.time()

    char_idxs = []
    for ch in s:
        ch_idx = all_chars.find(ch)
        if ch_idx > -1:
            char_idxs.append(ch_idx)

    x = torch.zeros(len(all_chars), seq_len)
    x[char_idxs[:seq_len],
      torch.arange(min(len(char_idxs), seq_len))] = 1
#         print(f"One-hot encoding took {time.time()-start}")
    return x


def onehot_decode(x, all_chars):
    assert len(x.shape) == 2
    char_idxs, t = np.where(x == 1.)
    char_idxs = char_idxs[np.argsort(t)]
    return "".join([all_chars[i] for i in char_idxs])


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def generate_dataset_metadata(data_path):
    offset_dict = {}
    with open(data_path, 'r') as f:
        offset_dict[0] = 0
        for i in range(1, 7026999):
            f.readline()  # move over header
            offset = f.tell()
            offset_dict[i] = offset
    metadata = {
        'data_path': os.path.abspath(data_path),
        'length': 7026999,
        'offset': offset_dict,
    }
    with open("trustpilot_metadata.json", 'w') as f:
        json.dump(metadata, f)
