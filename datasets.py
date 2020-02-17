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


class LazyTextDataset(Dataset):
    def __init__(self, metadata_path,
                 max_tokens=140,
                 all_chars='abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 strip_punctuations=False):
        with open(metadata_path, 'r') as f:
            d = json.load(f)
        self.data_path = d['data_path']
        self.length = d['length']
        self.offset = d['offset']

        self.max_tokens = max_tokens
        self.all_chars = all_chars
        self.strip_punctuations = strip_punctuations

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
            return self._parse_row(row)

    def _parse_row(self, row):
        s = preprocess_text(row[0], self.strip_punctuations)
        return onehot_encode(s, self.max_tokens, self.all_chars), \
            get_label(row[1])


def get_label(rating):
    rating = int(rating)
    if rating <= 2:  # Bad
        return 0
    elif rating == 3:  # Average
        return 1
    elif rating <= 5:  # Good
        return 2
    else:
        raise ValueError(f"Rating {rating} is not in the range [1, 5]")


def preprocess_text(s, strip_punctuations):
    s = unicodeToAscii(s.lower())
    # s = s.replace(" ", "")  # Remove spaces
    if strip_punctuations:
        s = s.translate(str.maketrans('', '', string.punctuation))
    return s


def onehot_encode(s, max_tokens, all_chars):
    start = time.time()

    char_idxs = []
    for ch in s:
        ch_idx = all_chars.find(ch)
        if ch_idx > -1:
            char_idxs.append(ch_idx)

    x = torch.zeros(max_tokens, len(all_chars))
    x[torch.arange(min(len(char_idxs), max_tokens)),
        char_idxs[:max_tokens]] = 1
#         print(f"One-hot encoding took {time.time()-start}")
    return x


def onehot_decode(x, all_chars):
    assert len(x.shape) == 2
    _, idxs = np.where(x == 1.)
    return "".join([all_chars[i] for i in idxs])


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
