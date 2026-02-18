import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class VjepaDataset(Dataset):
    """Dataset for vjepa embeddings mapped to ASL gloss labels.

    Expects an index CSV with columns: `video_id`, `gloss`, and `path_to_npy_file`.
    The `path_to_npy_file` may be absolute or relative; when relative it's resolved
    against `processed_dir`.
    Embeddings are expected to have shape (1, 2048, 1408) or (2048, 1408) and
    will be converted to a sequence tensor of shape [T, D] (T=1408, D=2048).
    """

    def __init__(self, index_csv: str, processed_dir: str, selected_glosses: List[str] = None):
        df = pd.read_csv(index_csv)
        if 'video_id' not in df.columns or 'gloss' not in df.columns or 'path_to_npy_file' not in df.columns:
            raise ValueError('index_csv must contain columns `video_id`, `gloss`, and `path_to_npy_file`')

        if selected_glosses is None:
            top = df['gloss'].value_counts().nlargest(300).index.tolist()
            df = df[df['gloss'].isin(top)]
        else:
            df = df[df['gloss'].isin(selected_glosses)]

        df = df.drop_duplicates(subset=['video_id'])

        self.processed_dir = processed_dir
        self.records = df[['video_id', 'gloss']].reset_index(drop=True)

        glosses = sorted(self.records['gloss'].unique())
        self.gloss2idx = {g: i for i, g in enumerate(glosses)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.records.loc[idx]
        # prefer explicit path from CSV; resolve relative paths against processed_dir
        raw_path = str(row['path_to_npy_file'])
        # if os.path.isabs(raw_path):
        #     fname = raw_path
        # else:
        #     fname = os.path.join(self.processed_dir, raw_path)
        # if not os.path.exists(fname):
        #     raise FileNotFoundError(fname)

        arr = np.load(raw_path)
        # handle expected shapes: (1, C, T) or (C, T)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim != 2:
            raise ValueError(f'Unsupported embedding shape {arr.shape} for file {raw_path}')
        # arr shape is [C, T] -> transpose to [T, C]
        emb = torch.from_numpy(arr.T).float()
        label = self.gloss2idx[row['gloss']]
        return emb, label


def collate_pad(batch):
    """Collate fn that pads variable-length embeddings.

    Each item is (emb: Tensor[T, D], label: int)
    Returns: padded Tensor [B, T, D], lengths [B], labels [B]
    """
    embs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
    # pad_sequence expects list of [T, D] tensors and returns [maxT, B, D]
    padded = pad_sequence(embs, batch_first=True)  # [B, maxT, D]
    return padded, lengths, labels
