import argparse
import os
import random

import numpy as np
import sklearn.model_selection as sk
import torch
from torch.utils.data import DataLoader, Subset

from dataset import VjepaDataset, collate_pad
from models.transformer_encoder_asl import TransformerASLEncoder


def stratified_split(dataset, val_frac=0.2, seed=42):
    records = dataset.records
    labels = records['gloss'].values
    idx = list(range(len(records)))
    train_idx, val_idx = sk.train_test_split(idx, test_size=val_frac, random_state=seed,
                                             stratify=labels)
    return train_idx, val_idx


def train_one_epoch(model, loader, optim, device):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total = 0
    running_loss = 0.0
    correct = 0
    for batch in loader:
        x, lengths, y = batch
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, lengths, y = batch
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    index_file_path = os.path.join('/home/dell/Desktop/ASLVideoTranslate/data/processed_videos', 'index.csv')
    processed_video_dir = os.path.join('/home/dell/Desktop/ASLVideoTranslate/data', 'processed_videos')
    
    ds = VjepaDataset(index_file_path, processed_video_dir)
    
    train_idx, val_idx = stratified_split(ds, val_frac=0.2)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pad)

    # infer input dim from first available sample
    sample_emb, _ = ds[0]
    input_dim = sample_emb.shape[1]
    num_classes = len(ds.gloss2idx)

    model = TransformerASLEncoder(input_dim=input_dim, num_classes=num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optim, device)
        val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch} train loss: {loss:.4f}  train acc: {train_acc:.4f}  val acc: {val_acc:.4f}')


if __name__ == '__main__':
    main()
