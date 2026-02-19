import argparse
import os
import time
import json

import numpy as np
import sklearn.model_selection as sk
import torch
from torch.utils.data import DataLoader, Subset

from dataset import VjepaDataset, collate_pad
from models.asl_classifier import GlossClassifier


def create_vocab_file(dataset, output_path):
    """
    Create vocabulary JSON file from dataset
    
    Args:
        dataset: VjepaDataset instance
        output_path: where to save vocab.json
    """
    gloss_to_idx = dataset.gloss2idx
    idx_to_gloss = {str(v): k for k, v in gloss_to_idx.items()}
    
    vocab_data = {
        'gloss_to_idx': gloss_to_idx,
        'idx_to_gloss': idx_to_gloss,
        'num_classes': len(gloss_to_idx)
    }
    
    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, indent=4)
    
    print(f"✓ Vocabulary saved to {output_path}")

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
    for batch_idx, batch in enumerate(loader):
        x, lengths, y = batch
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        if torch.isnan(x).any():
            print(f"WARNING: NaN detected in input batch {batch_idx}")
            continue

        logits = model(x)

        if torch.isnan(logits).any():
            print(f"WARNING: NaN detected in logits batch {batch_idx}")
            continue

        loss = loss_fn(logits, y)

        if torch.isnan(loss):
            print(f"WARNING: NaN loss at batch {batch_idx}, skipping")
            continue

        optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            
            logits = model(x)
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

    index_file_path = os.path.join('/home/dell/Desktop/ASLVideoTranslate/data/selected_videos', 'index.csv')
    processed_video_dir = os.path.join('/home/dell/Desktop/ASLVideoTranslate/data', 'selected_videos')
    
    ds = VjepaDataset(index_file_path, processed_video_dir)
    
    train_idx, val_idx = stratified_split(ds, val_frac=0.2)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pad)

    sample_emb, _ = train_ds[0]
    input_dim = sample_emb.shape[1]
    num_classes = len(ds.gloss2idx)

    model = GlossClassifier(
        vjepa_dim=input_dim, 
        num_classes=num_classes,
        dropout=0.2  
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 
        T_max=args.epochs,
        eta_min=1e-6
    )
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optim, device)
        val_acc = evaluate(model, val_loader, device)

        scheduler.step()
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'  Train Loss: {loss:.4f}  Train Acc: {train_acc:.4f}')
        print(f'  Val Acc: {val_acc:.4f}')
        print(f'  LR: {optim.param_groups[0]["lr"]:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_dir = '/home/dell/Desktop/ASLVideoTranslate/models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'gloss_classifier_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'val_acc': val_acc,
            }, model_path)
            print(f'  ✓ Saved best model (val_acc: {val_acc:.4f})')
    
    vocab_path = os.path.join(model_dir, "vocab.json")
    create_vocab_file(ds, vocab_path)

    print(f'\nTraining complete! Best val acc: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()