import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from transformers import DistilBertTokenizer

import config as Config
from dataset import FashionIQDataset, get_transforms
from model import CLIPModel
from utils import AvgMeter, get_lr
from preprocess import preprocess_dataset

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = FashionIQDataset(
        list(zip(dataframe["target"], dataframe["candidate"])),
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        shuffle=True if mode == "train" else False,
        drop_last = True
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(Config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["target_image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(Config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["target_image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df = preprocess_dataset("train")
    valid_df = preprocess_dataset("val")
    tokenizer = DistilBertTokenizer.from_pretrained(Config.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = CLIPModel().to(Config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=Config.patience, factor=Config.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(Config.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
