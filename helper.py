import re
from pathlib import Path
from fastcore.xtras import Path

import numpy as np
import pandas as pd
from functools import partial

import datasets
from transformers import (
    AutoTokenizer,
)

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import default_data_collator

from composer.loggers import WandBLogger
from composer import Trainer

def get_cpc_texts(cpc_path: Path):
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in (cpc_path/'CPCSchemeXML202105').ls():
        result = re.findall(pattern, file_name.name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(cpc_path/f'CPCTitleList202202/cpc-section-{cpc}_20220201.txt') as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        if cpc == 'C':
            cpc_result = 'C' + cpc_result
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results


def process_df(df, sep_token):
    df["section"] = df.context.str[0]
    df["sectok"] = "[" + df.section + "]"
    sectoks = list(df.sectok.unique())
    df["input"] = (
        df.sectok
        + df.anchor.str.lower()
        + sep_token
        + df.target
        + sep_token
        + df.context_text.str.lower()
    )
    return df, sectoks


def create_val_split(df: pd.DataFrame, val_prop: float = 0.2, seed: int = 42):
    anchors = df.anchor.unique()
    np.random.seed(seed)
    np.random.shuffle(anchors)
    val_sz = int(len(anchors) * val_prop)
    val_anchors = anchors[:val_sz]
    is_val = np.isin(df.anchor, val_anchors)
    idxs = np.arange(len(df))
    val_idxs = idxs[is_val]
    trn_idxs = idxs[~is_val]

    return trn_idxs, val_idxs


def tokenize_func(batch, tokenizer):
    return tokenizer(
        batch["input"],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )


def tokenize_and_split(df, tokenize_func, train=True):
    inps = "anchor", "target", "context"
    dataset = datasets.Dataset.from_pandas(df)
    tok_dataset = dataset.map(
        tokenize_func,
        batched=True,
        batch_size=None,
        remove_columns=inps + ("id", "input", "section", "sectok")
    )
    if train:
        tok_dataset = tok_dataset.rename_columns({"score": "labels"})
        trn_idxs, val_idxs = create_val_split(df)
        tok_dataset = datasets.DatasetDict(
        {"train": tok_dataset.select(trn_idxs), "test": tok_dataset.select(val_idxs)}
    )
    
    return tok_dataset

def create_dataloaders(tok_ds, bs, train=True):
    if train:
        train_dl = DataLoader(
            tok_ds["train"],
            batch_size=bs,
            shuffle=True,
            collate_fn=default_data_collator,
            drop_last=True
        )
        val_dl = DataLoader(
            tok_ds["test"],
            batch_size=bs,
            shuffle=False,
            collate_fn=default_data_collator,
            drop_last=False
        )

        return train_dl, val_dl
    else:
        test_dl = DataLoader(
            tok_ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        return test_dl

def prepare_data(train_df, tokenizer, sep_token, bs):
    train_df, sectoks = process_df(train_df, sep_token)
    tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
    tokenize = partial(tokenize_func, tokenizer=tokenizer)
    train_tok_ds = tokenize_and_split(train_df, tokenize)
    train_dl, val_dl = create_dataloaders(train_tok_ds, bs)
    
    return train_dl, val_dl

def prepare_optimizer_and_scheduler(composer_model, lr, wd, epochs, train_dl):
    optimizer = AdamW(
        params=composer_model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=wd,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_dl),
        epochs=epochs,
    )
    
    return optimizer, scheduler

def prepare_trainer(composer_model, 
                    optimizer, 
                    scheduler, 
                    train_dl, 
                    val_dl, 
                    epochs, 
                    run_name):
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        max_duration=f"{epochs}ep",
        optimizers=optimizer,
        schedulers=[scheduler],
        loggers=[WandBLogger(project="patent-phrase-to-phrase")],
        run_name=run_name,
        device="gpu",
        precision="amp_fp16",
        step_schedulers_every_batch=True,
        # seed=17,
    )
    
    return trainer

def fit(train_df,
        checkpoint,
        model, 
        run_name, 
        bs=32, 
        lr=2e-5, 
        wd=0.01, 
        epochs=4, 
        num_labels=1, 
        sep_token="[SEP]"):
    # preparing data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dl, val_dl = prepare_data(train_df, tokenizer, sep_token, bs)
    
    
    # preparing optimizer and scheduler
    optimizer, scheduler = prepare_optimizer_and_scheduler(model, 
                                                           lr, 
                                                           wd, 
                                                           epochs, 
                                                           train_dl)
    
    # preparing trainer
    trainer = prepare_trainer(model, 
                              optimizer, 
                              scheduler, 
                              train_dl, 
                              val_dl, 
                              epochs, 
                              run_name)
    
    # training
    trainer.fit()
    
    return trainer

def predict(trainer, test_dl):
    preds = trainer.predict(test_dl)[0]["logits"].numpy().astype(float)
    preds = np.clip(preds, 0, 1)
    preds = preds.round(2)
    preds = preds.squeeze()

    return preds