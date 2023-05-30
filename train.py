import warnings
import logging
import colorlog
from multiprocessing import cpu_count

from pathlib import Path
from functools import partial

import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader

from fastcore.xtras import Path  # for ls

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import default_data_collator

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import PearsonCorrCoef
from composer.models.huggingface import HuggingFaceModel
from composer import Trainer
from composer.metrics import CrossEntropy


# ignoring warnings
warnings.filterwarnings("ignore")

# setting up logging
# TODO: Fix color logging
logger = colorlog.getLogger("TRAIN")
logger.setLevel(colorlog.INFO)


def process_df(df: pd.DataFrame, train: bool = True):
    df["input"] = (
        "TEXT1: " + df.context + "; TEXT2: " + df.target + "; ANC1: " + df.anchor
    )
    dataset = Dataset.from_pandas(df)
    if train:
        dataset = dataset.rename_columns({"score": "labels"})
    return dataset


def tokenize_func(batch, tokenizer):
    return tokenizer(
        batch["input"],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # printing the configurations
    logger.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    # loading the dataset
    logger.info("Loading the dataset")
    path = Path(cfg.data.path)
    train_df = pd.read_csv(path / "train.csv")
    test_df = pd.read_csv(path / "test.csv")

    # processing the dataset
    logger.info("Processing the dataset")
    train_ds = process_df(train_df)
    eval_ds = process_df(test_df, train=False)
    print(train_ds)

    # tokenizing the dataset
    logger.info("Tokenizing the dataset")
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.checkpoint)

    tokenize = partial(tokenize_func, tokenizer=tokenizer)
    train_tok_ds = train_ds.map(tokenize, batched=True, batch_size=None)
    eval_tok_ds = eval_ds.map(tokenize, batched=True, batch_size=None)
    print(train_tok_ds)
    print(
        f"""Sample Input :{train_tok_ds[0]["input"]}
        \nSample Input Ids: {train_tok_ds[0]["input_ids"]}
        \nSample Attention Mask: {train_tok_ds[0]["attention_mask"]}
        \nSample Label: {train_tok_ds[0]["labels"]}"""
    )

    # splitting the dataset
    logger.info("Splitting the dataset")
    train_dds = train_tok_ds.train_test_split(
        test_size=cfg.train.val_size, shuffle=True, seed=42
    )
    print(train_dds)
    print("Checking if the dataset lengths are similar")
    print(len(train_dds["train"][0]["input_ids"]))
    print(len(train_dds["train"][1]["input_ids"]))

    # creating PyTorch dataloaders
    logger.info("Creating PyTorch dataloaders")
    train_dl = DataLoader(
        train_dds["train"],
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    val_dl = DataLoader(
        train_dds["test"],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    # get a sample batch and print the first element
    print("Sample batch")
    batch = next(iter(val_dl))
    print(batch["input_ids"][0])
    print(batch["token_type_ids"][0])
    print(batch["attention_mask"][0])
    print(batch["labels"][0])

    # loading the model
    logger.info("Loading the model and training configurations")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.train.checkpoint, num_labels=1
    )
    pears_corr = PearsonCorrCoef(num_outputs=1)
    composer_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        # metrics=[CrossEntropy(), pears_corr],
        eval_metrics=[CrossEntropy(), pears_corr],
        use_logits=True,
    )
    # setup optimizer and scheduler
    optimizer = AdamW(
        params=composer_model.parameters(),
        lr=cfg.train.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=cfg.train.wd,
    )
    # TODO: Cosine/FitOneCycleLR
    linear_lr_decay = LinearLR(
        optimizer, start_factor=1.0, end_factor=0, total_iters=150
    )

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        max_duration=cfg.train.max_duration,
        optimizers=optimizer,
        schedulers=[linear_lr_decay],
        device="gpu",
        precision="amp_fp16",
        # seed=17,
    )

    # training the model
    logger.info("Training the model")
    trainer.fit()
    # with torch.no_grad():
    #     batch = {k: v.to("cuda") for k, v in batch.items()}
    #     outputs = model(**batch)
    #     logits = outputs.logits
    # print(logits.shape)
    # print(batch["labels"].shape)
    # print(pears_corr.to("cuda")(logits, batch["labels"]))


if __name__ == "__main__":
    train()
