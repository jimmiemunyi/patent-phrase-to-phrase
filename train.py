import warnings
import logging
import colorlog
from multiprocessing import cpu_count

from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader

from fastcore.xtras import Path  # for ls

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import default_data_collator

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import PearsonCorrCoef, MeanSquaredError
from composer.models.huggingface import HuggingFaceModel
from composer.loggers import WandBLogger
from composer import Trainer


# ignoring warnings
warnings.filterwarnings("ignore")

# setting up logging
# TODO: Fix color logging
logger = colorlog.getLogger("TRAIN")
logger.setLevel(colorlog.INFO)


def process_df(df: pd.DataFrame, sep_token, tokenizer, train: bool = True):
    df["section"] = df.context.str[0]
    df["sectok"] = "[" + df.section + "]"
    sectoks = list(df.sectok.unique())
    print(f"Section Tokens: {sectoks}")
    if train:
        tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
    df["input"] = (
        df.sectok
        + sep_token
        + df.context
        + sep_token
        + df.anchor.str.lower()
        + sep_token
        + df.target
    )
    dataset = datasets.Dataset.from_pandas(df)
    if train:
        dataset = dataset.rename_columns({"score": "labels"})
    return dataset


def create_val_split(df: pd.DataFrame, val_prop: float = 0.2, seed: int = 42):
    anchors = df.anchor.unique()
    np.random.seed(seed)
    np.random.shuffle(anchors)
    print(f"Sample Anchors: {anchors[:5]}")
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


def predict(trainer, test_dl):
    preds = trainer.predict(test_dl)[0]["logits"].numpy().astype(float)
    preds = np.clip(preds, 0, 1)
    preds = preds.round(2)
    preds = preds.squeeze()

    return preds


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # printing the configurations
    logger.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    # loading the dataset
    logger.info("Loading the dataset")
    path = Path(cfg.data.path)
    train_df = pd.read_csv(path / "train.csv")
    test_df = pd.read_csv(path / "test.csv")

    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.checkpoint)
    # sep_token = tokenizer.sep_token
    sep_token = " [s] "

    # processing the dataset
    logger.info("Processing the dataset")
    train_ds = process_df(train_df, sep_token, tokenizer)
    eval_ds = process_df(test_df, sep_token, tokenizer, train=False)
    print(train_ds[0])

    print(f"Tokenizer Special Tokens: {tokenizer.all_special_tokens}")

    # tokenizing the dataset
    logger.info("Tokenizing the dataset")
    inps = "anchor", "target", "context"
    # TODO: Clean up this section
    tokenize = partial(tokenize_func, tokenizer=tokenizer)
    train_tok_ds = train_ds.map(
        tokenize,
        batched=True,
        batch_size=None,
        remove_columns=inps + ("id", "section"),
    )
    eval_tok_ds = eval_ds.map(
        tokenize,
        batched=True,
        batch_size=None,
        remove_columns=inps + ("id", "section"),
    )
    print("Sample tokenized dataset")
    print(train_tok_ds[0])

    # splitting the dataset
    trn_idxs, val_idxs = create_val_split(train_df, val_prop=cfg.train.val_size)
    logger.info("Splitting the dataset")
    train_dds = datasets.DatasetDict(
        {"train": train_tok_ds.select(trn_idxs), "test": train_tok_ds.select(val_idxs)}
    )
    print(train_dds)
    print("Checking if the dataset lengths are similar")
    print(
        len(train_dds["train"][0]["input_ids"])
        == len(train_dds["train"][1]["input_ids"])
    )

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
    test_dl = DataLoader(
        eval_tok_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    # loading the model
    logger.info("Loading the model and training configurations")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.train.checkpoint, num_labels=1
    )
    model.resize_token_embeddings(len(tokenizer))
    pears_corr = PearsonCorrCoef(num_outputs=1)
    # mse_metric = LossMetric(loss_function=nn.MSELoss(reduction="sum"))
    mse_metric = MeanSquaredError()
    composer_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        metrics=[pears_corr],
        eval_metrics=[mse_metric, pears_corr],
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
    # extracting from 4ep
    epochs = int(cfg.train.max_duration.split("ep")[0])

    one_cycle_lr = OneCycleLR(
        optimizer,
        max_lr=cfg.train.lr,
        steps_per_epoch=len(train_dl),
        epochs=epochs,
    )

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_logger = WandBLogger(
        project="patent-phrase-to-phrase",
    )
    if cfg.train.run_name:
        run_name = cfg.train.run_name
    else:
        run_name = cfg.train.checkpoint.split("/")[-1]

    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        max_duration=cfg.train.max_duration,
        optimizers=optimizer,
        schedulers=[one_cycle_lr],
        device="gpu",
        precision="amp_fp16",
        loggers=[wandb_logger],
        run_name=run_name,
        step_schedulers_every_batch=True,
        # seed=17,
    )

    # training the model
    logger.info("Training the model")
    trainer.fit()

    # evaluating the model
    if cfg.create_csv:
        logger.info("Evaluating the model")
        preds = predict(trainer, test_dl)
        submission = datasets.Dataset.from_dict({"id": eval_ds["id"], "score": preds})
        submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    train()
