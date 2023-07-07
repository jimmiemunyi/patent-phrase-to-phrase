import warnings
import logging
import colorlog
from multiprocessing import cpu_count

from pathlib import Path

import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer

from composer.loggers import WandBLogger
from composer.core.passes import AlgorithmPass, sort_to_back
from composer.algorithms import GradientClipping


from composer import Trainer

from helper import (
    get_cpc_texts,
    prepare_data,
    prepare_optimizer_and_scheduler,
    predict
)

from awp import AWP
from model import PatentModel



# ignoring warnings
warnings.filterwarnings("ignore")

# setting up logging
# TODO: Fix color logging
logger = colorlog.getLogger("TRAIN")
logger.setLevel(colorlog.INFO)


class algo_pass(AlgorithmPass):
    def __init__(self, algorithms):
        self.algorithms = algorithms
        super().__init__()

    def __call__(self):
        return sort_to_back(self.algorithms, AWP)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # printing the configurations
    logger.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    # loading the dataset
    logger.info("Loading the dataset")
    path = Path(cfg.data.path)
    cpc_path = Path(cfg.data.cpc_path)

    train_df = pd.read_csv(path / "train.csv")
    test_df = pd.read_csv(path / "test.csv")

    logger.info("Processing and merging the cpc context_text")
    cpc_texts = get_cpc_texts(cpc_path)
    train_df['context_text'] = train_df['context'].map(cpc_texts)

    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.checkpoint)
    # sep_token = tokenizer.sep_token
    sep_token = cfg.data.sep_token

    # creating PyTorch dataloaders
    train_dl, val_dl = prepare_data(train_df, tokenizer, 
                                    sep_token=sep_token, bs=cfg.train.batch_size)
    # loading the model
    logger.info("Loading the model and training configurations")
    model = PatentModel(cfg.train.checkpoint, tokenizer)
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        model, 
        cfg.train.lr, 
        cfg.train.wd, 
        cfg.train.epochs, 
        train_dl)
    
    clipping_type = 'value' # can also be 'adaptive' or 'value'
    gc = GradientClipping(clipping_type=clipping_type, 
                          clipping_threshold=cfg.gc.clip_value)
    algorithms = [gc]
    if cfg.train.awp == 'true':
        awp = AWP(start_epoch=cfg.awp.start, 
                  adv_lr=cfg.awp.adv_lr, 
                  adv_eps=cfg.awp.adv_eps,)
        algorithms.append(awp)


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_logger = WandBLogger(
        project="patent-phrase-to-phrase",
    )
    if cfg.train.run_name:
        run_name = cfg.train.run_name
    else:
        run_name = cfg.train.checkpoint.split("/")[-1]

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        max_duration=f"{cfg.train.epochs}ep",
        optimizers=optimizer,
        schedulers=[scheduler],
        loggers=[wandb_logger],
        # algorithm_passes=algo_pass(algorithms),
        device_train_microbatch_size='auto',
        run_name=run_name,
        algorithms=algorithms,
        device="gpu",
        precision="amp_fp16",
        step_schedulers_every_batch=True,
    )

    # training the model
    logger.info("Training the model")
    trainer.fit()

    # evaluating the model
    # if cfg.create_csv:
    #     logger.info("Evaluating the model")
    #     preds = predict(trainer, test_dl)
    #     submission = datasets.Dataset.from_dict({"id": eval_ds["id"], "score": preds})
    #     submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    train()
