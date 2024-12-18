import os
import argparse
import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from transformers import ElectraForTokenClassification, AutoTokenizer

from datasets import Dataset
from utils import evaluate_fn
from train import train_fn

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="./outputs",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--repo_id", type=str, default="tinh2312/Electra-VNJob-NER",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to the config model to be used")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--task", type=str, default="train", choices=["train", "eval"],
                        help="Whether to train or evaluate the model")

    parser.add_argument("--train_data_path", type=str, default="./data/vnjob_train.csv",
                        help="Path to the training dataset")
    parser.add_argument("--val_data_path", type=str, default="./data/vnjob_val.csv",
                        help="Path to the training dataset")
        
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train the model for")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the model training")

    parser.add_argument("--resume_checkpoints", type=str, default="",
                        help="Path to the checkpoints to be used for resuming training")
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    return parser

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def main():
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(args.experiment_name + ".log")])
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    repo_id = args.repo_id
    model = ElectraForTokenClassification.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model.to(device)
    if args.pretrained_path != "":
        print(f"Load checkpoint from file : {args.pretrained_path}")
        checkpoints = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoints['model'])
    
    val_dataset = Dataset(args.val_data_path, tokenizer, "val")
    if args.task == "train":
        train_dataset = Dataset(args.train_data_path, tokenizer, "train")
        train_fn(model, train_dataset, val_dataset, args)
    else:
        val_loader = DataLoader(val_dataset, shuffle=True,
                            batch_size=args.batch_size, collate_fn=val_dataset.data_collator)
        
        print("Evaluate model..!")
        model.train(False)
        val_loss, val_acc, val_recall, val_precision, val_f1 = evaluate_fn(model, val_loader, epoch=0, epochs=0)
        val_info = f"Valuation  loss: {val_loss}, acc: {val_acc}, recall: {val_recall}, precision: {val_precision}, f1: {val_f1}"
        print(val_info)
        logging.info(val_info)
        print("")
        return
    
if __name__ == "__main__":
    main()