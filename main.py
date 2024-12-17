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
from utils import train_epoch, evaluate, save_checkpoints, load_checkpoints

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
    parser.add_argument("--val_data_path", type=str, default="./data/vnjob_test.csv",
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
    
    train_dataset = Dataset(args.train_data_path, tokenizer, "train")
    val_dataset = Dataset(args.val_data_path, tokenizer, "val")
    
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=args.batch_size, collate_fn=train_dataset.data_collator)
    val_loader = DataLoader(val_dataset, shuffle=True,
                            batch_size=args.batch_size, collate_fn=val_dataset.data_collator)
    
    if args.pretrained_path != "":
        print(f"Load checkpoint from file : {args.pretrained_path}")
        checkpoints = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoints['model'])
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scheduler_factor,
                                                     patience=args.scheduler_patience)
    
    list_train_loss, list_train_acc, list_val_loss, list_val_acc = [], [], [], []
    top_train_acc, top_val_acc = 0, 0
    lr_progress = []
    epochs = args.epochs 

    if args.resume_checkpoints != "":
        print(f"Resume training from file : {args.resume_checkpoints}")
        resume_epoch = load_checkpoints(model, optimizer, args.resume_checkpoints, resume=True)
    else:
        resume_epoch = 0
    if args.task == "eval":
        print("Evaluate model..!")
        model.train(False)
        val_loss, val_acc, val_recall, val_precision, val_f1 = evaluate(model, val_loader, epoch=0, epochs=0)
        val_info = f"[{epoch + 1}] Valuation  loss: {val_loss}, acc: {val_acc}, recall: {val_recall}, precision: {val_precision}, f1: {val_f1}"
        print(val_info)
        logging.info(val_info)
        print("")
        return

    Path(args.experiment_name).mkdir(parents=True, exist_ok=True)
    for epoch in range(resume_epoch, epochs, 1):
        train_loss, train_acc, train_recall, train_precision, train_f1 = train_epoch(model, train_loader, optimizer,
                                               scheduler, epoch=epoch, epochs=epochs)

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)

        model.train(False)
        val_loss, val_acc, val_recall, val_precision, val_f1 = evaluate(model, val_loader, epoch=epoch, epochs=epochs)
        model.train(True)

        list_val_loss.append(val_loss)
        list_val_acc.append(val_acc)

        if train_acc > top_train_acc:
            top_train_acc = train_acc
            save_checkpoints(model, optimizer, (args.experiment_name, epoch))

        if val_acc > top_val_acc:
            top_val_acc = val_acc
            save_checkpoints(model, optimizer, (args.experiment_name, epoch))

        train_info = f"[{epoch + 1}] TRAIN  loss: {train_loss}, acc: {train_acc}, recall: {train_recall}, precision: {train_precision}, f1: {train_f1}"
        val_info = f"[{epoch + 1}] Valuation  loss: {val_loss}, acc: {val_acc}, recall: {val_recall}, precision: {val_precision}, f1: {val_f1}"
        print(train_info)
        logging.info(train_info)
        print(val_info)
        logging.info(val_info)
        print("")
        logging.info("")

        lr_progress.append(optimizer.param_groups[0]["lr"])

    fig, ax = plt.subplots()
    ax.plot(range(1, len(list_train_loss) + 1), list_train_loss, c="#D64436", label="Training loss")
    ax.plot(range(1, len(list_train_acc) + 1), list_train_acc, c="#00B09B", label="Training accuracy")

    if val_loader:
        ax.plot(range(1, len(list_val_acc) + 1), list_val_acc, c="#E0A938", label="Validation accuracy")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True,
               fontsize="xx-small")
    ax.grid()

    fig.savefig(args.experiment_name + "_loss.png")

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
    ax1.set(xlabel="Epoch", ylabel="LR", title="")
    ax1.grid()

    fig1.savefig(args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    
if __name__ == "__main__":
    main()