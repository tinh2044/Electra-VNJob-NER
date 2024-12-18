import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from utils import evaluate_fn, save_checkpoints, load_checkpoints, compute_metric

def train_epoch(model, dataloader, optimizer, scheduler=None, epoch=0, epochs=0):
    all_loss, all_acc, all_precision, all_recall, all_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    len_loadder = len(dataloader)
    loop = tqdm(enumerate(dataloader), total=len_loadder, leave=True, desc=f"Training epoch {epoch + 1}/{epochs}: ")
    for i, data in loop:
        labels = data["labels"]
        optimizer.zero_grad()
        outputs_model = model(**data)
        loss = outputs_model.loss
        outputs = outputs_model.logits
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

        results = compute_metric(outputs, labels)
        precision = results['precision']
        recall = results['recall']
        f1 = results['f1']
        acc = results['accuracy']

        all_acc += acc
        all_recall += recall
        all_precision += precision
        all_f1 += f1

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Recall: {recall:.3f}"
                             f", Precison: {precision:.3f}, , F1: {f1:.3f}")

    if scheduler:
        scheduler.step(all_loss / len_loadder)

    all_loss /= len_loadder
    all_acc /= len_loadder
    all_recall /= len_loadder
    all_precision /= len_loadder
    all_f1 /= len_loadder

    return all_loss, all_acc, all_recall, all_precision, all_f1

def train_fn(model, train_dataset, val_dataset, args):
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=args.batch_size, collate_fn=train_dataset.data_collator)
    val_loader = DataLoader(val_dataset, shuffle=True,
                            batch_size=args.batch_size, collate_fn=val_dataset.data_collator)

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
    
    Path(args.experiment_name).mkdir(parents=True, exist_ok=True)
    for epoch in range(resume_epoch, epochs, 1):
        train_loss, train_acc, train_recall, train_precision, train_f1 = train_epoch(model, train_loader, optimizer,
                                               scheduler, epoch=epoch, epochs=epochs)

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)

        model.train(False)
        val_loss, val_acc, val_recall, val_precision, val_f1 = evaluate_fn(model, val_loader, epoch=epoch, epochs=epochs)
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

    fig.savefig(args.experiment_name + "/loss.png")

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
    ax1.set(xlabel="Epoch", ylabel="LR", title="")
    ax1.grid()

    fig1.savefig(args.experiment_name + "/lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    
    return 