import os

import torch
from tqdm import tqdm
from torchmetrics.functional import accuracy, precision, recall, f1_score



ner_class_names = [ "O", "B-job_title", "I-job_title", "B-job_type", 
                   "I-job_type", "B-position_level", "I-position_level", 
                   "B-city", "I-city", "B-experience", "I-experience", 
                   "B-skills", "I-skills", "B-job_fields", "I-job_fields", "B-salary", "I-salary"]

def compute_metric(logits, labels):
    predictions = torch.argmax(logits, axis=-1)
    # true_labels = [[ner_class_names[l] for l in label if l!=-100] for label in labels]
    # true_predictions = [[ner_class_names[p] for p,l in zip(prediction, label) if l!=-100]
    #                   for prediction, label in zip(predictions, labels)]

    return {"precision": precision(predictions, labels, task="multiclass", num_classes=len(ner_class_names), average="macro"),
          "recall": recall(predictions, labels, task="multiclass", num_classes=len(ner_class_names), average="macro"),
          "f1": f1_score(predictions, labels, task="multiclass", num_classes=len(ner_class_names), average="macro"),
          "accuracy": accuracy(predictions, labels, task="multiclass", num_classes=len(ner_class_names))}

def save_checkpoints(model, optimizer, path_dir, epoch, name=None):
    if not os.path.exists(path_dir):
        print(f"Making directory {path_dir}")
        os.makedirs(path_dir)
    if name is None:
        filename = f'{path_dir}/checkpoints_{epoch}.pth'
    else:
        filename = f'{path_dir}/checkpoints_{epoch}_{name}.pth'
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, filename)


def load_checkpoints(model, optimizer, path, resume=True):
    if not os.path.exists(path):
        raise FileNotFoundError
    if os.path.isdir(path):
        epoch = max([int(x[x.index("_") + 1:len(x) - 4]) for x in os.listdir(path)])
        filename = f'{path}/checkpoints_{epoch}.pth'
        print(f'Loaded latest checkpoint: {epoch}')

        checkpoints = torch.load(filename)

    else:
        print(f"Load checkpoint from file : {path}")
        checkpoints = torch.load(path)

    model.load_state_dict(checkpoints['model'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    if resume:
        return checkpoints['epoch'] + 1
    else:
        return 1

def evaluate_fn(model, dataloader, epoch=0, epochs=0):
    all_loss, all_acc, all_precision, all_recall, all_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    len_loadder = len(dataloader)
    loop = tqdm(enumerate(dataloader), total=len_loadder, leave=True,
                desc=f"Evaluation epoch {epoch + 1}/{epochs}: ")

    for i, data in loop:
        labels = data["labels"]
        outputs_model = model(**data)
        loss = outputs_model.loss
        logits = outputs_model.logits

        results = compute_metric(logits, labels)
        precision = results['precision']
        recall = results['recall']
        f1 = results['f1']
        acc = results['accuracy']

        all_acc += acc
        all_recall += recall
        all_precision += precision
        all_f1 += f1

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Recall: {recall:.3f}, Precison: {precision:.3f}, , F1: {f1:.3f}")

    all_loss /= len_loadder
    all_acc /= len_loadder
    all_recall /= len_loadder
    all_precision /= len_loadder
    all_f1 /= len_loadder

    return all_loss, all_acc, all_recall, all_precision, all_f1

