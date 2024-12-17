# Electra-VNJob-NER

**Electra model for [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) task with Job recruitment information in Vietnam.**
Welcome to watch, star or fork.

<div align=center><img src="./imgs/model.png" width="500px"></div>

## VNJob dataset

### Data Formats

The VNJob dataset consists of training set `data/vnjob_train.csv` and val set `data/vnjob_val.csv`, and no test set is provided. There are 44273 training samples and 11086 test samples, and we will divide them appropriately later.

The dataset contains 9 types of entities: **Job title**, **Job type**, **Postition**, **City**, **Experience**, **Skills**, **Job fields**, **Salary**, and **Other**, the corresponding abbreviated tags are `job_title`, `job_type`, `position`, `city`, `experience`, `skills`, `job_fields`, `salary` and `O`.

The format is similar to that of the Co-NLL NER task 2002, adapted for Chinese. The data is presented in two-column format, where the first column consists of the **character** and the second is a **tag**. The tag is specified as follows:

|     Tag      | Meaning                                     |
| :----------: | ------------------------------------------- |
|      O       | Not part of a named entity                  |
| B-job_title  | Beginning character of a job title          |
| I-job_title  | Non-beginning character of a job type       |
|  B-job_type  | Beginning character of a job type           |
|  I-job_type  | Non-beginning character of a job type       |
|  B-position  | Beginning character of a job position       |
|  I-position  | Non-beginning character of a job position   |
|    B-city    | Beginning character of a city               |
|    I-city    | Non-beginning character of a city           |
| B-experience | Beginning character of a job experience     |
| I-experience | Non-beginning character of a job experience |
|   B-skills   | Beginning character of a skills of job      |
|   I-skills   | Non-beginning character of a skills of job  |
| B-job_fields | Beginning character of a job fields         |
| I-job_fields | Non-beginning character of a job fields     |
|   B-salary   | Beginning character of a job salary         |
|   I-salary   | Non-beginning character of a job salary     |

## Requirements

This repo was created on Python and PyTorch. The requirements are:

- torch==2.5.1
- numpy==1.26.4
- matplotlib==3.7.2
- pathlib==1.0.1
- transformers==4.47.0
- datasets==3.2.0
- tqdm==4.66.5
- torchmetrics==1.6.0
- pandas==2.0.3

## Results

Based on the best performance of the model on the validation set, the overall effect of the model is as follows:

|    Dataset     | Accuracy  | Recall    | Precision | F1 Score  |
| :------------: | :-------: | --------- | --------- | --------- |
|  training set  | **99.99** | **99.95** | **99.94** | **99.94** |
| validation set | **99.51** | **98.48** | **97.99** | **98.24** |

