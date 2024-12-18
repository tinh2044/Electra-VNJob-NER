# Electra-VNJob-NER

**Electra model for [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) task with job recruitment information in Vietnam.**  
Welcome to watch, star, or fork.

<div align="center">
    <img src="./imgs/model.png" width="500px" alt="Model Diagram">
</div>

---

## Table of Contents

- [Introduction](#introduction)
- [VNJob Dataset](#vnjob-dataset)
  - [Data Formats](#data-formats)
  - [Example Data](#example-data)
- [Requirements](#requirements)
- [Results](#results)
- [Usage](#usage)
  - [Training Electra Model for NER Task](#training-electra-model-for-ner-task)
  - [Evaluate Electra Model](#evaluate-electra-model)
  - [Interface Model with Gradio](#interface-model-with-gradio)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Introduction

This repository contains an implementation of the Electra model for Named Entity Recognition (NER) tailored to job recruitment data in Vietnam. NER is a crucial task in natural language processing, and this model is designed to identify specific entities such as **job titles**, **skills**, and **salary ranges** from Vietnamese job postings.

Electra, a transformer-based model, was chosen for its efficiency and effectiveness in pretraining tasks, making it suitable for this domain-specific application.

---

## VNJob Dataset

### Data Formats

The VNJob dataset consists of:

- **Training set**: `data/vnjob_train.csv`
- **Validation set**: `data/vnjob_val.csv`

There are **44,273 training samples** and **11,086 validation samples**. No separate test set is provided. Users may split the validation set for testing as needed.

The dataset contains 9 types of entities: **Job title**, **Job type**, **Position**, **City**, **Experience**, **Skills**, **Job fields**, **Salary**, and **Other**. The corresponding abbreviated tags are `job_title`, `job_type`, `position`, `city`, `experience`, `skills`, `job_fields`, `salary`, and `O`.

Data format:  
The data follows a two-column structure similar to the CoNLL NER 2002 format:

- The **first column** contains the text (character level).
- The **second column** contains the tag.

| Tag          | Description                             |
| ------------ | --------------------------------------- |
| O            | Not part of a named entity              |
| B-job_title  | Beginning of a job title                |
| I-job_title  | Inside a job title                      |
| B-job_type   | Beginning of a job type                 |
| I-job_type   | Inside a job type                       |
| B-position   | Beginning of a job position             |
| I-position   | Inside a job position                   |
| B-city       | Beginning of a city name                |
| I-city       | Inside a city name                      |
| B-experience | Beginning of job experience description |
| I-experience | Inside a job experience description     |
| B-skills     | Beginning of a job skill                |
| I-skills     | Inside a job skill                      |
| B-job_fields | Beginning of job fields                 |
| I-job_fields | Inside job fields                       |
| B-salary     | Beginning of a salary description       |
| I-salary     | Inside a salary description             |

### Example Data

| Character | Tag         |
| --------- | ----------- |
| Software  | B-job_title |
| Engineer  | I-job_title |
| at        | O           |
| Hanoi     | B-city      |

---

## Requirements

This project was developed in Python with PyTorch. Below are the dependencies:

- torch==2.5.1
- numpy==1.26.4
- matplotlib==3.7.2
- pathlib==1.0.1
- transformers==4.47.0
- datasets==3.2.0
- tqdm==4.66.5
- torchmetrics==1.6.0
- pandas==2.0.3

---

## Results

The model's performance on the VNJob validation set is as follows:

| Dataset        | Accuracy  | Recall    | Precision | F1 Score  |
| -------------- | --------- | --------- | --------- | --------- |
| Training set   | **99.99** | **99.95** | **99.94** | **99.94** |
| Validation set | **99.51** | **98.48** | **97.99** | **98.24** |

---

## Usage

### Clone the Repository

```bash
git clone https://github.com/tinh2044/Electra-VNJob-NER.git
cd ./Electra-VNJob-NER
```

### Create Virtual Environment

Use [conda](https://conda.io/projects/conda/en/latest/index.html) to avoid conflicts.

```bash
conda create --name ElectraNER
conda activate ElectraNER
```

### Install Requirements

```bash
pip install -r ./requirements.txt
```

### Download Dataset

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1ze6mcyi2UtaXNDfPuPo-EgmPdvsCYGe3?usp=drive_link).  
Ensure the `data` folder has the following structure after extraction:

```
|——data
    |——vnjob_train.csv
    |——vnjob_val.csv
```

### Training Electra Model for NER Task

Run the following command to train the model:

```bash
python -m train --mode train --epoch 200 --lr 0.001 --batch_size 32 \
        --repo_id tinh2312/Electra-VNJob-NER
```

### Evaluate Electra Model

```bash
python -m train --mode eval --batch_size 32 \
        --repo_id tinh2312/Electra-VNJob-NER
```

### Interface Model with Gradio

Use Gradio to launch a demo interface for the model:

```bash
gradio app.py --demo-name=demoe
```

---

## Contributing

Contributions are welcome!  
To contribute:

1. Fork this repository.
2. Create a new branch for your changes.
3. Submit a pull request with a clear description of your changes.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

- phongtranWW/crawler [[GitHub]](https://github.com/phongtranWW/crawler)
- CLARK, K. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020. [[Paper]](https://arxiv.org/abs/2003.10555)
- google-research/electra [[GitHub]](https://github.com/google-research/electra)
- huggingface/electra_model [[GitHub]](https://github.com/huggingface/transformers/blob/main/src/transformers/models/electra/modeling_electra.py)
- chakki-works/seqeval [[GitHub]](https://github.com/chakki-works/seqeval)
