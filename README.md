# Objectives

- Given data already exists in harddrive, train a model
- Keeping track of many models/ versions of model 
- Deploy it to be accessed, request publicly

# Core Objectives

## What model to train?

- Some transformer to convert pair of texts to pair of vectors, then some classification (SVM) to classify
- Prune, distill, modify a pretrained transformer

## What to keep track?

- Data
- Train loss, Validation loss (Cross Entropy Error)
- Models
- Checkpoints
- Hyperparameters
- System metric loggings (CPU, GPU, I/O usages)
- Model checkpoints

## What to optimize?

- Hyperparameters 
- Quantization
- Pruning
- Mixed precision training

## What to deploy?

- Deploy by mlflow on live sever (by ngrok) running on kaggle or collab
- Deploy by Triton?

## My computer is a potato

- Two seperate packages with shared components: local and cloud; local use cpu and lightweight, cloud use gpu ones.
- Seperate by system platform marker; cloud is `linux`, local is `win32`. You can check by this:

```python
import platform
platform.system()
```

**NOTE**: Because of this, any linux system will automatically install gpu versions! Check torch index in `pyproject.toml` before running `uv sync`

# Technicals

- PyLighting, Pytorch for models and training
- TensorRT, CUDnn for model optimizing
- Huggingface for pretrained transformers
- MLFlow for model versioning
- ipynb for runtime running
- PostgreSQL for metedata database
- Ngrok for open tunnel
- Collab and Kaggle for deployment

# Models



# File Structures

## .key

Stores private information such as token, dataset links

## /runtime

Ipynb notebooks that is run on cloud

## /scripts 

Scripts for faster setup in new envinronment, usually do:

- Pull codes from github publicly
- Update postgre database
- Save model registry file for permanent usage

**Note**: Dataset downloaded by hand

## /src

Main for model, loss function, dataset codes

## /src/dataloader

Dataloader for model

## /src/models

Models

## /src/lossfn

Custom loss functions as classes

## /src/trainer

Maybe in runtime?