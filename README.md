# MultiModel_OneGPU
A minimal working example for serving multiple models on one GPU when the activations of each model is large but the model size is small. 

There are two tricks:
1. For each model, reduce the batchsize so that the activations of each model is reduced. 
2. Use `torch.cuda.empty_cache()` to clear the cache after each request in the postprocessing step. 

In this example, you will learn from scratch how to deploy models with TorchServe. 

## Setup

### Conda Create/Activate

```
conda create -n multi_model_one_gpu python=3.9
conda activate multi_model_one_gpu
```

### Install TorchServe and Torch-Model-Archiver

```
pip install torch torchserve torch-model-archiver
```

### Run and Save Model (.pt)

```
python main.py
```

### Create Model Archive (.mar)

```
torch-model-archiver --model-name cross_encoder_model \
    --version 1.0 \
    --serialized-file cross_encoder_model/cross_encoder_model.pt \
    --handler handler.py \
    --export-path model_store \
    --extra-files "handler.py,main.py"\
    --force
```


## Command

### Start TorchServe
This will start one Bert model on one GPU, each inference with the batchsize of 400
```
./cross_encoder_run_script
```

This will start two Bert models on one GPU, each inference with the batchsize of 400
```
./cross_encoder_two_models_run_script
```

### Test
This will test one model every second
```
python request_model1.py 
```

This will test two models simultaneously on the GPU
```
python request_two_models.py
```

### Monitor the GPU Usage
```
nvidia-smi
```