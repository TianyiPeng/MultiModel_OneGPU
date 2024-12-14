# MultiModel_OneGPU
A minimal working example for serving multiple models on one GPU

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
This will start two models on one GPU
```
./cross_encoder_run_script
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
