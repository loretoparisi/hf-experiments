# mlpvision
MLP based Vision models examples: ResMLP, MLP-Mixer

## How to run
To load images embedding for provided models:
```
python src/mlpvision/run.py 
ResMLP trainable Parameters: 26.605M
ResMLP: torch.Size([1, 1000])
MLPMixer trainable Parameters: 18.528M
MLPMixer: torch.Size([1, 1000])
Perceiver trainable Parameters: 70.396M
Perceiver: torch.Size([1, 1000])
0 torch.Size([1, 3, 224, 224])
ResMLP pred: torch.Size([1, 1000])
MLPMixer pred: torch.Size([1, 1000])
torch.Size([1, 3, 224, 224])
torch.Size([1, 224, 224, 3])
Perceiver pred: torch.Size([1, 1000])
1 torch.Size([1, 3, 224, 224])
ResMLP pred: torch.Size([1, 1000])
MLPMixer pred: torch.Size([1, 1000])
torch.Size([1, 3, 224, 224])
torch.Size([1, 224, 224, 3])
Perceiver pred: torch.Size([1, 1000])
```

To train a model
```
python src/mlpvision/train.py 
```

## Models
### resmlp
Code: [res-mlp-pytorch](https://github.com/lucidrains/res-mlp-pytorch)
Paper: [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404)

### mlpmixer
Paper: [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
Code: [MLP-Mixer-pytorch](https://github.com/rishikksh20/MLP-Mixer-pytorch)

### Perceiver
Paper: [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
Code: [perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)

## Note
Currently not avaiable on HF models' hub.