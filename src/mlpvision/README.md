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
[1,  2000] loss: 2.305
[1,  4000] loss: 2.305
[1,  6000] loss: 2.306
[1,  8000] loss: 2.306
[1, 10000] loss: 2.307
[1, 12000] loss: 2.305
[2,  2000] loss: 2.305
[2,  4000] loss: 2.307
[2,  6000] loss: 2.306
[2,  8000] loss: 2.303
[2, 10000] loss: 2.307
[2, 12000] loss: 2.305
model saved as CNNish_cifar.pt
model loaded from CNNish_cifar.pt
Accuracy for class plane is: 47.5 %
Accuracy for class car   is: 0.0 %
Accuracy for class bird  is: 0.0 %
Accuracy for class cat   is: 0.0 %
Accuracy for class deer  is: 0.0 %
Accuracy for class dog   is: 9.8 %
Accuracy for class frog  is: 0.0 %
Accuracy for class horse is: 24.9 %
Accuracy for class ship  is: 0.0 %
Accuracy for class truck is: 0.0 %
GroundTruth:    cat  ship  ship plane
model loaded from CNNish_cifar.pt
Input: torch.Size([4, 3, 32, 32])
model out: torch.Size([4, 10])
Predicted:    dog plane plane   dog
```

## Models
### ResMLP
Code: [res-mlp-pytorch](https://github.com/lucidrains/res-mlp-pytorch)
Paper: [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404)

### MLP-Mixer
Paper: [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
Code: [MLP-Mixer-pytorch](https://github.com/rishikksh20/MLP-Mixer-pytorch)

### Perceiver
Paper: [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
Code: [perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)

## Note
Currently not avaiable on HF models' hub.