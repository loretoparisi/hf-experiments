# ImageNet Samples
Sample images from ImageNet. Sampled from https://github.com/ajschumacher/imagen

### How to load
To load these sample images into `torch` use the provided `LPCustomDataSet` class:

```python
batch_size = 2
my_dataset = LPCustomDataSet(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data'), transform=LPCustomDataSet.transform)
train_loader = torch.utils.data.DataLoader(my_dataset , batch_size=batch_size, shuffle=False, 
                               num_workers=4, drop_last=True, collate_fn=LPCustomDataSet.collate_fn)
for idx, img in enumerate(train_loader):
    print(idx, img.shape)
```