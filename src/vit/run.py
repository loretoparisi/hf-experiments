# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os,sys
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, '..'))
from lpdutils.lpimagedataset import LPImageDataSet

# to choose a different model by image size, patch size, and parameters number, see README
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224',
    cache_dir=os.getenv("cache_dir", "../../models"))
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224',
    cache_dir=os.getenv("cache_dir", "../../models"))

# load local dataset
batch_size = 2
num_workers = 2
my_dataset = LPImageDataSet(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data', 'imagenet'), transform=LPImageDataSet.transform)
imageloader = torch.utils.data.DataLoader(my_dataset, 
                                batch_size=batch_size,
                                shuffle=True, 
                                num_workers=num_workers, 
                                drop_last=True,
                                collate_fn=LPImageDataSet.collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    dataiter = iter(imageloader)
    images = dataiter.next()
    for idx, img in enumerate(images):
        img = img.to(device)
        inputs = feature_extractor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    img = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
