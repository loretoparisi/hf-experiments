# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021-2022 Loreto Parisi (loretoparisi at gmail dot com)

from rudalle.pipelines import generate_images, show
from rudalle import get_rudalle_model, get_tokenizer, get_vae
from rudalle.utils import seed_everything
from huggingface_hub import hf_hub_url, cached_download
import os
import torch

has_gpu = torch.cuda.is_available()
fp16 = True if has_gpu else False

# load models
model_filename = "pytorch_model.bin"
device = "cuda" if has_gpu else "cpu"
cache_dir=os.getenv("cache_dir", "../../models")
config_file_url = hf_hub_url(repo_id="minimaxir/ai-generated-pokemon-rudalle", filename=model_filename)
cached_download(config_file_url, cache_dir=cache_dir, force_filename=model_filename)
model = get_rudalle_model('Malevich', cache_dir=cache_dir, pretrained=False, fp16=fp16, device=device)
model.load_state_dict(torch.load(os.path.join(cache_dir,model_filename), map_location='cpu')) 
vae = get_vae().to(device)
tokenizer = get_tokenizer()

# generate
images_per_row = 4
num_rows = 1

# In theory you could get more specific Pokemon by
# specifying the type (see the repo linked in the intro),
# but it practice it does not influence generation much.
text = "Покемон"

gen_configs = [
        (2048, 0.995),
        (1536, 0.99),
        (1024, 0.99),
        (1024, 0.98),
        (512, 0.97),
        (384, 0.96),
        (256, 0.95),
        (128, 0.95), 
    ]

gen_configs = gen_configs[0:num_rows]

pil_images = []
scores = []

img_count = 0
for top_k, top_p in gen_configs:
      _pil_images, _scores = generate_images(text, tokenizer, model, vae, top_k=top_k, images_num=images_per_row, top_p=top_p)
      for images in _pil_images:
            images.save(f"./{img_count:03d}.png")
            img_count += 1
      show(_pil_images, images_per_row, size=28)
      pil_images += _pil_images
show([pil_image for pil_image in pil_images], images_per_row, size=56)