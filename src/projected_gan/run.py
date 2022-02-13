# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2022 Loreto Parisi (loretoparisi at gmail dot com)

import os, sys
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
# LP: to use root src folder
sys.path.insert(0, os.path.join(BASE_PATH, './projected_gan'))
sys.path.insert(0, os.path.join('..', 'lib'))

from PIL import Image
from projected_gan.gen_images import generate_images

from util import show_images

cache_dir=os.getenv("cache_dir", "../../models")
def inference(model_name, seeds):
    network_pkl = "file://"+os.path.join(cache_dir,model_name)
    outdir=os.path.join("..","output")
    generate_images(network_pkl=network_pkl,
        seeds=seeds,
        outdir= outdir,
        truncation_psi=1,
        noise_mode='const',
        translate='0,0',
        rotate=0,
        class_idx=None
    )
    images = [ Image.open(os.path.join(outdir,f"seed{seed:04d}.png")) for seed in seeds]
    return images

# pretrained models: pokemon | art_painting | church | cityscapes | clevr | ffhq | flowers | landscape
images = inference("pokemon.pkl",[4,5])
# ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
#show_images(images)