import os
import copy
import json
import torch
import numpy as np
from PIL import Image
from lavis.models import model_zoo, load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="caption_coco_opt6.7b",
    is_eval=True,
    device=device)

image_path = 'assets/data/bongard-ow/bongard_ow_test.json'
caption_path = 'blip2.json'

def main():
    captions = []

    with open(image_path, 'r') as f:
        bongard_ow_test = json.load(f)
        for sample in bongard_ow_test:
            uid = sample['uid']
            imageFiles = [os.path.join('assets/data/bongard-ow', imageFile) for imageFile in sample['imageFiles']]

            # preprocess the image
            # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
            image = [
                vis_processors["eval"](Image.open(imageFile).convert("RGB")).numpy()
                for imageFile in imageFiles
            ]
            image = torch.from_numpy(np.array(image)).to(device)

            # generate caption
            sample['captions'] = model.generate({"image": image})
            captions.append(copy.deepcopy(sample))

        with open(caption_path, "w") as file:
            json.dump(captions, file, indent=4)

if __name__ == '__main__':
    main()