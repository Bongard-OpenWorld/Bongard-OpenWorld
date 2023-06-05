Bongard-OpenWorld
===

## Installation

This codebase can be built from scratch on Ubuntu 20.04 with Python 3.10, PyTorch 1.13 and CUDA 11.7.

```bash
conda create -n bongard-ow python=3.10
conda activate bongard-ow
conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

##  Data Preparation

To download all images from the URLs, navigate to the root directory of Bongard-OpenWorld and run `scripts/crawl_images.py`.

```bash
cd Bongard-OpenWorld
python scripts/crawl_images.py
```

Please note that some links may be invalid due to the instability of the URLs. To ensure that the community can reproduce our results from scratch, we have provided a backup of all the images. You can download from [Google Drive](https://drive.google.com/file/d/1aXr3ihVq0mtzbl6ZNJMogYEyEY-WALNr/view?usp=sharing).


The images should be extracted to `assets/data/bongard-ow/images` and the file structure looks like:
```plain
assets
├── data
│   └── bongard-ow
│       ├── images
│       │   ├── 0000
│       │   ├── 0001
│       │   ├── ....
│       │   └── 1009
│       ├── bbox_data.pkl
│       ├── bongard_ow.json
│       ├── bongard_ow_train.json
│       ├── bongard_ow_val.json
│       └── bongard_ow_test.json
└── weights
```

Please note that this repository only hosts the code for Bongard-OpenWorld. All images of Bongard-OpenWorld are crawled from [Google Images](https://images.google.com) and should not be considered part of the source code.

We do not claim ownership of any image in Bongard-OpenWorld. Therefore, we strongly recommend that you delete all images immediately after benchmarking all approaches and evaluations.

## Traning Few-Shot Models

```bash
bash fewshot.sh 
```

## Inference Zero-Shot Models
```bash
bash zeroshot.sh 
```