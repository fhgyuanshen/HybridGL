# Hybrid Global-Local Representation with Augmented Spatial Guidance for Zero-Shot Referring Image Segmentation
<img src="assets/framework.png" width="100%">

# Visual Results
<img src="assets/visual.png" width="100%">

# Environment Setup
```
# Create&Activate conda env
conda create -n hybridgl python=3.10  
conda activate hybridgl

# Install Pytorch 2.0.1+cu117 version
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
# Install spacy for language processing
conda install -c conda-forge spacy=3.7.6 einops=0.8.0
pip install pydantic==2.9.1
python -m spacy download en_core_web_lg

# Install required packages
pip install opencv-python==4.10.0.84 matplotlib==3.9.2 markupsafe==2.1.5 h5py scikit-image==0.24.0 pycocotools==2.0.8
# Install GEM, make sure the open_clip version is 2.24.0
pip install gem_torch
pip install open_clip_torch==2.24.0

cd third_parth
cd modified_CLIP
pip install -e .

cd ..
cd segment-anything
pip install -e .
cd ..
mkdir checkpoints 
cd checkpoints 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

# Dataset
we follow [dataset setup](https://github.com/Seonghoon-Yu/Zero-shot-RIS/tree/master/refer) in [Zero-shot-RIS](https://github.com/Seonghoon-Yu/Zero-shot-RIS)
## 1. Download COCO 2014 train images
In "./refer/data/images/mscoco/images" path
```shell
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014
```

## 2. Download RefCOCO, RefCOCO+, and RefCOCOg annotations 
In "./refer/data" path
```shell
# RefCOCO
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
unzip refcoco.zip

# RefCOCO+
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
unzip refcoco+.zip

# RefCOCOg
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
unzip refcocog.zip
```

# Run demo
```
CUDA_VISIBLE_DEVICES=0 /path2env/bin/python Hybridgl_main.py --dataset refcoco(+/g) --split testA/testB/test/val --fusion_mode G2L/L2G/G2L&L2G
CUDA_VISIBLE_DEVICES=0 /path2env/bin/python Hybridgl_main.py --dataset refcocog --split val --fusion_mode G2L
...
```

## Acknowledgement
Our code is based on the following open-source projects: [CLIP](https://github.com/openai/CLIP), [Zero-shot-RIS](https://github.com/Seonghoon-Yu/Zero-shot-RIS), [GEM](https://github.com/WalBouss/GEM). we sincerely thanks to the developers of these resources!

# Citation
```
bibtex
@inproceedings{liu2025hybrid,
  title={Hybrid Global-Local Representation with Augmented Spatial Guidance for Zero-Shot Referring Image Segmentation},
  author={Liu, Ting and Li, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  pages={1--10}
}
```
