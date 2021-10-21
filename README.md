# Distributional Depth-Based Estimation of Object Articulation Models

## Instructions to run the code

### Installing prerequisites and environment
```commandline
conda env create -f env.yaml
conda activate dustnet
cd /path/to/the/repository/
```

### Downloading datasets and pretrained model weights
* Evaluation datasets: [Link](https://drive.google.com/file/d/1RBob0J46DW3O-S58zmnOH-b8arKUQljA/view?usp=sharing)
* Pretrained weights: [Link](https://drive.google.com/file/d/1Ig8YntLNJMn1ysHWjro576bsccctiMp2/view?usp=sharing)

### Running the evaluation script
```commandline
python evaluate_model.py --model-dir <pretrained-model-dir> --model-name <model-name> --test-file <test-dir-name> --model-type <vm-ortho, vm-st, vm-st-svd> --output-dir <output-dir>
```
