# DUST-net
Code for the **Distributional Depth-Based Estimation of Object Articulation Models** paper presented at the *5th Annual Conference on Robot Learning (CoRL)*, 2021. Full paper available [here](https://arxiv.org/abs/2108.05875). [[Project webpage]](https://pearl-utexas.github.io/DUST-net/)


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

### [Optional] Training on custom dataset
* Generate dataset using our fork of the Synthetic articulated dataset generator from [here](https://github.com/jainajinkya/SyntheticArticulatedData).
* Run the following command to train DUST-net on the generated datasets
```commandline
TO BE INCLUDED SOON
```

### Contact
In case of any questions or queries, please feel free to contact [Ajinkya Jain](mailto:ajinkya[at]utexas.edu).
