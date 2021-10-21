import argparse
import copy
import os
import random

import h5py
import numpy as np
import torch
from torchvision import transforms

from data_loader.data_loader import (
    TwoVMDataset,
    VMStiefelDataset,
    VMStiefelNoisyDataset,
)
from model.loss import VMSoftOrthoLoss, VMStiefelLoss, VMStiefelSVDLoss
from model.model import VMSoftOrthoNet, VMStiefelNet, VMStiefelSVDNet
from trainer.trainer import ModelTrainer
from noise_models import DropPixels, DropPixelsMasked

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train object learner on articulated object dataset."
    )
    parser.add_argument("--name", type=str, help="jobname", default="test")
    parser.add_argument(
        "--train-file", type=str, default="../data/test/microwave/complete_data.hdf5"
    )
    parser.add_argument(
        "--test-file", type=str, default="../data/test/microwave/complete_data.hdf5"
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=None,
        help="number of total training samples (n_object_instants)",
    )
    parser.add_argument(
        "--ntest",
        type=int,
        default=None,
        help="number of test samples (n_object_instants)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of iterations through data"
    )
    parser.add_argument("--batch", type=int, default=40, help="batch size")
    parser.add_argument("--nwork", type=int, default=8, help="num_workers")
    parser.add_argument(
        "--val-freq", type=int, default=5, help="frequency at which to validate"
    )
    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--device", type=int, default=0, help="cuda device")
    parser.add_argument(
        "--model-type", type=str, default=None, help="vm-st, vm-ortho, vm-st-svd"
    )
    parser.add_argument(
        "--load-wts",
        action="store_true",
        default=False,
        help="Should load model wts from prior run?",
    )
    parser.add_argument(
        "--wts-dir", type=str, default="models/", help="Dir of saved model wts"
    )
    parser.add_argument(
        "--prior-wts", type=str, default="test", help="Name of saved model wts"
    )
    parser.add_argument(
        "--fix-seed", action="store_true", default=False, help="Should fix seed or not"
    )
    # parser.add_argument('--lr-scheduler', default=['30', '.1'], nargs='+',
    #                     help='number of iters (arg 0) before applying gamma (arg 1) to lr')
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=1e-2,
        help="weight decay in optimizer",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=25.0, help="gradient clipping threshold"
    )
    parser.add_argument(
        "--net-size",
        nargs="+",
        type=int,
        default=None,
        help="Pass a list of int defining O/P MLP layer",
    )

    parser.add_argument(
        "--noisy-labels", action="store_true", default=False, help="Noisy labels?"
    )
    parser.add_argument(
        "--noisy-labels-file",
        type=str,
        help="path of noisy label file",
        required=None,
    )

    parser.add_argument(
        "--training-stage",
        "-stg",
        type=int,
        default=1,
        help="Training stage# 1, 2, or 3",
        required=True,
    )

    args = parser.parse_args()

    if args.ntrain is None:
        with h5py.File(args.train_file, "r") as f:
            args.ntrain = len(f)

    if args.ntest is None:
        with h5py.File(args.test_file, "r") as f:
            args.ntest = len(f)

    print(args)
    print("cuda?", torch.cuda.is_available())

    if args.fix_seed:
        SEED = 1237
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    if args.net_size is None:
        args.net_size = [1024]

    # noiser = DropPixels(p=0.1)
    # noiser = DropPixelsMasked(p=0.1)

    if args.noisy_labels and args.noisy_labels_file is None:
        raise ValueError("Please pass noisy lables file as well")

    # setup trainer
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    # Dataset
    train_dataset_config = {
        "data_file": args.train_file,
        "nsample": args.ntrain,
        "ndof": 1,
        "transform": None,
        "bounds": None,
        "normalize": False,
        "noisy_labels_file": args.noisy_labels_file,
        "m_mag_beta": 50,
        "config_beta": 50,
    }

    if args.model_type == "vm-ortho":
        train_set = TwoVMDataset(**train_dataset_config)
    else:
        if args.noisy_labels:
            train_set = VMStiefelNoisyDataset(**train_dataset_config)
        else:
            train_set = VMStiefelDataset(**train_dataset_config)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.nwork,
        pin_memory=True,
    )

    val_dataset_config = copy.copy(train_dataset_config)
    val_dataset_config["data_file"] = args.test_file
    val_dataset_config["nsample"] = args.ntest
    val_dataset_config["bounds"] = train_set.bounds

    if args.model_type == "vm-ortho":
        validation_set = TwoVMDataset(**val_dataset_config)
    else:
        validation_set = VMStiefelDataset(**val_dataset_config)

    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.nwork,
        pin_memory=True,
    )

    # Learning setup
    if args.model_type == "vm-ortho":
        loss_fn = VMSoftOrthoLoss
        network = VMSoftOrthoNet(args.net_size, img_seq_len=16)
    elif args.model_type == "vm-st":
        loss_fn = VMStiefelLoss
        network = VMStiefelNet(args.net_size, img_seq_len=16)
    else:
        # DEFAULT: "vm-st-svd"
        loss_fn = VMStiefelSVDLoss
        network = VMStiefelSVDNet(args.net_size, img_seq_len=16)

    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # optimizer = torch.optim.Adam(network.parameters(),
    #                             lr=args.learning_rate,
    #                             weight_decay=args.weight_decay)

    # scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_schedule, gamma=lr_gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Load Saved wts
    if args.load_wts:
        checkpoint = torch.load(
            os.path.join(args.wts_dir, args.prior_wts + ".pt"), map_location=device
        )
        network.load_state_dict(checkpoint["model_state_dict"])

        """
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        """

    ## Debug
    torch.autograd.set_detect_anomaly(True)

    ## Model Trainer
    trainer_config = {
        "model": network,
        "train_loader": train_loader,
        "test_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": loss_fn,
        "epochs": args.epochs,
        "name": args.name,
        "test_freq": args.val_freq,
        "device": device,
        "grad_th": args.grad_clip,
        "logs_dir": "runs/",
        "plots_dir": "plots/",
        "ndof": 1,
        "model_type": args.model_type,
        "training_stage": args.training_stage,
    }

    trainer = ModelTrainer(**trainer_config)

    # train
    best_model = trainer.train()

    # #Test best model
    # trainer.test_best_model(best_model, fname_suffix='_posttraining')
