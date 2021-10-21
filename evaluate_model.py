import argparse
import copy
import os

import h5py
import numpy as np
import torch

from data_loader.data_loader import TwoVMDataset, VMStiefelDataset
from model.metrics import angular_maad, maad, screw_loss
from model.model import VMSoftOrthoNet, VMStiefelNet, VMStiefelSVDNet
from utils.utils import (
    convert_labels_VMSoftOrtho,
    convert_labels_VMSt,
    convert_predictions_VMSoftOrtho,
    convert_predictions_VMSt,
    convert_predictions_VMStSVD,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")

    parser.add_argument("--model-dir", type=str, default="models/")
    parser.add_argument("--model-name", type=str, default="test_lstm")
    parser.add_argument(
        "--model-type", type=str, default="vm-st-svd", help="vm-st, vm-ortho, vm-st-svd"
    )
    parser.add_argument(
        "--test-file", type=str, default="../data/test/microwave/complete_data.hdf5"
    )
    parser.add_argument("--output-dir", type=str, default="./saved/plots/")
    parser.add_argument(
        "--ntest",
        type=int,
        default=None,
        help="number of test samples (n_object_instants)",
    )
    parser.add_argument(
        "--ndof",
        type=int,
        default=1,
        help="how many degrees of freedom in the object class?",
    )
    parser.add_argument("--batch", type=int, default=40, help="batch size")
    parser.add_argument("--nwork", type=int, default=8, help="num_workers")
    parser.add_argument("--device", type=int, default=0, help="cuda device")
    parser.add_argument("--obj", type=str, default="microwave")
    parser.add_argument(
        "--detailed", action="store_true", default=False, help="Detailed stats?"
    )
    parser.add_argument(
        "--net-size",
        nargs="+",
        type=int,
        default=None,
        help="Pass a list of int defining O/P MLP layer",
    )

    args = parser.parse_args()

    if args.ntest is None:
        with h5py.File(args.test_file, "r") as f:
            args.ntest = len(f)

    print(args)
    print("cuda?", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if args.net_size is None:
        args.net_size = [1024]

    # Dataset
    # bnds = np.load(os.path.join(os.path.dirname(args.test_file), "bounds.npy"))

    test_dataset_config = {
        "data_file": args.test_file,
        "nsample": args.ntest,
        "ndof": 1,
        "transform": None,
        # "bounds": bnds,
        "normalize": False,
    }

    if args.model_type == "vm-ortho":
        test_set = TwoVMDataset(**test_dataset_config)
    else:
        test_set = VMStiefelDataset(**test_dataset_config)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.nwork,
        pin_memory=True,
    )

    if args.model_type == "vm-ortho":
        best_model = VMSoftOrthoNet(args.net_size, img_seq_len=16)
    elif args.model_type == "vm-st":
        best_model = VMStiefelNet(args.net_size, img_seq_len=16)
    else:
        best_model = VMStiefelSVDNet(args.net_size, img_seq_len=16)

    checkpoint = torch.load(os.path.join(args.model_dir, args.model_name + ".pt"))
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.float().to(device)
    best_model.eval()

    maad_l = torch.tensor([0], dtype=float, device=device)
    maad_m_ori = torch.tensor([0], dtype=float, device=device)
    maad_m_mag = torch.tensor([0], dtype=float, device=device)
    maad_th = torch.tensor([0], dtype=float, device=device)
    maad_d = torch.tensor([0], dtype=float, device=device)

    screw_ori = torch.tensor([0], dtype=float, device=device)
    screw_dist = torch.tensor([0], dtype=float, device=device)
    screw_th = torch.tensor([0], dtype=float, device=device)
    screw_d = torch.tensor([0], dtype=float, device=device)
    screw_ortho = torch.tensor([0], dtype=float, device=device)

    beta_l = torch.tensor([0], dtype=float, device=device)  # lambda 1
    beta_m_ori = torch.tensor([0], dtype=float, device=device)  # lambda 2
    beta_m_mag = torch.tensor([0], dtype=float, device=device)
    beta_th = torch.tensor([0], dtype=float, device=device)
    beta_d = torch.tensor([0], dtype=float, device=device)

    if args.detailed:
        all_ori_err_mean = torch.empty(0)
        all_ori_err_std = torch.empty(0)
        all_dist_err_mean = torch.empty(0)
        all_dist_err_std = torch.empty(0)
        all_q_err_mean = torch.empty(0)
        all_q_err_std = torch.empty(0)
        all_d_err_mean = torch.empty(0)
        all_d_err_std = torch.empty(0)

        obj_idxs = torch.empty(0)  # Recording object indexes for analysis

        # Data collection for post-processing
        all_labels = torch.empty(0)
        all_preds = torch.empty(0)

    with torch.no_grad():
        for X in test_loader:
            depth, labels = X["depth"].to(device), X["label"].to(device)
            predictions = best_model(depth)

            if args.model_type == "vm-ortho":
                pred, cov = convert_predictions_VMSoftOrtho(predictions)
                labels = convert_labels_VMSoftOrtho(labels)

            elif args.model_type == "vm-st":
                pred, cov = convert_predictions_VMSt(predictions)
                labels = convert_labels_VMSt(labels)

            elif args.model_type == "vm-st-svd":
                pred, cov = convert_predictions_VMStSVD(predictions)
                labels = convert_labels_VMSt(labels)

            # Calculate Error statistics
            batch_size = labels.size(0)
            maad_l += angular_maad(labels[:, :, :3], pred[:, :, :3]) * batch_size
            maad_m_ori += angular_maad(labels[:, :, 3:6], pred[:, :, 3:6]) * batch_size

            maad_m_mag += (
                maad(labels[:, :, 3:6].norm(dim=-1), pred[:, :, 3:6].norm(dim=-1))
                * batch_size
            )

            maad_th += maad(labels[:, :, -2], pred[:, :, -2]) * batch_size
            maad_d += maad(labels[:, :, -1], pred[:, :, -1]) * batch_size

            # Screw Loss
            ori, dist, th, d, ortho = screw_loss(target_=labels, pred_=pred)
            screw_ori += ori * batch_size
            screw_dist += dist * batch_size
            screw_th += th * batch_size
            screw_d += d * batch_size
            screw_ortho += ortho * batch_size

            # Uncertainty
            l1, l2, b_m, b_th, b_d = cov.mean(dim=0)
            beta_l += l1 * batch_size
            beta_m_ori += l2 * batch_size
            beta_m_mag += b_m * batch_size
            beta_th += b_th * batch_size
            beta_d += b_d * batch_size

    # Report mean values
    maad_l /= test_set.length
    maad_m_ori /= test_set.length
    maad_m_mag /= test_set.length
    maad_th /= test_set.length
    maad_d /= test_set.length

    screw_ori /= test_set.length
    screw_dist /= test_set.length
    screw_th /= test_set.length
    screw_d /= test_set.length
    screw_ortho /= test_set.length

    beta_l /= test_set.length
    beta_m_ori /= test_set.length
    beta_m_mag /= test_set.length
    beta_th /= test_set.length
    beta_d /= test_set.length

    print(
        "MAAD Losses:\nl_ori: {:.4f}, m_ori: {:.4f}, m_mag: {:.4f}, theta: {:.4f}, d: {:.4f}".format(
            maad_l.item(),
            maad_m_ori.item(),
            maad_m_mag.item(),
            maad_th.item(),
            maad_d.item(),
        )
    )

    print(
        "\nScrew Losses:\nOri: {:.4f}, Dist: {:.4f}, theta: {:.4f}, d: {:.4f}, Ortho: {:.4f}\n".format(
            screw_ori.item(),
            screw_dist.item(),
            screw_th.item(),
            screw_d.item(),
            screw_ortho.item(),
        )
    )

    print(
        "\nConcentrations :\nbeta_l: {:.4f}, beta_m_ori: {:.4f}, beta_m_mag: {:.4f}, beta_theta: {:.4f}, beta_d: {:.4f}".format(
            beta_l.item(),
            beta_m_ori.item(),
            beta_m_mag.item(),
            beta_th.item(),
            beta_d.item(),
        )
    )
