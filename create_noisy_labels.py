import os

import dill
import h5py
import numpy as np
import torch
import argparse
from tqdm import tqdm

from model.von_mises_stiefel import VonMisesFisherStiefel
from data_loader.data_loader import ArticulationDataset


#### Create noisy labels
def generate_noisy_samples(data_file, conc, n_samples=16, batch_size=10):
    # load the dataset file and extract all Mu (or GT labels) from the dataset
    # raw_labels = []

    # labels_data = h5py.File(data_file, "r")

    # print("Loading GT lables...")
    # for obj in tqdm(labels_data.keys()):
    #     raw_labels.append(np.array(labels_data[obj][label_type])[0, :6])

    # M = torch.tensor(raw_labels).float()
    # M = M.view(-1, 2, 3).transpose(-1, -2)

    # Take input pre-specified K matrix
    D_ = conc.unsqueeze(dim=0).repeat(batch_size, 1)

    with h5py.File(data_file, "r") as f:
        data_len = len(f)

    dataset_config = {
        "data_file": data_file,
        "nsample": data_len,
    }

    data_set = ArticulationDataset(**dataset_config)
    dataset_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    print("\nGenerating samples...")
    samples = torch.empty(0)

    for X in tqdm(dataset_loader):
        M = X["label"][:, 0, :6].view(-1, 2, 3).transpose(-1, -2).to(D_.device)
        vmst = VonMisesFisherStiefel(loc=M, Diag=D_)
        samples = torch.cat((samples, vmst.sample((n_samples,))), dim=1)

    return samples.cpu()


def save_samples(savepath, samples):
    dill.dump(samples, open(savepath, "wb"))
    print("Stored noisy_lables at: {}".format(savepath))


def identity_direction_samples(conc_diag, num_samples=1000):
    D_ = conc_diag.unsqueeze(0)
    I_ = torch.eye(3, 2).unsqueeze(0).float().to(D_.device)

    vm = VonMisesFisherStiefel(loc=I_, Diag=D_)
    samples = vm.sample((num_samples,))
    return samples.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noisy label creator")
    parser.add_argument("--data-file", type=str, default="complete_data.hdf5")
    parser.add_argument(
        "--K-diag",
        "-K",
        type=float,
        nargs="+",
        help="Diagonal values of K matrix",
        required=True,
    )
    parser.add_argument("--n-samples", "-ns", type=int, default=16, help="# of samples")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", type=int, default=0, help="cuda device")

    args = parser.parse_args()

    # setup trainer
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    # conc = torch.tensor(args.K_diag).reshape(2, 2).float().to(device)
    conc_diag = torch.tensor(args.K_diag).float().to(device)

    # samples = generate_noisy_samples(
    #     args.data_file, conc_diag, args.n_samples, args.batch_size
    # )
    # fname = os.path.join(
    #     os.path.dirname(args.data_file),
    #     "noisy_labels_K_{}_{}.dill".format(args.K_diag[0], args.K_diag[-1]),
    # )

    samples = identity_direction_samples(conc_diag, args.n_samples)
    samples = samples.squeeze()  # removing redundant dims

    import pdb

    pdb.set_trace()
    fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "identity_noisy_labels_K_{}_{}_ns_{}.dill".format(
            args.K_diag[0], args.K_diag[-1], args.n_samples
        ),
    )

    save_samples(fname, samples)
