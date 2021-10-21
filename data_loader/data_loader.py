import copy
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import dill

from utils.utils import convert_to_spherical_labels


class ArticulationDataset(Dataset):
    def __init__(self, **kwargs):
        super(ArticulationDataset, self).__init__()

        self.data_file = os.path.abspath(kwargs["data_file"])
        self.labels_data = None
        self.labels = self.get_raw_labels()
        self.length = kwargs.get("nsample")
        self.ndof = kwargs["ndof"] if "ndof" in kwargs else 1
        self.transform = kwargs.get("transform")
        self.threshold_max_depth = 4.0  # Max depth of kinect is 3.5 m
        self.bounds = kwargs.get("bounds")

        self.root_dir = os.path.dirname(self.data_file)

        normalize = kwargs["normalize"] if "normalize" in kwargs else False
        if normalize:
            if self.bounds is None:
                self.bounds = self.compute_bounds(self.labels)
                bnds_filename = os.path.join(self.root_dir, "bounds.npy")
                np.save(bnds_filename, self.bounds)
                print("Saved label bounds file at:{}".format(bnds_filename))

            self.labels = self.normalize_labels(self.labels, self.bounds)
            print("Labels normalized!")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.labels_data is None:
            self.labels_data = h5py.File(self.data_file, "r")

        # One Sample for us corresponds to one instantiation of an object type
        obj_data = self.labels_data["obj_" + str(idx).zfill(6)]

        # Load depth image
        depth_imgs = torch.tensor(obj_data["depth_imgs"])

        # Correcting depth images so that they correspond to a threshold depth
        depth_imgs *= 12.0
        depth_imgs[depth_imgs > self.threshold_max_depth] = self.threshold_max_depth
        depth_imgs /= self.threshold_max_depth

        # transform depth images
        obj_pose_in_world = np.array(obj_data["embedding_and_params"])[-7:]

        # if self.transform is not None:
        #     depth_imgs = self.transform(
        #         (depth_imgs, torch.tensor(obj_pose_in_world[:3]))
        #     )
        depth_imgs = self.transform(depth_imgs)

        # # Other channels
        # rel_normed_depth_imgs = torch.tensor(obj_data["rel_normalized_depth_imgs"])
        # image_gradients_xy = torch.tensor(obj_data["XY_gradients"])
        # image_gradients_flow = torch.tensor(obj_data["optical_flow_imgs"])

        channel_1 = depth_imgs
        channel_2 = depth_imgs  # rel_normed_depth_imgs
        channel_3 = depth_imgs  # image_gradients_xy
        # channel_4 = image_gradients_flow
        imgs = torch.cat(
            (
                channel_1.unsqueeze(1),
                channel_2.unsqueeze(1),
                channel_3.unsqueeze(1),
                # channel_4.unsqueeze_(1),
            ),
            dim=1,
        ).float()

        # Load labels
        # label = torch.from_numpy(np.array(obj_data['local_labels'])).float()
        # label = torch.from_numpy(np.array(obj_data['global_labels'])).float()
        label = self.labels[idx]

        sample = {"depth": imgs, "label": label}

        return sample

    def get_raw_labels(self, label_type="global_labels"):
        raw_labels = []

        if self.labels_data is None:
            self.labels_data = h5py.File(self.data_file, "r")

        for obj in self.labels_data.keys():
            raw_labels.append(np.array(self.labels_data[obj][label_type]))

        return torch.Tensor(raw_labels).float()

    def compute_bounds(self, labels):
        bounds = np.zeros((labels.shape[2], 2))
        for d in range(labels.shape[2]):
            dmin = labels[:, :, d].min()
            dmax = labels[:, :, d].max()
            bounds[d, 0] = dmin
            bounds[d, 1] = dmax
        return bounds

    def normalize_dim(self, data, bounds, dimension):
        dmax, dmin = bounds[dimension, 1], bounds[dimension, 0]
        data[:, :, dimension] = (data[:, :, dimension] - dmin) / (dmax - dmin)

    def normalize_labels(self, labels, bounds):
        for d in range(labels.shape[2]):
            dmin = bounds[d, 0]
            dmax = bounds[d, 1]
            if (dmin == 0 and dmax == 0) or (dmin == 1 and dmax == 1) or (dmin == dmax):
                pass
            else:
                self.normalize_dim(labels, bounds, d)
        return labels


class TwoVMDataset(ArticulationDataset):
    def __init__(self, **kwargs):
        normalize = kwargs["normalize"] if "normalize" in kwargs else False
        kwargs["normalize"] = False  # Using different normalization than super

        super(TwoVMDataset, self).__init__(**kwargs)

        self.new_labels = convert_to_spherical_labels(self.labels)

        if normalize:
            if self.bounds is None:
                self.bounds = self.compute_bounds(self.new_labels)
                bnds_filename = os.path.join(self.root_dir, "bounds.npy")
                np.save(bnds_filename, self.bounds)
                print("Saved label bounds file at:{}".format(bnds_filename))

            self.new_labels = self.normalize_labels(self.new_labels, self.bounds)
            print("Labels normalized!")

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["label"] = self.new_labels[idx]
        return sample


class VMStiefelDataset(ArticulationDataset):
    def __init__(self, **kwargs):
        super(VMStiefelDataset, self).__init__(**kwargs)
        self.new_labels = self.convert_labels(self.labels)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["label"] = self.new_labels[idx]
        return sample

    def convert_labels(self, labels):
        l, m, th, d = (
            labels[:, :, :3],
            labels[:, :, 3:6],
            labels[:, :, 6:7],
            labels[:, :, 7:],
        )
        m_norm = m.norm(dim=-1, keepdim=True)
        m = m / m_norm
        return torch.cat((l, m, m_norm, th, d), dim=-1)


class VMStiefelNoisyDataset(VMStiefelDataset):
    def __init__(self, **kwargs):
        super(VMStiefelNoisyDataset, self).__init__(**kwargs)

        screw_noise_samples_filepath = kwargs.get("noisy_labels_file")
        self.screw_noise_samples = dill.load(
            open(screw_noise_samples_filepath, "rb")
        ).float()

        # construct rotation matrices
        M = self.new_labels[:, 0, :6].view(-1, 2, 3).transpose(-1, -2)
        self.rot_mat = torch.cat(
            (M, torch.cross(M[:, :, 0], M[:, :, 1]).unsqueeze(-1)), dim=-1
        )

        # config noise
        self.m_mag_beta = kwargs.get("m_mag_beta")
        self.config_beta = kwargs.get("config_beta")

        label_seq_len = 15
        self.m_mag_noise = torch.normal(
            torch.zeros(label_seq_len, 1), std=(1 / self.m_mag_beta)
        )
        self.config_noise = torch.normal(
            torch.zeros(label_seq_len, 1), std=(1 / self.config_beta)
        )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        idxs = np.random.choice(
            self.screw_noise_samples.size(0), size=15, replace=False
        )
        noise_samples = self.screw_noise_samples[idxs, :, :]
        X = torch.matmul(self.rot_mat[idx], noise_samples)

        # X = self.noisy_labels_screw[:, idx, :, :]
        sample["label"][:, 0:3] = X[:, :, 0]
        sample["label"][:, 3:6] = X[:, :, 1]

        # add noisy version of configs
        sample["label"][:, -3:-2] += self.m_mag_noise
        sample["label"][:, -2:-1] += self.config_noise
        sample["label"][:, -1:] += self.config_noise

        return sample
