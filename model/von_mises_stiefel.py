import random
import torch
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
import os

import matplotlib.pyplot as plt
import numpy as np
import dill
from geomstats.geometry.stiefel import Stiefel
from mpl_toolkits import mplot3d
from scipy.linalg import polar
from scipy.special import hyp0f1
from utils.math_utils import (
    hyp_geom_0f1_mat_approx,
    load_hyp_geom_0f1_sage,
    eval_hyp_geom_0f1_function,
)
from tqdm import tqdm

DIRPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM_FILE_PATH = os.path.join(DIRPATH, "data", "hyp_geom_0f1_trunc_25_f.dill")


def _rejection_sample(x, st, F_mat, diag_phi):
    x = x.float()
    done = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

    with tqdm(total=x.size(0) * x.size(1)) as pbar:
        while not done.all():
            u = torch.rand(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
            st_samples = torch.from_numpy(
                st.random_uniform(n_samples=x.size(0) * x.size(1))
            )
            st_samples = st_samples.reshape(x.shape).float().to(x.device)
            accept = (
                u
                < torch.diagonal(
                    torch.matmul(F_mat.transpose(-1, -2), st_samples)
                    - torch.diag_embed(diag_phi),
                    dim1=-2,
                    dim2=-1,
                )
                .sum(-1)
                .exp()
            )

            if accept.any():
                acc = (
                    accept.unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, 1, x.size(-2), x.size(-1))
                )
                x = torch.where(acc, st_samples, x)
                done = done | acc
                op = torch.all(torch.all(done, dim=-1), dim=-1)
                pbar.update(len(op[op]) - pbar.n)

    return x


class VonMisesFisherStiefel(Distribution):
    """
    A von Mises - Fisher distribution on Stiefel Manifold.

    The ``loc``  ``conc`` args are matrices of shape 3x2 and 2x2

    Args:
        loc (Tensor): mean direction of the distribution
        concentration (Tensor): concentration matrix
    """

    # arg_constraints = {
    #     "loc": constraints.real_vector,
    #     "concentration": constraints.positive,
    # }
    # support = constraints.real_vector
    has_rsample = False

    # def __init__(self, loc, concentration, validate_args=None):
    def __init__(self, loc, Diag, validate_args=None):
        self.loc = loc
        # self.concentration = concentration
        batch_shape = self.loc.shape[:-1]
        event_shape = self.loc.shape[-1:]

        super(VonMisesFisherStiefel, self).__init__(
            batch_shape, event_shape, validate_args
        )

        # Parameters for sampling
        self.stiefel = Stiefel(3, 2)

        # Assuming V = I_{2x2} and U = M
        self.diag_phi = Diag
        self.F_mat = torch.matmul(self.loc, torch.diag_embed(Diag, dim1=-2, dim2=-1))

        # self.F_mat = torch.bmm(self.loc, self.concentration)
        # self.gamma, self.diag_phi, self.delta = torch.svd(self.F_mat)

        self.norm_factors = self.calculate_norm_factors()

        if torch.isinf(self.norm_factors).any():
            print("inf in norm_factors!")

        self.log_norm_factors = None

    def calculate_norm_factors(self):
        # norm_factors = torch.empty(
        #     self.loc.size(0), dtype=self.loc.dtype, device=self.loc.device
        # )

        fn = dill.load(open(NORM_FILE_PATH, "rb"))

        ## assuming complete batch have same conc
        D_ = 0.25 * self.diag_phi[0, :] ** 2
        D_ = D_.double()

        norm_factor_ = fn(1.5, D_[0], D_[1]).float()
        print("Norm factor: ", norm_factor_)
        norm_factors = norm_factor_.unsqueeze(dim=0).repeat(self.loc.size(0), 1)

        # D_sq_mat = 0.25 * self.diag_phi ** 2

        # for i in range(norm_factors.size(0)):
        #     norm_factors[i] = fn(1.5, D_sq_mat[i, 0], D_sq_mat[i, 1])

        return norm_factors

    def log_pdf(self, X):
        if self.log_norm_factors is None:
            self.log_norm_factors = self.norm_factors.log()

        return (
            torch.diagonal(
                torch.matmul(self.F_mat.transpose(-1, -2), X),
                dim1=-2,
                dim2=-1,
            ).sum(-1)
            - self.log_norm_factors
        )

    def pdf(self, X):
        return torch.exp(self.log_pdf(X))

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        ## Assuming shape to be N samples X B batch X 2 dimensions (event)
        shape = self._extended_shape(sample_shape)
        samples = torch.empty(shape, dtype=self.loc.dtype, device=self.loc.device)
        samples = _rejection_sample(samples, self.stiefel, self.F_mat, self.diag_phi)
        return samples.view(shape)

    def expand(self, batch_shape):
        try:
            return super(VonMisesFisherStiefel, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get("_validate_args")
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self):
        """
        Mean direction Stiefel manifold
        """
        return self.loc

    @lazy_property
    def variance(self):
        """
        The provided variance is the circular one.
        """
        # return 1 - (_log_modified_bessel_fn(self.concentration, order=1) -
        #             _log_modified_bessel_fn(self.concentration, order=0)).exp()
        raise NotADirectoryError


class VonMisesStiefel(Stiefel):
    def __init__(
        self,
        n,
        p,
        polar_form=False,
        left_svector=np.array([]),
        diag_phi=np.array([]),
        right_svector=np.array([]),
        M=np.array([]),
        K=np.array([]),
        truncate_at=15,
    ):
        super(VonMisesStiefel, self).__init__(n, p)

        # We define the von Mises distribution having parameter Matrix F
        #   F (n, p) = Gamma (n, p) * Diag_phi (p, p) * Delta (p, p)
        if (
            not polar_form
            and not left_svector.size == 0
            and not diag_phi.size == 0
            and not right_svector.size == 0
        ):
            self.gamma = left_svector
            self.diag_phi = (
                diag_phi if diag_phi.ndim == 1 else np.diag(diag_phi)
            )  # Save as a diagonal array
            self.delta = right_svector
            self.F_mat = np.matmul(
                np.matmul(self.gamma, np.diag(self.diag_phi)), self.delta
            )

            # In polar form F = M K, where
            # M := Mode of distribution, or mean orientation
            # K := Concentration, or Elliptical part
            self.M, self.K = polar(self.F_mat)

        elif polar_form and not M.size == 0 and not K.size == 0:
            self.M = M
            self.K = K
            self.F_mat = np.matmul(M, K)
            self.gamma, diag_phi, self.delta = np.linalg.svd(self.F_mat)
            self.diag_phi = diag_phi

        else:
            raise ValueError("Please pass appropriate values in constructor!")

        # self.norm_factor = hyp_geom_0f1_mat_approx(
        #     n / 2, 0.25 * self.diag_phi ** 2, trunc=truncate_at
        # )

        # self.norm_factor = load_hyp_geom_0f1_sage(b=1.5, D=0.25 * self.diag_phi ** 2)
        self.norm_factor = eval_hyp_geom_0f1_function(
            b=1.5, D=0.25 * self.diag_phi ** 2, path=NORM_FILE_PATH
        )
        print("Normalization factor: ", self.norm_factor)

    def pdf(self, X):
        return np.exp(np.trace(np.matmul(self.F_mat.T, X))) / self.norm_factor

    def random_samples(self, num=1):
        samples = []
        counter = 1
        max_height = np.exp(np.sum(self.diag_phi))

        pbar = tqdm(total=num)
        while len(samples) < num:
            u = max_height * random.random()
            st_sample = self.random_uniform()  # sample uniformly on Stiefel
            counter += 1
            if u < np.exp(
                np.trace(np.matmul(self.F_mat.T, st_sample) - np.diag(self.diag_phi))
            ):
                samples.append(st_sample)
                pbar.update(1)

        print("Percentage of samples chosen:{:.4f}".format(100 * (num / counter)))
        pbar.close()
        return np.array(samples)


if __name__ == "__main__":
    st = Stiefel(3, 2)
    mu = st.random_uniform(1)
    gamma, d_phi, delta = np.linalg.svd(mu, full_matrices=False)
    # vm = VonMisesStiefel(3, 2, gamma, np.array([0.01, 0.01]), delta)
    vm = VonMisesStiefel(3, 2, polar_form=True, M=mu, K=np.diag(np.array([10, 10])))

    data = vm.random_samples(10, scale=1e-3)

    total_prob = 0.0

    for d in data:
        total_prob += vm.pdf(d)

    print("Total prob of {} samples is:".format(10, total_prob))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_xlim((-1.0, 1.0))
    # ax.set_ylim((-1.0, 1.0))
    # ax.set_zlim((-1.0, 1.0))
    # ax.scatter(0.0, 0.0, 0.0, c="k")

    # for i in range(data.shape[0]):
    #     ax.plot(
    #         [0.0, data[i, 0, 0]],
    #         [0.0, data[i, 1, 0]],
    #         [0.0, data[i, 2, 0]],
    #         c="r",
    #         alpha=0.4,
    #     )
    #     ax.plot(
    #         [0.0, data[i, 0, 1]],
    #         [0.0, data[i, 1, 1]],
    #         [0.0, data[i, 2, 1]],
    #         c="g",
    #         alpha=0.4,
    #     )

    # ax.plot([0.0, mu[0, 0]], [0.0, mu[1, 0]], [0.0, mu[2, 0]], c="k", linewidth=4)
    # plt.show()
