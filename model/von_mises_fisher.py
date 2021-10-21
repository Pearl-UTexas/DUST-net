import math

import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
from utils.math_utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
    angle_axis_to_rotation_matrix,
)


def _vm_theta_max_value(conc):
    return conc / (2 * math.pi * (1 - (-2 * conc).exp()))


def _vm_theta_pdf(theta, conc):
    return (
        (conc / (2 * torch.sinh(conc)))
        * torch.exp(conc * torch.cos(theta))
        * torch.sin(theta)
    )


def _rejection_sample_theta(x, conc):
    done = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    while not done.all():
        r = torch.rand((2,) + x.shape, dtype=x.dtype, device=x.device)
        r1, r2 = r.unbind()
        r1 = np.pi * r1
        r2 = _vm_theta_max_value(conc) * r2
        accept = r2 < _vm_theta_pdf(r1, conc)
        if accept.any():
            x = torch.where(accept, r1, x)
            done = done | accept
    return x % np.pi


def _rotate_spherical(rot_mat, spherical):
    cart = spherical_to_cartesian(spherical).permute(1, 2, 0)
    rotated = torch.bmm(rot_mat, cart).permute(2, 0, 1)
    return cartesian_to_spherical(rotated.contiguous())


class VonMisesFisher(Distribution):
    """
    A von Mises - Fisher distribution (3D) on unit 2-sphere.

    This implementation uses spherical coordinates (r=1, theta , phi). Theta is
    defined from +z axis and phi is defined from +x axis.
    The ``loc`` = [``theta``, ``phi``] and ``conc`` args can be any real number
    (to facilitate unconstrained optimization), but are interpreted as
    theta modulo pi and phi modulo 2 pi.

    Args:
        loc (Tensor): mean direction of the distribution
        concentration (Tensor): concentration parameter
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "concentration": constraints.positive,
    }
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, loc, concentration, validate_args=None):
        # self.loc, self.concentration = broadcast_all(loc, concentration)
        self.loc = loc
        self.concentration = concentration
        batch_shape = loc.shape[:-1]
        event_shape = self.loc.shape[-1:]

        # Parameters for sampling
        self.loc_cart = spherical_to_cartesian(
                torch.cat((torch.ones_like(self.loc[:, :1]), self.loc), dim=1)
        )  # set r=1 for all theta phi tuples
        z_axes = torch.tensor([0, 0, 1], dtype=loc.dtype, device=loc.device).repeat(
            batch_shape + (1,)
        )
        angs = self.loc.view(-1, 2)[:, 0]  # extracting theta
        axes = torch.cross(z_axes, self.loc_cart, dim=-1)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        # self.rot_mat = axis_angle_to_matrix(angs.unsqueeze(-1) * axes)
        self.rot_mat = angle_axis_to_rotation_matrix(axes, angs)
        self.log_normalizer = (
            self.concentration.log()
            - self.concentration.sinh().log()
            - math.log(4 * math.pi)
        )

        super(VonMisesFisher, self).__init__(batch_shape, event_shape, validate_args)

    def log_prob_cartesian(self, x):
        # return self.concentration * dot_product_batch(self.loc_cart, x) + log_normalizer
        return (
            (x * self.loc_cart).sum(dim=-1).unsqueeze(dim=-1) * self.concentration
            + self.log_normalizer
        ).squeeze(dim=-1)

    def log_prob_spherical(self, s):
        return (
            (
                (
                    s[:, :, 0].cos() * self.loc[:, 0].cos()
                    + s[:, :, 0].sin()
                    * self.loc[:, 0].sin()
                    * (self.loc[:, 1] - s[:, :, 1]).cos()
                )
            ).unsqueeze(dim=-1)
            * self.concentration
            + self.log_normalizer
        ).squeeze(dim=-1)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        ## Assuming shape to be N samples X B batch X 2 dimensions (event)
        shape = self._extended_shape(sample_shape)
        phi = (2 * np.pi) * torch.rand(
            shape[:-1] + (1,), dtype=self.loc.dtype, device=self.loc.device
        )
        theta = torch.empty(
            shape[:-1] + (1,), dtype=self.loc.dtype, device=self.loc.device
        )
        theta = _rejection_sample_theta(theta, self.concentration)
        samples = torch.cat((torch.ones_like(theta), theta, phi), dim=-1)
        samples = _rotate_spherical(self.rot_mat, samples).view(-1, 3)[:, 1:]
        return samples.view(shape)

    def expand(self, batch_shape):
        try:
            return super(VonMisesFisher, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get("_validate_args")
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self):
        """
        The provided mean is the spherical one on unit sphere.
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
