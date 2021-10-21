import pickle
from math import factorial, sqrt
import os
from os.path import abspath, dirname

import numpy as np
import sympy
import torch
from tqdm import tqdm
from sage.all import *
from scipy.special import gamma, iv
from sympy.utilities.lambdify import lambdify
import dill

dill.settings["recurse"] = True

dirpath = dirname(dirname(abspath(__file__)))
load(os.path.join(dirpath, "third_party", "Zonal.sage"))


def hyp_geom_coeff(a, l):
    if l == 0:
        return 1
    return (a + l - 1) * hyp_geom_coeff(a, l - 1)


def hyp_geom_0f1_coeffs(k, c, d1, d2):
    sqrt_d1_d2 = sqrt(d1 + d2)

    res = gamma(c + 2 * k) / (
        hyp_geom_coeff(c - 0.5, k) * hyp_geom_coeff(c, 2 * k) * factorial(k)
    )
    res *= (d1 * d2) ** k / (sqrt_d1_d2 ** (c + 2 * k - 1))
    res *= iv(c + 2 * k - 1, 2 * sqrt_d1_d2)
    return res


def hyp_geom_0f1_mat_approx(c, D, trunc=10):
    d1, d2 = D if D.ndim == 1 else np.diag(D)
    res = 0
    for k in range(trunc):
        res += hyp_geom_0f1_coeffs(k, c, d1, d2)
    return res


""" Using zonal polynomials"""


def eval_generalized_pochhammer_symbol(a, partition):
    K = len(partition)
    prod = 1
    for i in range(K):
        prod *= hyp_geom_coeff(a - (i / 2), partition[i])
    return prod


def accel_asc(n):
    # A method to calculate integer partitions. Shamelessly stolen from
    # https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning

    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[: k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[: k + 1]


def hyp_geom_0f1_mat_zonal(b1, D):
    d1, d2 = D if D.ndim == 1 else np.diag(D)

    val = 1
    for m in np.arange(1, 20):
        parts = sorted(list(accel_asc(m.item())))  ## Need to typecast m to int
        parts.reverse()
        for part in parts:
            part.reverse()
            zonal_poly = CZonal(part, [d1, d2])

            if zonal_poly >= 1e-7:
                val += zonal_poly / (
                    eval_generalized_pochhammer_symbol(b1, part) * factorial(m.item())
                )

    return val


def hyp_geom_coeff_symbolic(a, l):
    prod = 1
    for k in np.arange(1, l + 1):
        prod *= a + k - 1

    return prod


def eval_generalized_pochhammer_symbol_symbolic(a, partition):
    K = len(partition)
    prod = 1
    for i in range(K):
        prod *= hyp_geom_coeff_symbolic(a - (i / 2), partition[i])
    return prod


def load_hyp_geom_0f1_sage(b, D, path=None):
    if path is None:
        fname = hyp_geom_0f1_mat_zonal_symbolic()
        f = load(fname + ".sobj")
    else:
        f = load(path)

    d1_, d2_ = D if D.ndim == 1 else np.diag(D)
    return f(b1=b, d1=d1_, d2=d2_)


def eval_hyp_geom_0f1_function(b, D, path=None):
    fn = dill.load(open(path, "rb"))
    d1_, d2_ = D if D.ndim == 1 else np.diag(D)
    return fn(b, d1_, d2_)


def create_lambdified_hyp_geom_0f1_functions(path=None, trunc_at=20):
    if path is None:
        path = hyp_geom_0f1_mat_zonal_symbolic(trunc_at=trunc_at)

    with open(path, "rb") as fl:
        fns = pickle.load(fl)

    b1, d1, d2 = sympy.symbols("b1 d1 d2")
    lambda_f = lambdify([b1, d1, d2], fns["f"])
    lambda_grad_f_b1 = lambdify([b1, d1, d2], fns["grad_f_b1"])
    lambda_grad_f_d1 = lambdify([b1, d1, d2], fns["grad_f_d1"])
    lambda_grad_f_d2 = lambdify([b1, d1, d2], fns["grad_f_d2"])

    FUNC_FILE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    dill.dump(
        lambda_f,
        open(
            os.path.join(
                FUNC_FILE_PATH, "hyp_geom_0f1_trunc_{}_f.dill".format(trunc_at)
            ),
            "wb",
        ),
    )
    dill.dump(
        lambda_grad_f_b1,
        open(
            os.path.join(
                FUNC_FILE_PATH, "hyp_geom_0f1_trunc_{}_grad_f_b1.dill".format(trunc_at)
            ),
            "wb",
        ),
    )
    dill.dump(
        lambda_grad_f_d1,
        open(
            os.path.join(
                FUNC_FILE_PATH, "hyp_geom_0f1_trunc_{}_grad_f_d1.dill".format(trunc_at)
            ),
            "wb",
        ),
    )
    dill.dump(
        lambda_grad_f_d2,
        open(
            os.path.join(
                FUNC_FILE_PATH, "hyp_geom_0f1_trunc_{}_grad_f_d2.dill".format(trunc_at)
            ),
            "wb",
        ),
    )
    print("Lambdified files saved at {}".format(FUNC_FILE_PATH))


def load_hyp_geom_0f1_function(path=None):
    if path is None:
        path = hyp_geom_0f1_mat_zonal_symbolic(trunc_at=20)

    with open(path, "rb") as fl:
        fns = pickle.load(fl)

    b1, d1, d2 = sympy.symbols("b1 d1 d2")
    lambda_f = lambdify([b1, d1, d2], fns["f"])
    lambda_grad_f_b1 = lambdify([b1, d1, d2], fns["grad_f_b1"])
    lambda_grad_f_d1 = lambdify([b1, d1, d2], fns["grad_f_d1"])
    lambda_grad_f_d2 = lambdify([b1, d1, d2], fns["grad_f_d2"])
    return [lambda_f, lambda_grad_f_b1, lambda_grad_f_d1, lambda_grad_f_d2]


def hyp_geom_0f1_mat_zonal_symbolic(trunc_at=10):
    var("b1", "d1", "d2")
    hyp_geom_0f1 = 1
    for m in np.arange(1, trunc_at):
        parts = sorted(list(accel_asc(m.item())))  ## Need to typecast m to int
        parts.reverse()
        for part in parts:
            part.reverse()
            zonal_poly = CZonal(part, [d1, d2])
            hyp_geom_0f1 += zonal_poly / (
                eval_generalized_pochhammer_symbol_symbolic(b1, part)
                * factorial(m.item())
            )

    dirpath = dirname(dirname(abspath(__file__)))

    ## For sage file
    # fname = os.path.join(dirpath, "data", "hyp_geom_0f1_trunc_{}".format(trunc_at))
    # save(hyp_geom_0f1, fname)
    # print("Symbolic 0f1 file saved at:{}".format(fname + ".sobj"))

    ## Converting expression to sympy for working with pytorch
    fns = {}
    fns["f"] = hyp_geom_0f1._sympy_()
    fns["grad_f_b1"] = hyp_geom_0f1.diff(b1)._sympy_()
    fns["grad_f_d1"] = hyp_geom_0f1.diff(d1)._sympy_()
    fns["grad_f_d2"] = hyp_geom_0f1.diff(d2)._sympy_()

    fname = os.path.join(dirpath, "data", "hyp_geom_0f1_trunc_{}.pkl".format(trunc_at))
    with open(fname, "wb") as f:
        pickle.dump(fns, f)

    return fname


def hyp_geom_0f1_mat_zonal_symbolic_truncation_analysis(b, D, trunc_at=50, freq=5):
    d1_, d2_ = D if D.ndim == 1 else np.diag(D)

    dirpath = dirname(dirname(abspath(__file__)))
    vals = {}

    var("b1", "d1", "d2")
    hyp_geom_0f1 = 1

    for m in tqdm(np.arange(1, trunc_at)):
        parts = sorted(list(accel_asc(m.item())))  ## Need to typecast m to int
        parts.reverse()
        for part in parts:
            part.reverse()
            zonal_poly = CZonal(part, [d1, d2])
            hyp_geom_0f1 += zonal_poly / (
                eval_generalized_pochhammer_symbol_symbolic(b1, part)
                * factorial(m.item())
            )

        if m % freq == 0:
            vals[m] = hyp_geom_0f1(b1=b, d1=d1_, d2=d2_)

            fns = {}
            fns["f"] = hyp_geom_0f1._sympy_()
            fns["grad_f_b1"] = hyp_geom_0f1.diff(b1)._sympy_()
            fns["grad_f_d1"] = hyp_geom_0f1.diff(d1)._sympy_()
            fns["grad_f_d2"] = hyp_geom_0f1.diff(d2)._sympy_()

            fname = os.path.join(dirpath, "data", "hyp_geom_0f1_trunc_{}.pkl".format(m))
            with open(fname, "wb") as f:
                pickle.dump(fns, f)

    return vals


#######################################
#####  Metrics  #######################
#######################################


def angle_bw_vectors_batch(v1, v2, eps=1e-12):
    ## Input shapes Tensors: Batch X 3 range of arccos ins [0, pi)
    return torch.acos(
        torch.clamp(
            torch.mul(v1[:, :3], v2[:, :3]).sum(dim=-1)
            / (torch.norm(v1[:, :3], dim=-1) * torch.norm(v2[:, :3], dim=-1) + eps),
            min=-1,
            max=1,
        )
    )


def spherical_to_cartesian(sph):
    assert sph.size(-1) == 3
    orig_shape = sph.shape
    sph = sph.view(-1, 3)

    # assert (
    #     sph[:, 0] >= 0
    #     and (sph[:, 1] >= 0 and sph[:, 1] <= np.pi)
    #     and (sph[:, 2] >= 0 and sph[:, 2] < 2 * np.pi)
    # )

    x = sph[:, 0] * torch.cos(sph[:, 2]) * torch.sin(sph[:, 1])
    y = sph[:, 0] * torch.sin(sph[:, 2]) * torch.sin(sph[:, 1])
    z = sph[:, 0] * torch.cos(sph[:, 1])

    return torch.stack((x, y, z), dim=-1).view(orig_shape)


def cartesian_to_spherical(cart):
    assert cart.size(-1) == 3
    orig_shape = cart.shape
    cart = cart.view(-1, 3)

    r = cart.norm(dim=-1, keepdim=True)
    theta = torch.atan2(cart[:, :-1].norm(dim=-1, keepdim=True), cart[:, -1:])
    phi = (torch.atan2(cart[:, 1], cart[:, 0]) + 2 * np.pi) % (
        2 * np.pi
    )  # correcting range of phi to be between 0 to 2pi
    return torch.cat((r, theta, phi.unsqueeze(-1)), dim=-1).view(orig_shape)


def dot_product_batch(v1, v2):
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).squeeze_(1)


def chordal_distance(v1, v2, r=1):
    v1 = v1.numpy()
    v2 = v2.numpy()
    return r * np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


def angle_axis_to_rotation_matrix(angle_axis, theta):
    # Stolen from PyTorch geometry library. Modified for our code
    angle_axis_shape = angle_axis.shape
    angle_axis_ = angle_axis.contiguous().view(-1, 3)
    theta_ = theta.contiguous().view(-1, 1)

    k_one = 1.0
    normed_axes = angle_axis_ / angle_axis_.norm(dim=-1, keepdim=True)
    wx, wy, wz = torch.chunk(normed_axes, 3, dim=1)
    cos_theta = torch.cos(theta_)
    sin_theta = torch.sin(theta_)

    r00 = cos_theta + wx * wx * (k_one - cos_theta)
    r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
    r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
    r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
    r11 = cos_theta + wy * wy * (k_one - cos_theta)
    r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
    r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
    r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
    r22 = cos_theta + wz * wz * (k_one - cos_theta)
    rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
    return rotation_matrix.view(list(angle_axis_shape[:-1]) + [3, 3])


def distance_bw_plucker_lines(target, prediction, eps=1e-10):
    """Input shapes Tensors: Batch X #Images X 8
    # Based on formula from Plücker Coordinates for Lines in the Space by Prof. Yan-bin Jia
    # Verified by https://keisan.casio.com/exec/system/1223531414
    """
    norm_cross_prod = torch.norm(
        torch.cross(target[:, :, :3], prediction[:, :, :3], dim=-1), dim=-1
    )
    dist = torch.zeros_like(norm_cross_prod)

    # Checking for Parallel Lines
    if torch.any(norm_cross_prod <= eps):
        zero_idxs = (norm_cross_prod <= eps).nonzero(as_tuple=True)
        scales = (
            torch.norm(prediction[zero_idxs][:, :3], dim=-1)
            / torch.norm(target[zero_idxs][:, :3], dim=-1)
            + eps
        )
        dist[zero_idxs] = torch.norm(
            torch.cross(
                target[zero_idxs][:, :3],
                (
                    target[zero_idxs][:, 3:6]
                    - prediction[zero_idxs][:, 3:6] / scales.unsqueeze(-1)
                ),
            ),
            dim=-1,
        ) / (
            torch.mul(target[zero_idxs][:, :3], target[zero_idxs][:, :3]).sum(dim=-1)
            + eps
        )

    # Skew Lines: Non zero cross product
    nonzero_idxs = (norm_cross_prod > eps).nonzero(as_tuple=True)
    dist[nonzero_idxs] = torch.abs(
        torch.mul(target[nonzero_idxs][:, :3], prediction[nonzero_idxs][:, 3:6]).sum(
            dim=-1
        )
        + torch.mul(target[nonzero_idxs][:, 3:6], prediction[nonzero_idxs][:, :3]).sum(
            dim=-1
        )
    ) / (norm_cross_prod[nonzero_idxs] + eps)
    return dist


def orientation_difference_bw_plucker_lines(target, prediction, eps=1e-6):
    """Input shapes Tensors: Batch X #Images X 8
    range of arccos ins [0, pi)"""
    return torch.acos(
        torch.clamp(
            torch.mul(target[:, :, :3], prediction[:, :, :3]).sum(dim=-1)
            / (
                torch.norm(target[:, :, :3], dim=-1)
                * torch.norm(prediction[:, :, :3], dim=-1)
                + eps
            ),
            min=-1,
            max=1,
        )
    )


def theta_config_error(target, prediction):
    rot_tar = angle_axis_to_rotation_matrix(target[:, :, :3], target[:, :, -2]).view(
        -1, 3, 3
    )
    rot_pred = angle_axis_to_rotation_matrix(
        prediction[:, :, :3], prediction[:, :, -2]
    ).view(-1, 3, 3)
    I_ = torch.eye(3).reshape((1, 3, 3))
    I_ = I_.repeat(rot_tar.size(0), 1, 1).to(target.device)
    return torch.norm(
        I_ - torch.bmm(rot_pred, rot_tar.transpose(1, 2)), dim=(1, 2), p=2
    ).view(target.shape[:2])


def d_config_error(target, prediction):
    tar_d = target[:, :, -1].unsqueeze(-1)
    pred_d = prediction[:, :, -1].unsqueeze(-1)
    tar_d = target[:, :, :3] * tar_d
    pred_d = prediction[:, :, :3] * pred_d
    return (tar_d - pred_d).norm(dim=-1)
