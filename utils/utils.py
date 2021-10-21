import numpy as np
import torch

from utils.math_utils import cartesian_to_spherical, spherical_to_cartesian

#######################################
#####  Dataset  Utils #################
#######################################


def interpret_labels_ours(label, bounds):
    bnds_low = torch.tensor([bnd[0] for bnd in bounds]).float().to(label.device)
    bnds_high = torch.tensor([bnd[1] for bnd in bounds]).float().to(label.device)
    corrected = label * (bnds_high - bnds_low) + bnds_low
    return corrected


def convert_to_cartesian_labels(labels):
    l_sph, m_sph, th, d = (
        labels[:, :, :2],
        labels[:, :, 2:5],
        labels[:, :, 5:6],
        labels[:, :, 6:],
    )
    l_cart = spherical_to_cartesian(
        torch.cat((torch.ones_like(l_sph[:, :, 0:1]), l_sph), dim=-1)
    )
    m_cart = spherical_to_cartesian(m_sph)
    return torch.cat((l_cart, m_cart, th, d), dim=-1)


def convert_to_spherical_labels(labels):
    l_cart, m_cart, th, d = (
        labels[:, :, :3],
        labels[:, :, 3:6],
        labels[:, :, 6:7],
        labels[:, :, 7:],
    )
    l_sph = cartesian_to_spherical(l_cart)
    m_sph = cartesian_to_spherical(m_cart)
    return torch.cat((l_sph[:, :, 1:3], m_sph, th, d), dim=-1)


def convert_predictions_VMSoftOrtho(pred_, img_seq_len=15):
    # Appropriately shaping modes of predicted distribution
    # for evaluation
    pr_ = torch.cat((pred_[:, :2], pred_[:, 6:7], pred_[:, 3:5]), dim=-1)
    pr_ = pr_.unsqueeze(dim=1).repeat((1, img_seq_len, 1))
    pr_ = torch.cat(
        (
            pr_,
            pred_[:, 8:23].unsqueeze(dim=-1),
            pred_[:, 24:-1].unsqueeze(dim=-1),
        ),
        dim=-1,
    )
    pr_ = convert_to_cartesian_labels(pr_)

    # Uncertainty
    cov_ = torch.cat(
        (pred_[:, 2:3], pred_[:, 5:6], pred_[:, 7:8], pred_[:, 23:24], pred_[:, -1:]),
        dim=-1,
    )
    return pr_, cov_


def convert_labels_VMSoftOrtho(labels_):
    return convert_to_cartesian_labels(labels_)


def convert_predictions_VMSt(pred_, img_seq_len=15):
    # Appropriately shaping modes of predicted distribution
    # for evaluation
    F_mat = pred_[:, :6].view(-1, 2, 3).transpose(-1, -2)
    U, D, V = torch.svd(F_mat, some=True, compute_uv=True)
    M = torch.matmul(U, V.transpose(-1, -2))
    K = torch.matmul(
        V, torch.matmul(torch.diag_embed(D, dim1=-2, dim2=-1), V.transpose(-1, -2))
    )
    M = M.transpose(-1, -2).reshape(-1, 6)
    pr_ = torch.cat((M[:, :3], pred_[:, 6:7] * M[:, 3:6]), dim=-1)
    pr_ = pr_.unsqueeze(dim=1).repeat((1, img_seq_len, 1))
    pr_ = torch.cat(
        (
            pr_,
            pred_[:, 8:23].unsqueeze(dim=-1),
            pred_[:, 24:-1].unsqueeze(dim=-1),
        ),
        dim=-1,
    )

    # Uncertainty
    K_diag = torch.diagonal(K, dim1=-2, dim2=-1)
    cov_ = torch.cat(
        (K_diag[:, 0:1], K_diag[:, 1:], pred_[:, 7:8], pred_[:, 23:24], pred_[:, -1:]),
        dim=-1,
    )
    return pr_, cov_


def convert_labels_VMSt(labels_):
    l, m, m_norm, conf = (
        labels_[:, :, :3],
        labels_[:, :, 3:6],
        labels_[:, :, 6:7],
        labels_[:, :, 7:9],
    )
    m = m * m_norm
    return torch.cat((l, m, conf), dim=-1)


## SVD Version
def construct_U_mat(euler_angs):
    alpha = (2 * np.pi / 6) * euler_angs[:, 0:1]  # Mapping to [0, 2*pi]
    beta = (np.pi / 6) * euler_angs[:, 1:2]  # Mapping to [0, pi]
    gamma = (2 * np.pi / 6) * euler_angs[:, 2:3]  # Mapping to [0, 2*pi]
    # alpha = 2 * np.pi * euler_angs[:, 0:1]  # Mapping to [0, 2*pi]
    # beta = np.pi * euler_angs[:, 1:2]  # Mapping to [0, pi]
    # gamma = 2 * np.pi * euler_angs[:, 2:3]  # Mapping to [0, 2*pi]
    c_a, s_a = alpha.cos(), alpha.sin()
    c_b, s_b = beta.cos(), beta.sin()
    c_g, s_g = gamma.cos(), gamma.sin()
    return torch.cat(
        (
            c_a * c_b,
            c_a * s_b * s_g - s_a * c_g,
            s_a * c_b,
            s_a * s_b * s_g + c_a * c_g,
            -s_b,
            c_b * s_g,
        ),
        dim=-1,
    ).reshape(-1, 3, 2)


def construct_V_mat(theta):
    theta = (2 * np.pi / 6) * theta  # Mapping to [0, 2*pi]
    # theta = 2 * np.pi * theta  # Mapping to [0, 2*pi]
    c_th, s_th = theta.cos(), theta.sin()
    return torch.cat(
        (c_th, -s_th, s_th, c_th),
        dim=-1,
    ).reshape(-1, 2, 2)


def convert_predictions_VMStSVD(pred_, img_seq_len=15):
    # Appropriately shaping modes of predicted distribution
    # for evaluation
    U_mat = construct_U_mat(pred_[:, :3])
    V_mat = construct_V_mat(pred_[:, 3:4])
    D = pred_[:, 4:6]

    M = torch.matmul(U_mat, V_mat.transpose(-1, -2))
    K = torch.matmul(
        V_mat,
        torch.matmul(torch.diag_embed(D, dim1=-2, dim2=-1), V_mat.transpose(-1, -2)),
    )
    M = M.transpose(-1, -2).reshape(-1, 6)
    pr_ = torch.cat((M[:, :3], pred_[:, 6:7] * M[:, 3:6]), dim=-1)
    pr_ = pr_.unsqueeze(dim=1).repeat((1, img_seq_len, 1))
    pr_ = torch.cat(
        (
            pr_,
            pred_[:, 8:23].unsqueeze(dim=-1),
            pred_[:, 24:-1].unsqueeze(dim=-1),
        ),
        dim=-1,
    )

    # Uncertainty
    K_diag = torch.diagonal(K, dim1=-2, dim2=-1)
    cov_ = torch.cat(
        (K_diag[:, 0:1], K_diag[:, 1:], pred_[:, 7:8], pred_[:, 23:24], pred_[:, -1:]),
        dim=-1,
    )

    return pr_, cov_


from model.metrics import screw_loss


def calculate_screw_accuracy(target, pred, model_type):
    target_ = target.detach()
    pred_ = pred.detach()

    if model_type == "vm-ortho":
        pr_, _ = convert_predictions_VMSoftOrtho(pred_=pred_)
        tar_ = convert_labels_VMSoftOrtho(labels_=target_)
    elif model_type == "vm-st":
        pr_, _ = convert_predictions_VMSt(pred_=pred_)
        tar_ = convert_labels_VMSt(labels_=target_)
    else:
        pr_, _ = convert_predictions_VMStSVD(pred_=pred_)
        tar_ = convert_labels_VMSt(labels_=target_)

    return screw_loss(target_=tar_, pred_=pr_)
