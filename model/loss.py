import math
import os

import torch
from utils.math_utils import load_hyp_geom_0f1_function
from utils.utils import construct_U_mat, construct_V_mat
import pdb


def vm_log_prob_spherical(loc, conc, x, eps=1e-9):
    ## Assume shape of x to be N samples X Batch X event size ( torch distribution convention)
    # log_normalizer = (
    #     (conc + eps).log() - ((conc + eps).sinh() + eps).log() - math.log(4 * math.pi)
    # )
    log_normalizer = (
        (conc + eps).log()
        - conc
        - math.log(2 * math.pi)
        - (1 - (-2 * conc).exp() + eps).log()
    )

    return (
        (
            x[:, :, 0].cos() * loc[:, 0].cos()
            + x[:, :, 0].sin() * loc[:, 0].sin() * (loc[:, 1] - x[:, :, 1]).cos()
        ).unsqueeze(dim=-1)
        * (conc + eps)
        + log_normalizer
    ).squeeze(dim=-1)


def soft_ortho_loss(l, m):
    ## Working with spherical coordinates
    return (
        l[:, 0].cos() * m[:, 0].cos()
        + l[:, 0].sin() * m[:, 0].sin() * (l[:, 1] - m[:, 1]).cos()
    ) ** 2


def VMSoftOrthoLoss(target, prediction, eps=1e-10, training_stage=None):
    l_nll = (
        -vm_log_prob_spherical(
            loc=prediction[:, :2],
            conc=prediction[:, 2:3],
            x=target.transpose(0, 1)[:, :, :2],
            eps=eps,
        )
        .sum(dim=0)
        .mean()
    )

    m_nll = (
        -vm_log_prob_spherical(
            loc=prediction[:, 3:5],
            conc=prediction[:, 5:6],
            x=target.transpose(0, 1)[:, :, 3:5],
            eps=eps,
        )
        .sum(dim=0)
        .mean()
    )

    mod_m_nll = (
        -target.size(1) * (prediction[:, 7] + eps).log()
        + 0.5
        * prediction[:, 7] ** 2
        * ((target[:, :, 2] - prediction[:, 6:7]) ** 2).sum(dim=-1)
    ).mean()

    th_nll = (
        -target.size(1) * (prediction[:, 23] + eps).log()
        + 0.5
        * prediction[:, 23] ** 2
        * ((target[:, :, -2] - prediction[:, 8:23]) ** 2).sum(dim=-1)
    ).mean()

    d_nll = (
        -target.size(1) * (prediction[:, -1] + eps).log()
        + 0.5
        * prediction[:, -1] ** 2
        * ((target[:, :, -1] - prediction[:, 24:-1]) ** 2).sum(dim=-1)
    ).mean()

    ortho = soft_ortho_loss(prediction[:, :2], prediction[:, 3:5]).mean()

    loss = l_nll + m_nll + mod_m_nll + th_nll + d_nll + ortho
    # print("l_nll:{:.4f}, m_nll:{:.4f}, mod_m_nll:{:.4f}, th_nll:{:.4f}, d_nll:{:.4f}, ortho:{:.4f}".format(l_nll, m_nll, mod_m_nll, th_nll, d_nll, ortho))
    return loss


## Stiefel loss
class LogVMStiefelNormFactor(torch.autograd.Function):
    DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FUNCTION_PATH = os.path.join(DIR_PATH, "data", "hyp_geom_0f1_trunc_25.pkl")
    fn_, grad_f_b1_, grad_f_d1_, grad_f_d2_ = load_hyp_geom_0f1_function(
        path=FUNCTION_PATH
    )

    @staticmethod
    def forward(ctx, D, fn=fn_):
        D_sq_mat = (0.25) * D.double() ** 2
        b = 1.5 * torch.ones_like(D_sq_mat[:, 0:1])
        fn_val = fn(b, D_sq_mat[:, 0:1], D_sq_mat[:, 1:2]).squeeze()
        ctx.save_for_backward(D, D_sq_mat, fn_val)
        return fn_val.log().float()

    @staticmethod
    def backward(ctx, grad_output, fn=fn_, grad_f_d1=grad_f_d1_, grad_f_d2=grad_f_d2_):
        D, D_sq_mat, fn_val = ctx.saved_tensors
        grad_D_mat = None

        b = 1.5 * torch.ones_like(D_sq_mat[:, 0:1])
        grad_D_sq_mat = torch.cat(
            (
                grad_f_d1(b, D_sq_mat[:, 0:1], D_sq_mat[:, 1:2]),
                grad_f_d2(b, D_sq_mat[:, 0:1], D_sq_mat[:, 1:2]),
            ),
            dim=-1,
        )

        if grad_D_sq_mat.isnan().any():
            print("NaNs in grad_D_sq_mat!")

        if torch.isneginf(grad_D_sq_mat).any():
            print("neginf in grad_D_sq_mat!")

        if torch.isinf(grad_D_sq_mat).any():
            print("inf in grad_D_sq_mat!")

        if (
            grad_D_sq_mat.isnan().any()
            or torch.isneginf(grad_D_sq_mat).any()
            or torch.isinf(grad_D_sq_mat).any()
        ):
            pdb.set_trace()

        # gradient of sq
        grad_D_mat = 0.5 * grad_D_sq_mat * D

        # gradient of log
        grad_D_mat = (torch.div(grad_D_mat, fn_val.unsqueeze(-1))).float()
        # print("LogVM norm factor grad: ", (grad_output.unsqueeze(-1).unsqueeze(-1) * grad_F_mat).mean())
        return grad_output.unsqueeze(-1) * grad_D_mat


def VMStiefelLoss(target, prediction, eps=1e-9, training_stage=1):
    F_mat = prediction[:, :6].view(-1, 2, 3).transpose(-1, -2)
    X = (
        target[:, :, :6].view(target.size(0), target.size(1), 2, 3).transpose(-1, -2)
    )  # this ensure columns have norm 1

    U, D, V = torch.svd(F_mat, some=True, compute_uv=True)

    if training_stage in [1, 2]:
        D_ = 1.0 * torch.ones_like(D)
        F_ = torch.matmul(
            U, torch.matmul(torch.diag_embed(D_, dim1=-2, dim2=-1), V.transpose(-2, -1))
        )

        dist_nll = (
            target.size(1) * LogVMStiefelNormFactor.apply(D_)
            - torch.diagonal(
                torch.matmul(F_.unsqueeze(dim=1).transpose(-1, -2), X), dim1=-2, dim2=-1
            )
            .sum(-1)  # trace
            .sum(-1)  # summing for total error
        ).mean()  # mean across batches

    else:
        D = torch.clamp(D, min=1e-6, max=50.0)

        dist_nll = (
            target.size(1) * LogVMStiefelNormFactor.apply(D)
            - torch.diagonal(
                torch.matmul(F_mat.unsqueeze(dim=1).transpose(-1, -2), X),
                dim1=-2,
                dim2=-1,
            )
            .sum(-1)  # trace
            .sum(-1)  # summing for total error
        ).mean()  # mean across batches

    mod_m_nll = (
        -target.size(1) * (prediction[:, 7] + eps).log()
        + 0.5
        * prediction[:, 7] ** 2
        * ((target[:, :, 6] - prediction[:, 6:7]) ** 2).sum(dim=-1)
    ).mean()

    th_nll = (
        -target.size(1) * (prediction[:, 23] + eps).log()
        + 0.5
        * prediction[:, 23] ** 2
        * ((target[:, :, -2] - prediction[:, 8:23]) ** 2).sum(dim=-1)
    ).mean()

    d_nll = (
        -target.size(1) * (prediction[:, -1] + eps).log()
        + 0.5
        * prediction[:, -1] ** 2
        * ((target[:, :, -1] - prediction[:, 24:-1]) ** 2).sum(dim=-1)
    ).mean()

    if training_stage == 1:
        loss = dist_nll
    elif training_stage == 2:
        loss = dist_nll + mod_m_nll + th_nll + d_nll
    elif training_stage == 3:
        loss = (1 / target.size(1)) * dist_nll + mod_m_nll + th_nll + d_nll
        print("\t\tD_1:{:.4f}   D_2:{:.4f}".format(D[:, 0].mean(), D[:, 1].mean()))
    else:
        raise ValueError("Training State can only be one of 1, 2, or 3")

    # print("dist_nll:{:.4f}, mod_m_nll:{:.4f}, th_nll:{:.4f}, d_nll:{:.4f}".format(dist_nll, mod_m_nll, th_nll, d_nll))

    return loss


def VMStiefelSVDLoss(target, prediction, eps=1e-9, training_stage=1):
    X = (
        target[:, :, :6].view(target.size(0), target.size(1), 2, 3).transpose(-1, -2)
    )  # this ensure columns have norm 1

    U_mat = construct_U_mat(prediction[:, :3])
    V_mat = construct_V_mat(prediction[:, 3:4])
    D = prediction[:, 4:6]

    if training_stage in [1, 2]:
        D_ = 1.0 * torch.ones_like(D)
        F_trans_ = torch.matmul(
            torch.matmul(V_mat, torch.diag_embed(D_, dim1=-2, dim2=-1)),
            U_mat.transpose(-2, -1),
        )

        dist_nll = (
            target.size(1) * LogVMStiefelNormFactor.apply(D_)
            - torch.diagonal(
                torch.matmul(F_trans_.unsqueeze(dim=1), X), dim1=-2, dim2=-1
            )
            .sum(-1)  # trace
            .sum(-1)  # summing for total error
        ).mean()  # mean across batches

    else:
        D = torch.clamp(D, min=1e-6, max=50.0)
        F_trans = torch.matmul(
            torch.matmul(V_mat, torch.diag_embed(D, dim1=-2, dim2=-1)),
            U_mat.transpose(-2, -1),
        )
        dist_nll = (
            target.size(1) * LogVMStiefelNormFactor.apply(D)
            - torch.diagonal(
                torch.matmul(F_trans.unsqueeze(dim=1), X), dim1=-2, dim2=-1
            )
            .sum(-1)  # trace
            .sum(-1)  # summing for total error
        ).mean()  # mean across batches

    mod_m_nll = (
        -target.size(1) * (prediction[:, 7] + eps).log()
        + 0.5
        * prediction[:, 7] ** 2
        * ((target[:, :, 6] - prediction[:, 6:7]) ** 2).sum(dim=-1)
    ).mean()

    th_nll = (
        -target.size(1) * (prediction[:, 23] + eps).log()
        + 0.5
        * prediction[:, 23] ** 2
        * ((target[:, :, -2] - prediction[:, 8:23]) ** 2).sum(dim=-1)
    ).mean()

    d_nll = (
        -target.size(1) * (prediction[:, -1] + eps).log()
        + 0.5
        * prediction[:, -1] ** 2
        * ((target[:, :, -1] - prediction[:, 24:-1]) ** 2).sum(dim=-1)
    ).mean()

    if training_stage == 1:
        loss = dist_nll
    elif training_stage == 2:
        loss = dist_nll + mod_m_nll + th_nll + d_nll
    elif training_stage == 3:
        loss = (1 / target.size(1)) * dist_nll + mod_m_nll + th_nll + d_nll
        print("\t\tD_1:{:.4f}   D_2:{:.4f}".format(D[:, 0].mean(), D[:, 1].mean()))
    else:
        raise ValueError("Training State can only be one of 1, 2, or 3")

    # print("dist_nll:{:.4f}, mod_m_nll:{:.4f}, th_nll:{:.4f}, d_nll:{:.4f}".format(dist_nll, mod_m_nll, th_nll, d_nll))

    return loss
