import torch
from utils.math_utils import (
    angle_bw_vectors_batch,
    spherical_to_cartesian,
    orientation_difference_bw_plucker_lines,
    distance_bw_plucker_lines,
    angle_axis_to_rotation_matrix,
    theta_config_error,
    d_config_error,
)


def maad(target_, prediction_):
    return (torch.abs(target_ - prediction_)).mean()


def angular_maad(target_, prediction_):
    maad_loss = target_.new_zeros((1,))
    for i in range(target_.size(0)):
        maad_loss += torch.abs(
            angle_bw_vectors_batch(target_[i, :, :], prediction_[i, :, :])
        ).mean()
    return maad_loss / target_.size(0)


## Screw Loss
def screw_loss(target_, pred_, detailed=False):
    """Based on Spatial distance
    Input shapes: Batch X Objects X images+
    """
    # Spatial Distance loss
    ori_error = orientation_difference_bw_plucker_lines(target_, pred_)
    dist_error = distance_bw_plucker_lines(target_, pred_)

    # Configuration Loss
    theta_error = theta_config_error(target_, pred_)
    d_error = d_config_error(target_, pred_)

    # orthogonal
    ortho_error = torch.mul(pred_[:, :, :3], pred_[:, :, 3:6]).sum(dim=-1)

    if detailed:
        return ori_error, dist_error, theta_error, d_error, ortho_error
    else:
        return (
            ori_error.abs().mean(),
            dist_error.abs().mean(),
            theta_error.abs().mean(),
            d_error.abs().mean(),
            ortho_error.abs().mean(),
        )
