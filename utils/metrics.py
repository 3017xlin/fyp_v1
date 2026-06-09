import time

import numpy as np
import torch

import pyvista as pv

from utils.reorganize import reorganize
from utils.batch import case_to_model_inputs, _unwrap_to_raw

NU = np.array(1.56e-5)


def rsquared(predict, true):
    mean = true.mean(dim=0)
    return 1 - ((true - predict) ** 2).sum(dim=0) / ((true - mean) ** 2).sum(dim=0)


def rel_err(a, b):
    return np.abs((a - b) / a)


def WallShearStress(Jacob_U, normals):
    S = .5 * (Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    ShearStress = 2 * NU.reshape(-1, 1, 1) * S
    ShearStress = (ShearStress * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)
    return ShearStress


@torch.no_grad()
def Infer_test(device, models, hparams, data, coef_norm=None):
    """Single forward per model over all query points.

    Uses the eager underlying KDViT (not compiled / not DDP-wrapped) so
    that variable-N eval doesn't trigger recompilation.
    """
    data_device = data.pos.device
    tim = np.zeros(len(models))
    outs = []
    for n, model in enumerate(models):
        raw = _unwrap_to_raw(model)
        raw.eval()
        inputs = case_to_model_inputs(data, raw.use_did)
        start = time.time()
        out = raw(**inputs).squeeze(0).to(data_device, dtype=data.y.dtype)
        tim[n] += time.time() - start

        if coef_norm is not None:
            mean_out = coef_norm['mean_out'].to(data_device)
            std_out = coef_norm['std_out'].to(data_device)
            surf_bc_phys = coef_norm.get('surf_bc_phys', None)
            if surf_bc_phys is None:
                surf_bc_phys = torch.zeros(4, dtype=mean_out.dtype,
                                           device=data_device)
            else:
                surf_bc_phys = surf_bc_phys.to(data_device,
                                               dtype=mean_out.dtype)
            bc_norm = (surf_bc_phys - mean_out) / (std_out + 1e-8)
            out[data.surf, :2] = bc_norm[:2]
            out[data.surf, 3] = bc_norm[3]
        else:
            out[data.surf, :2] = torch.zeros_like(out[data.surf, :2])
            out[data.surf, 3] = torch.zeros_like(out[data.surf, 3])
        outs.append(out)
    return outs, tim


def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf):
    internals = []
    airfoils = []
    nut_encoding = coef_norm.get('nut_encoding', 'linear')
    for out in outs:
        intern = internal.copy()
        aerofoil = airfoil.copy()
        out = out.detach().cpu()
        if isinstance(bool_surf, torch.Tensor):
            bool_surf_np = bool_surf.cpu().numpy()
        else:
            bool_surf_np = bool_surf
        point_mesh = intern.points[bool_surf_np, :2]
        point_surf = aerofoil.points[:, :2]
        out = (out * (coef_norm['std_out'] + 1e-8) + coef_norm['mean_out']).numpy()
        if nut_encoding == 'log':
            out[:, 3] = np.exp(out[:, 3])
        out[bool_surf_np, :2] = np.zeros_like(out[bool_surf_np, :2])
        out[bool_surf_np, 3] = np.zeros_like(out[bool_surf_np, 3])
        intern.point_data['U'][:, :2] = out[:, :2]
        intern.point_data['p'] = out[:, 2]
        intern.point_data['nut'] = out[:, 3]
        surf_p = intern.point_data['p'][bool_surf_np]
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        aerofoil.point_data['p'] = surf_p
        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)
        internals.append(intern)
        airfoils.append(aerofoil)
    return internals, airfoils


def Compute_coefficients(internals, airfoils, bool_surf, Uinf, angle, keep_vtk=False):
    if isinstance(bool_surf, torch.Tensor):
        bool_surf = bool_surf.cpu().numpy()
    coefs = []
    if keep_vtk:
        new_internals = []; new_airfoils = []
    for internal, airfoil in zip(internals, airfoils):
        intern = internal.copy()
        aerofoil = airfoil.copy()
        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]
        intern = intern.compute_derivative(scalars='U', gradient='pred_grad')
        surf_grad = intern.point_data['pred_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2]
        surf_p = intern.point_data['p'][bool_surf]
        surf_grad = reorganize(point_mesh, point_surf, surf_grad)
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        Wss_pred = WallShearStress(surf_grad, -aerofoil.point_data['Normals'])
        aerofoil.point_data['wallShearStress'] = Wss_pred
        aerofoil.point_data['p'] = surf_p
        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)
        WP_int = -aerofoil.cell_data['p'][:, None] * aerofoil.cell_data['Normals'][:, :2]
        Wss_int = (aerofoil.cell_data['wallShearStress'] * aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
        WP_int = (WP_int * aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
        force = Wss_int - WP_int
        alpha = angle * np.pi / 180
        basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        force_rot = basis @ force
        coef = 2 * force_rot / Uinf ** 2
        coefs.append(coef)
        if keep_vtk:
            new_internals.append(intern); new_airfoils.append(aerofoil)
    if keep_vtk:
        return coefs, new_internals, new_airfoils
    else:
        return coefs
