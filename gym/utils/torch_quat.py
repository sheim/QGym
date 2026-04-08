"""Pure-torch quaternion utilities — no IsaacGym dependency.

All functions use the scalar-last [x, y, z, w] convention, matching IsaacGym's
torch_utils interface.  These are used as fallbacks when IsaacGym is not
installed (e.g. MuJoCo-only environments).
"""

import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize along the last dimension."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q.  q is [x, y, z, w]."""
    xyz = q[..., :3]
    w = q[..., 3:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q.  q is [x, y, z, w]."""
    q_w = q[..., 3:]  # [..., 1]
    q_vec = q[..., :3]  # [..., 3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """Euler angles (XYZ extrinsic) to quaternion [x, y, z, w]."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qx, qy, qz, qw], dim=-1)


def get_axis_params(value, axis_idx, x_val=0.0, dtype=float, n_dims=3):
    """Returns a list with `value` at `axis_idx`, `x_val` elsewhere."""
    params = [x_val] * n_dims
    params[axis_idx] = value
    return params


def to_torch(x, dtype=torch.float, device="cpu", requires_grad=False):
    """Convert array-like to torch tensor."""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
