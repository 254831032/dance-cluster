"""骨架特征：补帧 → 归一化 → 角度特征 → 镜像 → 重采样。

对「不同体型、不同机位、镜像」刻意做了消除：
- 平移：以髋中心为原点；
- 缩放：以躯干长度（髋中心→肩中点）为 1 个单位；
- 朝向：用「髋-肩」轴建立人体坐标系，消除身体朝向/机位差异；
- 镜像：左右点交换 + x 取反，与「角度特征」配合。
"""

from __future__ import annotations

import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
NOSE = 0

MIRROR_MAP = np.arange(33)
for _a, _b in [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]:
    MIRROR_MAP[_a], MIRROR_MAP[_b] = _b, _a

ANGLE_TRIPLETS = [
    (L_SHOULDER, L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW, R_WRIST),
    (L_HIP, L_SHOULDER, L_ELBOW),
    (R_HIP, R_SHOULDER, R_ELBOW),
    (L_SHOULDER, L_HIP, L_KNEE),
    (R_SHOULDER, R_HIP, R_KNEE),
    (L_HIP, L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE, R_ANKLE),
]
POSITION_JOINTS = [L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

EPS = 1e-8


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """b 处的夹角，(..., 3) 输入，返回 (...)。"""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / np.maximum(n1 * n2, EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def fill_missing(world: np.ndarray, visibility: np.ndarray, present: np.ndarray,
                 min_visibility: float = 0.35) -> np.ndarray:
    """按时间对缺失点做线性插值。world: (F,33,3)。"""
    if world.shape[0] == 0:
        return world
    out = world.copy()
    bad = (np.asarray(visibility) < min_visibility) | (~np.asarray(present))[:, None]
    for lm in range(world.shape[1]):
        for axis in range(3):
            vals = out[:, lm, axis]
            good = ~bad[:, lm]
            if good.sum() < 2:
                vals[~good] = vals[good].mean() if good.any() else 0.0
                continue
            idx = np.arange(world.shape[0])
            vals[~good] = np.interp(idx[~good], idx[good], vals[good])
    return out


def normalize(world: np.ndarray) -> np.ndarray:
    """移到髋中心、按躯干长度缩放、转到人体坐标系。返回 (F,33,3)。"""
    if world.shape[0] == 0:
        return world
    hip_c = (world[:, L_HIP] + world[:, R_HIP]) / 2
    sh_c = (world[:, L_SHOULDER] + world[:, R_SHOULDER]) / 2
    origin = hip_c[:, None, :]
    p = world - origin

    scale = np.linalg.norm(sh_c - hip_c, axis=-1)
    scale = np.maximum(scale, 1e-3)[:, None, None]
    p = p / scale

    up = (sh_c - hip_c)
    up = up / np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), EPS)
    right = world[:, R_HIP] - world[:, L_HIP]
    right = right - up * np.sum(right * up, axis=-1, keepdims=True)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), EPS)
    fwd = np.cross(right, up)
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=-1, keepdims=True), EPS)

    # rot[..., j] 是第 j 根基向量；把 p 投影到这组基上
    rot = np.stack([right, up, fwd], axis=-1)          # (F,3,3)
    return np.einsum("fli,fij->flj", p, rot)           # p' = p·基


def joint_angles(x: np.ndarray) -> np.ndarray:
    """(F,33,3) -> (F, 8) 关节角。"""
    return np.stack([_angle(x[:, a], x[:, b], x[:, c]) for a, b, c in ANGLE_TRIPLETS], axis=-1)


def mirror(x: np.ndarray) -> np.ndarray:
    """左右镜像：交换对称点，x 取反。"""
    y = x[:, MIRROR_MAP]
    y = y * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    return y


def resample(x: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    """线性插值重采样到 target_fps。"""
    if x.shape[0] == 0 or abs(src_fps - target_fps) < 1e-3:
        return x
    n_out = max(2, int(round(x.shape[0] * target_fps / src_fps)))
    t_in = np.linspace(0, 1, x.shape[0])
    t_out = np.linspace(0, 1, n_out)
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(t_out, t_in, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((n_out,) + x.shape[1:]).astype(np.float32)


def feature_vector(x: np.ndarray) -> np.ndarray:
    """(F,33,3) -> (F, 8+8*3) 的定长帧特征：关节角 + 关键点人体坐标。"""
    ang = joint_angles(x)
    pos = x[:, POSITION_JOINTS].reshape(x.shape[0], -1)
    return np.concatenate([ang, pos], axis=1).astype(np.float32)


def prepare_coords(
    world: np.ndarray,
    visibility: np.ndarray,
    present: np.ndarray,
    src_fps: float,
    target_fps: float = 15.0,
    min_visibility: float = 0.35,
    trim: bool = True,
) -> np.ndarray:
    """完整链路：插值 → 归一化 → 重采样。返回人体坐标系下的 (F,33,3)。"""
    if world.shape[0] == 0 or not np.any(present):
        return np.zeros((0, 33, 3), dtype=np.float32)
    x = fill_missing(world, visibility, present, min_visibility)
    x = normalize(x)
    x = resample(x, src_fps, target_fps)
    if trim and x.shape[0] > 4:
        energy = np.linalg.norm(x, axis=(1, 2))
        keep = np.where(energy > 1e-6)[0]
        if keep.size > 2:
            x = x[keep[0]: keep[-1] + 1]
    return x.astype(np.float32)


def prepare_features(
    world: np.ndarray,
    visibility: np.ndarray,
    present: np.ndarray,
    src_fps: float,
    target_fps: float = 15.0,
    min_visibility: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (原特征, 镜像特征)，供匹配时取最小距离。"""
    x = prepare_coords(world, visibility, present, src_fps, target_fps, min_visibility)
    if x.shape[0] == 0:
        empty = np.zeros((0, 32), dtype=np.float32)
        return empty, empty
    return feature_vector(x), feature_vector(mirror(x))

