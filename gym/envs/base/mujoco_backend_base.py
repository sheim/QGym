"""Shared base for MuJoCo backends (CPU and Warp).

Handles URDF loading via MjSpec, model configuration (free joint, ground
plane, physics params), metadata extraction, and contact index building.
Subclasses implement tensor allocation, step, reset, and rendering.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
import mujoco

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.base.sim_backend import SimBackend

# Quaternion convention helpers: MuJoCo [w,x,y,z] ↔ task-layer [x,y,z,w]
WXYZ_TO_XYZW = [1, 2, 3, 0]
XYZW_TO_WXYZ = [3, 0, 1, 2]


class MuJocoBackendBase(SimBackend):
    """Abstract base with shared MuJoCo setup logic.

    Subclasses must implement: _allocate_tensors(), step(), reset_dof_state(),
    and the state tensor properties (dof_pos, dof_vel, dof_state, root_states,
    rigid_body_states, contact_forces).
    """

    def __init__(self) -> None:
        self._mjm: mujoco.MjModel = None
        self._device: str = "cpu"
        self._num_envs: int = 0
        self._num_dof: int = 0
        self._num_bodies: int = 0
        self._dof_names: list = []
        self._body_names: list = []

        # Floating-base offsets (0 for fixed-base)
        self._has_free_joint: bool = False
        self._qpos_offset: int = 0
        self._qvel_offset: int = 0

        # Contact body indices (set by setup)
        self._penalised_contact_indices: torch.Tensor = None
        self._termination_contact_indices: torch.Tensor = None

    # ── SimBackend.device ─────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def num_dof(self) -> int:
        return self._num_dof

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def dof_names(self) -> list:
        return self._dof_names

    @property
    def body_names(self) -> list:
        return self._body_names

    def find_body_index(self, name: str) -> int:
        return self._body_names.index(name)

    # ── Contact indices ────────────────────────────────────────────────────────

    @property
    def penalised_contact_indices(self) -> torch.Tensor:
        return self._penalised_contact_indices

    @property
    def termination_contact_indices(self) -> torch.Tensor:
        return self._termination_contact_indices

    # ── World building ─────────────────────────────────────────────────────────

    def _load_model(self, cfg) -> mujoco.MjModel:
        """Load URDF, configure model (free joint, ground, physics), return MjModel."""
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        # Cache URDF effort/velocity limits — MuJoCo drops these on import
        # (it expects them on actuators, but we have none).
        self._urdf_limits = self._parse_urdf_limits(asset_path)
        spec = self._load_urdf_spec(asset_path)
        spec.compiler.balanceinertia = True

        # Add free joint for floating-base robots
        if not getattr(cfg.asset, "fix_base_link", True):
            root_body = spec.worldbody.first_body()
            freejoint = root_body.add_freejoint()
            freejoint.name = "root"

        # Menagerie-style viewer defaults
        spec.visual.global_.azimuth = 150
        spec.visual.global_.elevation = -20
        spec.visual.quality.shadowsize = 4096
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]

        # Gradient skybox + directional light apply to all scenes
        sky = spec.add_texture()
        sky.name = "skybox"
        sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1 = [0.3, 0.5, 0.7]
        sky.rgb2 = [0.0, 0.0, 0.0]
        sky.width = 512
        sky.height = 3072

        light = spec.worldbody.add_light()
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.pos = [0, 0, 1.5]
        light.dir = [0, 0, -1]
        light.castshadow = True

        # Checker ground plane only when terrain config requests one
        terrain_cfg = getattr(cfg, "terrain", None)
        if (
            terrain_cfg is not None
            and getattr(terrain_cfg, "mesh_type", None) == "plane"
        ):
            gtex = spec.add_texture()
            gtex.name = "groundplane"
            gtex.type = mujoco.mjtTexture.mjTEXTURE_2D
            gtex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
            gtex.mark = mujoco.mjtMark.mjMARK_EDGE
            gtex.rgb1 = [0.2, 0.3, 0.4]
            gtex.rgb2 = [0.1, 0.2, 0.3]
            gtex.markrgb = [0.8, 0.8, 0.8]
            gtex.width = 300
            gtex.height = 300

            gmat = spec.add_material()
            gmat.name = "groundplane"
            gmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
            gmat.texrepeat = [5, 5]
            gmat.texuniform = True
            gmat.reflectance = 0.2

            ground = spec.worldbody.add_geom()
            ground.type = mujoco.mjtGeom.mjGEOM_PLANE
            ground.size = [0, 0, 0.05]
            ground.material = "groundplane"
            sf = getattr(terrain_cfg, "static_friction", 1.0)
            df = getattr(terrain_cfg, "dynamic_friction", 1.0)
            ground.friction = [sf, df, 0.0001]

        # Check for manually set mjModel attributes
        if hasattr(cfg, "mjspec_attributes"):
            for name in dir(cfg.mjspec_attributes):
                if not name.startswith("_"):
                    setattr(spec, name, getattr(cfg.mjspec_attributes, name))

        if hasattr(cfg, "mjspec_option_attributes"):
            for name in dir(cfg.mjspec_option_attributes):
                if not name.startswith("_"):
                    setattr(spec.option, name, getattr(cfg.mjspec_option_attributes, name))

        mjm = spec.compile()

        # Physics parameters from cfg
        sim_dt = getattr(cfg, "sim_dt", None)
        if sim_dt is None:
            sim_dt = getattr(cfg.sim, "dt", 0.005) if hasattr(cfg, "sim") else 0.005
        mjm.opt.timestep = sim_dt
        sim_cfg = getattr(cfg, "sim", None)
        if sim_cfg is not None and hasattr(sim_cfg, "gravity"):
            mjm.opt.gravity[:] = np.array(sim_cfg.gravity, dtype=np.float64)
        if getattr(cfg.asset, "disable_gravity", False):
            mjm.opt.gravity[:] = 0.0

        return mjm

    def _configure_model(self, mjm: mujoco.MjModel, cfg, device: str) -> None:
        """Detect floating-base, set damping/contacts, extract metadata."""
        self._mjm = mjm

        # Detect floating-base (free joint adds 1 extra qpos for quaternion)
        self._has_free_joint = mjm.nq == mjm.nv + 1
        if self._has_free_joint:
            self._qpos_offset = 7
            self._qvel_offset = 6
            self._num_dof = mjm.nv - 6
        else:
            assert mjm.nq == mjm.nv, (
                f"Unexpected nq/nv: nq={mjm.nq}, nv={mjm.nv}. "
                f"Expected nq==nv (fixed-base) or nq==nv+1 (free joint)."
            )
            self._qpos_offset = 0
            self._qvel_offset = 0
            self._num_dof = mjm.nv

        # Apply damping/armature to actuated DOFs only
        mjm.dof_damping[self._qvel_offset :] = cfg.asset.joint_damping
        mjm.dof_armature[self._qvel_offset :] = getattr(cfg.asset, "rotor_inertia", 0.0)

        # Contacts: disable for fixed-base (no ground), keep for floating-base
        if not self._has_free_joint:
            mjm.geom_contype[:] = 0
            mjm.geom_conaffinity[:] = 0

        # Extract metadata
        self._body_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(mjm.nbody)
        ]
        self._num_bodies = mjm.nbody

        jnt_start = 1 if self._has_free_joint else 0
        self._dof_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(jnt_start, mjm.njnt)
        ]

        # Contact index tensors
        self._penalised_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "penalize_contacts_on", []), device
        )
        self._termination_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "terminate_after_contacts_on", []), device
        )

    def _run_task_callbacks(self, mjm: mujoco.MjModel, task) -> None:
        """Call task hooks (num_dof, env origins, dof props)."""
        if task is not None:
            task.num_dof = self._num_dof
        if task is not None and hasattr(task, "_get_env_origins"):
            task._get_env_origins()
        if task is not None and hasattr(task, "_process_dof_props"):
            task._process_dof_props(self._make_dof_props(mjm), env_id=0)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_contact_indices(self, name_patterns: list, device: str) -> torch.Tensor:
        indices = []
        for pattern in name_patterns:
            for i, bname in enumerate(self._body_names):
                if pattern in bname:
                    indices.append(i)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _make_dof_props(self, mjm: mujoco.MjModel) -> dict:
        """Build DOF-properties dict expected by task._process_dof_props.

        Keys match the IsaacGym interface: lower, upper, velocity, effort.
        Only includes actuated joints (skips free joint for floating-base).
        Effort/velocity come from <limit> tags parsed out of the URDF —
        MuJoCo's URDF importer discards those because it expects them on
        actuators, which this backend does not create.
        """
        jnt_start = 1 if self._has_free_joint else 0
        n = mjm.njnt - jnt_start
        limited = mjm.jnt_limited[jnt_start : mjm.njnt].astype(bool)
        lower = np.where(limited, mjm.jnt_range[jnt_start : mjm.njnt, 0], -1e6)
        upper = np.where(limited, mjm.jnt_range[jnt_start : mjm.njnt, 1], 1e6)
        effort = np.full(n, 1e6, dtype=np.float64)
        velocity = np.full(n, 1e6, dtype=np.float64)
        for i, jname in enumerate(self._dof_names):
            if jname in self._urdf_limits:
                eff, vel = self._urdf_limits[jname]
                effort[i] = eff
                velocity[i] = vel
        return {"lower": lower, "upper": upper, "velocity": velocity, "effort": effort}

    @staticmethod
    def _load_urdf_spec(urdf_path: str) -> "mujoco.MjSpec":
        """Load URDF as MjSpec, preserving <visual> meshes where MuJoCo supports them.

        MuJoCo's URDF importer discards <visual> geoms during file parse unless
        the URDF embeds <mujoco><compiler discardvisual="false"/></mujoco>. The
        same embedded tag also leaves the spec mutable post-parse — without it,
        later add_texture/add_material calls are silently pruned at compile.
        We always inject the tag; for URDFs whose <visual> blocks reference
        meshes MuJoCo can't read (.dae/.collada), we first strip those blocks
        so the compile doesn't fail looking for a decoder.
        """
        root = ET.parse(urdf_path).getroot()
        has_unsupported = any(
            (m.get("filename") or "").lower().endswith((".dae", ".collada"))
            for m in root.iter("mesh")
        )
        if has_unsupported:
            for link in root.iter("link"):
                for v in list(link.findall("visual")):
                    link.remove(v)
        mj = root.find("mujoco")
        if mj is None:
            mj = ET.SubElement(root, "mujoco")
        comp = mj.find("compiler")
        if comp is None:
            comp = ET.SubElement(mj, "compiler")
        comp.set("discardvisual", "false")
        comp.set("strippath", "false")
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        spec.modelfiledir = os.path.abspath(os.path.dirname(urdf_path))
        return spec

    @staticmethod
    def _parse_urdf_limits(urdf_path: str) -> dict:
        """Read <joint><limit effort=... velocity=.../></joint> from URDF.

        Returns {joint_name: (effort, velocity)}.  Joints without a <limit>
        tag or without both attributes are absent — caller decides default.
        """
        out: dict = {}
        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if name is None or limit is None:
                continue
            eff = limit.get("effort")
            vel = limit.get("velocity")
            if eff is None or vel is None:
                continue
            out[name] = (float(eff), float(vel))
        return out
