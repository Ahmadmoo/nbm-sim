"""Scene definitions (spec §4). Geometry is defined once here and reused for rendering,
colliders and the evaluation mesh, so all three share the same transforms."""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

from .motion import Box, Workspace


@dataclass(frozen=True)
class BoxSpec:
    name: str
    center: tuple
    size: tuple
    color: tuple = (0.5, 0.5, 0.5)
    textured: bool = False
    semantic: str = "background"
    collider: bool = True


@dataclass
class SceneSpec:
    scene_id: str
    boxes: list
    target_name: str
    lights: dict = field(default_factory=dict)
    exclude_below_z: float | None = None   # evaluation region lower bound (removes the table top)
    default_camera_position: tuple = (0.65, -0.65, 1.10)

    @property
    def target(self) -> BoxSpec:
        return next(b for b in self.boxes if b.name == self.target_name)

    @property
    def target_center(self):
        return np.asarray(self.target.center, dtype=np.float64)

    def colliders(self):
        return [Box(b.name, b.center, b.size) for b in self.boxes if b.collider]

    def workspace(self, cfg):
        c = self.target_center
        return Workspace((float(c[0]), float(c[1])), tuple(cfg.horizontal_workspace_radius), tuple(cfg.camera_height))

    def task_region(self, margin=0.02):
        t = self.target
        lo = np.asarray(t.center) - np.asarray(t.size) / 2 - margin
        hi = np.asarray(t.center) + np.asarray(t.size) / 2 + margin
        if self.exclude_below_z is not None:
            lo[2] = max(lo[2], self.exclude_below_z)
        return lo, hi


def textured_cube_scene():
    table_z = 0.80
    side = 0.20
    legs = [BoxSpec(f"table_leg_{i}", (sx * 0.45, sy * 0.45, (table_z - 0.04) / 2), (0.05, 0.05, table_z - 0.04),
                    color=(0.35, 0.28, 0.2), semantic="table")
            for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])]
    return SceneSpec(
        scene_id="textured_cube",
        boxes=[BoxSpec("floor", (0.0, 0.0, -0.01), (10.0, 10.0, 0.02), color=(0.45, 0.45, 0.45), semantic="floor"),
               BoxSpec("table_top", (0.0, 0.0, table_z - 0.02), (1.0, 1.0, 0.04), color=(0.55, 0.45, 0.35),
                       semantic="table"),
               *legs,
               BoxSpec("target", (0.0, 0.0, table_z + side / 2), (side, side, side), textured=True,
                       semantic="target")],
        target_name="target",
        lights=dict(dome_intensity=600.0, distant_intensity=2000.0, distant_rpy_deg=(35.0, 20.0, 0.0)),
        exclude_below_z=table_z + 0.005,
    )


def textured_plane_scene():
    """Fronto-parallel textured panel (front face at world y = 1.0) with four thin landmark squares whose
    front faces sit at y = 0.999 (analytic intrinsics, depth and axis-sign checks)."""
    lm = [BoxSpec(f"landmark_{i}", (x, 0.9995, z), (0.03, 0.001, 0.03), color=(1.0, 0.1, 0.1),
                  semantic=f"landmark_{i}")
          for i, (x, z) in enumerate([(-0.25, 1.40), (0.25, 1.40), (-0.25, 1.00), (0.30, 1.05)])]
    return SceneSpec(
        scene_id="textured_plane",
        boxes=[BoxSpec("floor", (0.0, 0.0, -0.01), (10.0, 10.0, 0.02), color=(0.45, 0.45, 0.45), semantic="floor"),
               BoxSpec("target", (0.0, 1.005, 1.2), (2.0, 0.01, 2.0), textured=True, semantic="target"),
               *lm],
        target_name="target",
        lights=dict(dome_intensity=900.0, distant_intensity=1000.0, distant_rpy_deg=(0.0, 0.0, 0.0)),
        default_camera_position=(0.0, 0.0, 1.2),
    )


SCENES = {"textured_cube": textured_cube_scene, "textured_plane": textured_plane_scene}


def get_scene(scene_id):
    if scene_id not in SCENES:
        raise KeyError(f"unknown scene_id {scene_id!r}; available: {sorted(SCENES)}")
    return SCENES[scene_id]()


# ---------- evaluation geometry ----------

_FACES = [  # (normal axis, sign) in the order +x -x +y -y +z -z
    (0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]


def box_mesh(center, size):
    """Quad-per-face box mesh: vertices (24,3), faces (6,4), uv (24,2) into a 3x2 atlas."""
    c, h = np.asarray(center, float), np.asarray(size, float) / 2
    verts, faces, uvs = [], [], []
    for fi, (ax, s) in enumerate(_FACES):
        u_ax, v_ax = [a for a in range(3) if a != ax]
        corners = []
        for cu, cv in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            p = np.zeros(3)
            p[ax], p[u_ax], p[v_ax] = s, cu, cv
            corners.append(c + p * h)
        if s * (1, -1, 1)[ax] < 0:   # keep counter-clockwise winding seen from outside
            corners = corners[::-1]
        tile_u, tile_v = fi % 3, fi // 3
        base = len(verts)
        for k, p in enumerate(corners):
            verts.append(p)
            cu, cv = [(0, 0), (1, 0), (1, 1), (0, 1)][k]
            uvs.append(((tile_u + 0.02 + 0.96 * cu) / 3.0, (tile_v + 0.02 + 0.96 * cv) / 2.0))
        faces.append([base, base + 1, base + 2, base + 3])
    return np.array(verts), np.array(faces), np.array(uvs)


def sample_box_surface(center, size, spacing=0.001, include_bottom=True):
    """Regular grid samples on the box faces (deterministic). Returns points (N,3) and face ids (N,)."""
    c, s = np.asarray(center, float), np.asarray(size, float)
    pts, ids = [], []
    for fi, (ax, sg) in enumerate(_FACES):
        if not include_bottom and ax == 2 and sg < 0:
            continue
        u_ax, v_ax = [a for a in range(3) if a != ax]
        nu, nv = max(2, int(round(s[u_ax] / spacing)) + 1), max(2, int(round(s[v_ax] / spacing)) + 1)
        uu, vv = np.meshgrid(np.linspace(-0.5, 0.5, nu), np.linspace(-0.5, 0.5, nv), indexing="ij")
        p = np.zeros((uu.size, 3))
        p[:, ax] = sg * 0.5 * s[ax]
        p[:, u_ax] = uu.ravel() * s[u_ax]
        p[:, v_ax] = vv.ravel() * s[v_ax]
        pts.append(c + p)
        ids.append(np.full(uu.size, fi))
    return np.concatenate(pts), np.concatenate(ids)


def evaluation_geometry(spec: SceneSpec, spacing=0.001):
    """Full-target and accessible-surface reference samples. Accessibility is fixed offline:
    a face resting on a support (the cube bottom on the table) is excluded."""
    t = spec.target
    full, _ = sample_box_surface(t.center, t.size, spacing, include_bottom=True)
    acc, _ = sample_box_surface(t.center, t.size, spacing, include_bottom=spec.exclude_below_z is None)
    v, f, _ = box_mesh(t.center, t.size)
    return dict(vertices=v, faces=f, full_surface=full, accessible_surface=acc, spacing=spacing)


# ---------- texture and USD asset generation ----------

def make_texture(seed, size=(1024, 1536)):
    """Deterministic multi-scale smooth-noise RGB atlas (uint8, H x W x 3)."""
    rng = np.random.default_rng(seed)
    H, W = size
    img = np.zeros((H, W, 3))
    for cell, amp in [(96, 1.0), (32, 0.6), (12, 0.35), (5, 0.2)]:
        g = rng.random((H // cell + 2, W // cell + 2, 3))
        yy, xx = np.arange(H) / cell, np.arange(W) / cell
        y0, x0 = yy.astype(int), xx.astype(int)
        fy, fx = (yy - y0)[:, None, None], (xx - x0)[None, :, None]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a, b = g[y0][:, x0], g[y0][:, x0 + 1]
        c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
        img += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
    img = (img - img.min()) / (img.max() - img.min())
    return (np.clip(0.08 + 0.84 * img, 0, 1) * 255).astype(np.uint8)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_textured_box_usd(path, size, texture_png):
    """USD asset: one quad mesh with a 3x2 UV atlas and a UsdPreviewSurface material. Requires ``pxr``."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    v, f, uv = box_mesh((0.0, 0.0, 0.0), size)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in v]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(f)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(f.ravel().tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    normals = []
    for face in f:
        p0, p1, p2 = v[face[0]], v[face[1]], v[face[2]]
        n = np.cross(p1 - p0, p2 - p0)
        normals += [n / np.linalg.norm(n)] * 4
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in uv]))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*(-np.asarray(size) / 2)), Gf.Vec3f(*(np.asarray(size) / 2))]))

    mat = UsdShade.Material.Define(stage, "/Asset/Looks/Mat")
    sh = UsdShade.Shader.Define(stage, "/Asset/Looks/Mat/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, "/Asset/Looks/Mat/StReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, "/Asset/Looks/Mat/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(pathlib.Path(texture_png).name))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()


def build_assets(spec: SceneSpec, asset_dir, texture_seed):
    """Write texture + USD for every textured box; return the asset table (spec §4.2)."""
    from PIL import Image

    d = pathlib.Path(asset_dir) / spec.scene_id
    d.mkdir(parents=True, exist_ok=True)
    table = []
    for i, b in enumerate(spec.boxes):
        if not b.textured:
            continue
        png = d / f"{b.name}_texture_seed{texture_seed + i}.png"
        Image.fromarray(make_texture(texture_seed + i)).save(png)
        usd = d / f"{b.name}.usda"
        write_textured_box_usd(usd, b.size, png)
        table.append(dict(id=b.name, path=str(usd.resolve()), sha256=sha256(usd), texture=str(png.resolve()),
                          texture_sha256=sha256(png), units="m", scale=[1, 1, 1], translation=list(b.center),
                          orientation_wxyz=[1, 0, 0, 0], semantic=b.semantic,
                          collision="analytic axis-aligned box", source="procedural (nbm_sim.scene)",
                          license="generated", conversion="none"))
    (d / "assets.json").write_text(json.dumps(table, indent=2))
    return table


def spawn_scene(spec: SceneSpec, asset_table, quat_order):
    """Spawn boxes and lights into the current Isaac stage (Isaac Lab spawners)."""
    import isaaclab.sim as sim_utils

    from .geometry import quat_to_order, rotmat_to_quat_wxyz, so3_exp

    assets = {a["id"]: a for a in asset_table}
    for b in spec.boxes:
        tags = [("class", b.semantic)]
        if b.textured:
            cfg = sim_utils.UsdFileCfg(usd_path=assets[b.name]["path"], semantic_tags=tags)
        else:
            cfg = sim_utils.CuboidCfg(size=tuple(b.size), semantic_tags=tags,
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(b.color),
                                                                                  roughness=0.9))
        cfg.func(f"/World/Scene/{b.name}", cfg, translation=tuple(b.center))
    L = spec.lights
    dome = sim_utils.DomeLightCfg(intensity=L["dome_intensity"], color=(1.0, 1.0, 1.0))
    dome.func("/World/Lights/Dome", dome)
    r, p, y = np.radians(L["distant_rpy_deg"])
    R = so3_exp([0, 0, y]) @ so3_exp([0, p, 0]) @ so3_exp([r, 0, 0])
    sun = sim_utils.DistantLightCfg(intensity=L["distant_intensity"], angle=0.5)
    sun.func("/World/Lights/Distant", sun, orientation=tuple(quat_to_order(rotmat_to_quat_wxyz(R), quat_order).tolist()))
