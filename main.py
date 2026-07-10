import math
import sys
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import imgui
from imgui.integrations.pygame import PygameRenderer


# ---------------------------------------------------------------------------
# Terrain data model
# ---------------------------------------------------------------------------

class Terrain:
    """Holds a heightmap grid and the derived render data (vertex positions,
    normals, colors, and the fixed triangle/line index buffers). Editing
    only ever touches the heightmap + the derived arrays; the topology
    (which vertices form which triangles) never changes."""

    def __init__(self, width=50, height=50, scale=5.0, seed=0):
        self.width = width
        self.height = height
        self.scale = scale
        self.seed = seed

        n = width * height
        self.vertices = np.zeros((n, 3), dtype="float32")
        self.normals = np.zeros((n, 3), dtype="float32")
        self.colors = np.zeros((n, 3), dtype="float32")

        # x/z positions never change after creation, only y (height) does
        xs = np.arange(width) - width / 2.0
        zs = np.arange(height) - height / 2.0
        grid_x, grid_z = np.meshgrid(xs, zs)  # shape (height, width)
        self.vertices[:, 0] = grid_x.flatten()
        self.vertices[:, 2] = grid_z.flatten()

        self.tris = self._build_triangles(width, height)
        self.line_indices = self._build_line_indices(width, height)

        self.regenerate(seed)

    # -- topology (fixed) --------------------------------------------------

    @staticmethod
    def _build_triangles(width, height):
        tris = []
        for z in range(height - 1):
            for x in range(width - 1):
                i0 = x + z * width
                i1 = x + (z + 1) * width
                i2 = (x + 1) + z * width
                i3 = (x + 1) + (z + 1) * width
                tris.append((i0, i1, i2))
                tris.append((i2, i1, i3))
        return np.array(tris, dtype="uint32")

    @staticmethod
    def _build_line_indices(width, height):
        lines = []
        for z in range(height - 1):
            for x in range(width - 1):
                i0 = x + z * width
                i1 = x + (z + 1) * width
                i2 = (x + 1) + (z + 1) * width
                lines += [i0, i1, i1, i2, i2, i0]
        return np.array(lines, dtype="uint32")

    # -- generation ----------------------------------------------------------

    def regenerate(self, seed=None):
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        phase_x = rng.uniform(0, 10)
        phase_z = rng.uniform(0, 10)

        width, height, scale = self.width, self.height, self.scale
        xs = (np.arange(width) / scale) + phase_x
        zs = (np.arange(height) / scale) + phase_z
        gx, gz = np.meshgrid(xs, zs)  # (height, width)

        y = (np.sin(gx) * np.cos(gz) * 2.0) + (np.sin(gx * 2) * 0.5)
        self.vertices[:, 1] = y.flatten()
        self.recompute_derived()

    def flatten_all(self, y_value=0.0):
        self.vertices[:, 1] = y_value
        self.recompute_derived()

    # -- derived data (normals + colors), recomputed after any edit --------

    def recompute_derived(self):
        v = self.vertices
        tris = self.tris
        v0, v1, v2 = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        lengths[lengths < 1e-8] = 1.0
        face_normals = face_normals / lengths

        vertex_normals = np.zeros_like(v)
        np.add.at(vertex_normals, tris[:, 0], face_normals)
        np.add.at(vertex_normals, tris[:, 1], face_normals)
        np.add.at(vertex_normals, tris[:, 2], face_normals)
        n_len = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        n_len[n_len < 1e-8] = 1.0
        self.normals = (vertex_normals / n_len).astype("float32")

        y = v[:, 1]
        y_min, y_max = float(y.min()), float(y.max())
        span = max(y_max - y_min, 1e-6)
        t = np.clip((y - y_min) / span, 0.0, 1.0)

        low = np.array([0.45, 0.32, 0.18])
        mid = np.array([0.25, 0.55, 0.2])
        high = np.array([0.9, 0.9, 0.85])

        colors = np.empty((len(y), 3), dtype="float32")
        lower_half = t < 0.5
        k1 = (t[lower_half] / 0.5)[:, None]
        colors[lower_half] = low * (1 - k1) + mid * k1
        k2 = ((t[~lower_half] - 0.5) / 0.5)[:, None]
        colors[~lower_half] = mid * (1 - k2) + high * k2
        self.colors = colors

    # -- grid <-> world helpers ---------------------------------------------

    def world_to_grid(self, wx, wz):
        return wx + self.width / 2.0, wz + self.height / 2.0

    def grid_to_world_index(self, gx, gz):
        return gx, gz

    def height_at(self, gx, gz):
        """Bilinear-interpolated height at fractional grid coords."""
        width, height = self.width, self.height
        gx = np.clip(gx, 0.0, width - 1.0 - 1e-6)
        gz = np.clip(gz, 0.0, height - 1.0 - 1e-6)
        x0, z0 = int(math.floor(gx)), int(math.floor(gz))
        x1, z1 = x0 + 1, z0 + 1
        tx, tz = gx - x0, gz - z0

        h00 = self.vertices[z0 * width + x0, 1]
        h10 = self.vertices[z0 * width + x1, 1]
        h01 = self.vertices[z1 * width + x0, 1]
        h11 = self.vertices[z1 * width + x1, 1]
        h0 = h00 * (1 - tx) + h10 * tx
        h1 = h01 * (1 - tx) + h11 * tx
        return h0 * (1 - tz) + h1 * tz

    def normal_at(self, gx, gz):
        """Bilinear-interpolated vertex normal at fractional grid coords.
        Used by the foliage system to reject placement on steep slopes."""
        width, height = self.width, self.height
        gx = np.clip(gx, 0.0, width - 1.0 - 1e-6)
        gz = np.clip(gz, 0.0, height - 1.0 - 1e-6)
        x0, z0 = int(math.floor(gx)), int(math.floor(gz))
        x1, z1 = x0 + 1, z0 + 1
        tx, tz = gx - x0, gz - z0
        n00 = self.normals[z0 * width + x0]
        n10 = self.normals[z0 * width + x1]
        n01 = self.normals[z1 * width + x0]
        n11 = self.normals[z1 * width + x1]
        n0 = n00 * (1 - tx) + n10 * tx
        n1 = n01 * (1 - tx) + n11 * tx
        n = n0 * (1 - tz) + n1 * tz
        length = np.linalg.norm(n)
        return n / length if length > 1e-8 else np.array([0.0, 1.0, 0.0], dtype="float32")

    def in_bounds_grid(self, gx, gz):
        return 0 <= gx <= self.width - 1 and 0 <= gz <= self.height - 1

    # -- picking (raymarch against the heightmap) ---------------------------

    def raycast(self, origin, direction, max_dist=150.0, step=0.5):
        """March a ray through the heightfield and return the world-space
        hit point, or None. Refines the crossing with linear interpolation
        for a stable pick even with a coarse step size."""
        t = 0.0
        prev_t = 0.0
        prev_diff = None
        while t < max_dist:
            p = origin + direction * t
            gx, gz = self.world_to_grid(p[0], p[2])
            if self.in_bounds_grid(gx, gz):
                terrain_y = self.height_at(gx, gz)
                diff = p[1] - terrain_y
                if prev_diff is not None and prev_diff > 0 >= diff:
                    ratio = prev_diff / (prev_diff - diff)
                    hit_t = prev_t + (t - prev_t) * ratio
                    return origin + direction * hit_t
                prev_diff = diff
            prev_t = t
            t += step
        return None

    # -- brush editing --------------------------------------------------

    def apply_brush(self, hit_point, radius, strength, mode, flatten_height=None):
        """mode: 'raise' | 'lower' | 'flatten' | 'smooth'"""
        hx, hz = hit_point[0], hit_point[2]
        vx = self.vertices[:, 0]
        vz = self.vertices[:, 2]
        dist = np.sqrt((vx - hx) ** 2 + (vz - hz) ** 2)
        mask = dist <= radius
        if not np.any(mask):
            return

        # smooth cosine falloff: 1 at center, 0 at edge
        falloff = 0.5 * (1.0 + np.cos(np.pi * (dist[mask] / radius)))

        if mode == "raise":
            self.vertices[mask, 1] += strength * falloff
        elif mode == "lower":
            self.vertices[mask, 1] -= strength * falloff
        elif mode == "flatten":
            target = flatten_height if flatten_height is not None else hit_point[1]
            cur = self.vertices[mask, 1]
            self.vertices[mask, 1] = cur + (target - cur) * falloff * min(strength, 1.0)
        elif mode == "smooth":
            # blend each affected vertex toward the local neighborhood average
            width = self.width
            idxs = np.nonzero(mask)[0]
            avg = np.empty(len(idxs), dtype="float32")
            for i, idx in enumerate(idxs):
                x = idx % width
                z = idx // width
                x0, x1 = max(0, x - 1), min(width - 1, x + 1)
                z0, z1 = max(0, z - 1), min(self.height - 1, z + 1)
                # gather the 3x3 neighborhood manually (grid isn't a contiguous row slice)
                ys = []
                for zz in range(z0, z1 + 1):
                    row_start = zz * width
                    ys.extend(self.vertices[row_start + x0: row_start + x1 + 1, 1])
                avg[i] = float(np.mean(ys))
            cur = self.vertices[mask, 1]
            self.vertices[mask, 1] = cur + (avg - cur) * strength * falloff
        self.recompute_derived()


# ---------------------------------------------------------------------------
# Foliage system
# ---------------------------------------------------------------------------
#
# Foliage is stored as flat instance arrays (positions/rotations/scales/
# types), each tagged with the fractional grid coordinate it was sampled
# at. That grid coordinate is what lets us "re-glue" an instance to the
# terrain after an edit: we just re-sample height_at()/normal_at() at the
# same (gx, gz) rather than re-scattering everything from scratch.
#
# Foliage is never placed automatically. The scene starts empty and the
# user paints instances on with the "Paint Foliage" tool (or, optionally,
# triggers an explicit one-shot random scatter via the "Scatter Fill"
# button). Terrain edits only re-glue/cull existing foliage; they never
# add new instances on their own.

FOLIAGE_TREE = 0
FOLIAGE_BUSH = 1
FOLIAGE_GRASS = 2
FOLIAGE_LEAVES = 3
FOLIAGE_MUD = 4
FOLIAGE_SAND = 5
FOLIAGE_ROCK = 6
FOLIAGE_FLOWER = 7
FOLIAGE_DEADTREE = 8

# Per-type placement rules: height band the terrain must fall in, minimum
# "up-ness" of the surface normal (1.0 = perfectly flat, lower = steeper
# slopes allowed), the relative chance of picking this type when several
# qualify (used by the "All" mix option), and the random uniform scale
# range applied at render time.
FOLIAGE_RULES = [
    dict(name="tree",      min_h=-0.6, max_h=1.6, min_up=0.85, weight=0.18, scale=(0.8, 1.5)),
    dict(name="bush",      min_h=-0.8, max_h=1.8, min_up=0.72, weight=0.16, scale=(0.5, 0.9)),
    dict(name="grass",     min_h=-0.9, max_h=2.0, min_up=0.55, weight=0.20, scale=(0.6, 1.2)),
    dict(name="leaves",    min_h=-0.7, max_h=1.4, min_up=0.60, weight=0.11, scale=(0.7, 1.3)),
    dict(name="mud",       min_h=-1.2, max_h=-0.1, min_up=0.40, weight=0.08, scale=(0.8, 1.6)),
    dict(name="sand",      min_h=-1.2, max_h=-0.2, min_up=0.50, weight=0.08, scale=(0.9, 1.8)),
    dict(name="rock",      min_h=-1.2, max_h=1.8, min_up=0.35, weight=0.09, scale=(0.5, 1.4)),
    dict(name="flower",    min_h=-0.5, max_h=1.2, min_up=0.70, weight=0.07, scale=(0.6, 1.1)),
    dict(name="dead tree", min_h=-0.4, max_h=1.0, min_up=0.65, weight=0.03, scale=(0.7, 1.3)),
]

FOLIAGE_TYPE_LABELS = [r["name"].capitalize() for r in FOLIAGE_RULES] + ["All (mix)"]
FOLIAGE_ALL = "all"  # sentinel used in place of a type index to mean "mix all types"


class Foliage:
    """Holds instanced props (trees/bushes/grass/leaves/mud/sand/rocks/
    flowers/dead trees) scattered on a Terrain and keeps them glued to it
    as the heightmap is edited."""

    def __init__(self):
        self.grid_pos = np.zeros((0, 2), dtype="float64")
        self.positions = np.zeros((0, 3), dtype="float32")
        self.rotations = np.zeros((0,), dtype="float32")
        self.scales = np.zeros((0,), dtype="float32")
        self.types = np.zeros((0,), dtype="int32")

    def __len__(self):
        return len(self.positions)

    def clear(self):
        self.__init__()

    # -- full (re)generation --------------------------------------------
    # Kept as an explicit, user-triggered convenience (e.g. a "Scatter Fill"
    # button) - it is never called automatically by the app.

    def generate(self, terrain, density=1.0, seed=None):
        """Scatter foliage across the whole terrain from scratch."""
        rng = np.random.default_rng(seed if seed is not None else int(terrain.seed) + 1)
        area = max((terrain.width - 1) * (terrain.height - 1), 1)
        target = max(int(area * 0.35 * density), 0)

        grid_pos, positions, rotations, scales, types = [], [], [], [], []
        attempts = 0
        max_attempts = target * 8 + 300
        while len(positions) < target and attempts < max_attempts:
            attempts += 1
            gx = rng.uniform(0.5, terrain.width - 1.5)
            gz = rng.uniform(0.5, terrain.height - 1.5)
            inst = self._sample(terrain, gx, gz, rng)
            if inst is None:
                continue
            gpos, wpos, rot, scale, ftype = inst
            grid_pos.append(gpos)
            positions.append(wpos)
            rotations.append(rot)
            scales.append(scale)
            types.append(ftype)

        self.grid_pos = np.array(grid_pos, dtype="float64").reshape(-1, 2)
        self.positions = np.array(positions, dtype="float32").reshape(-1, 3)
        self.rotations = np.array(rotations, dtype="float32")
        self.scales = np.array(scales, dtype="float32")
        self.types = np.array(types, dtype="int32")

    def _sample(self, terrain, gx, gz, rng):
        """Try to place one instance at (gx, gz), picking among whichever
        types' placement rules qualify at that spot (weighted). Returns
        None if no type qualifies. Used by generate() and by the "All"
        mix option in the paint brush."""
        if not terrain.in_bounds_grid(gx, gz):
            return None
        h = float(terrain.height_at(gx, gz))
        up_y = float(terrain.normal_at(gx, gz)[1])

        candidates = [r for r in FOLIAGE_RULES if r["min_h"] <= h <= r["max_h"] and up_y >= r["min_up"]]
        if not candidates:
            return None
        weights = np.array([c["weight"] for c in candidates], dtype="float64")
        weights /= weights.sum()
        rule = candidates[int(rng.choice(len(candidates), p=weights))]
        ftype = FOLIAGE_RULES.index(rule)

        wx = gx - terrain.width / 2.0
        wz = gz - terrain.height / 2.0
        wpos = (wx, h, wz)
        rot = float(rng.uniform(0, 2 * math.pi))
        lo, hi = rule["scale"]
        scale = float(rng.uniform(lo, hi))
        return (gx, gz), wpos, rot, scale, ftype

    def _sample_type(self, terrain, gx, gz, ftype, rng):
        """Try to place one instance of a *specific* type at (gx, gz).
        Returns None if that type's placement rule isn't satisfied there
        (e.g. painting sand on a steep cliff)."""
        if not terrain.in_bounds_grid(gx, gz):
            return None
        h = float(terrain.height_at(gx, gz))
        up_y = float(terrain.normal_at(gx, gz)[1])
        rule = FOLIAGE_RULES[ftype]
        if not (rule["min_h"] <= h <= rule["max_h"] and up_y >= rule["min_up"]):
            return None

        wx = gx - terrain.width / 2.0
        wz = gz - terrain.height / 2.0
        wpos = (wx, h, wz)
        rot = float(rng.uniform(0, 2 * math.pi))
        lo, hi = rule["scale"]
        scale = float(rng.uniform(lo, hi))
        return (gx, gz), wpos, rot, scale, ftype

    # -- user-driven painting ---------------------------------------------

    def add_instances(self, terrain, center, radius, count, type_filter, rng,
                       min_spacing=0.4, soft_edge=True):
        """Paint new foliage within `radius` of `center`. `type_filter` is
        either a FOLIAGE_* index (place only that type, skipping spots
        where its rule fails) or FOLIAGE_ALL ("all") to mix every type
        that qualifies at each sampled spot, same as the old auto-scatter.
        `count` candidate points are tried per call; candidates too close
        to an existing (or just-added) instance are skipped so holding the
        mouse down doesn't pile hundreds of instances on the same spot.

        When `soft_edge` is True, candidates are accepted with a cosine
        falloff based on their distance from `center` (dense in the middle,
        thinning out toward the rim), matching the feel of the sculpt
        brush instead of a hard-edged disc of uniform density."""
        if count <= 0:
            return
        new_grid, new_pos, new_rot, new_scale, new_type = [], [], [], [], []
        for _ in range(count):
            angle = rng.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(rng.uniform(0.0, 1.0))

            if soft_edge:
                falloff = 0.5 * (1.0 + math.cos(math.pi * (r / radius)))
                if rng.uniform(0.0, 1.0) > falloff:
                    continue

            wx = center[0] + math.cos(angle) * r
            wz = center[2] + math.sin(angle) * r
            gx, gz = terrain.world_to_grid(wx, wz)
            if not terrain.in_bounds_grid(gx, gz):
                continue

            if type_filter == FOLIAGE_ALL:
                inst = self._sample(terrain, gx, gz, rng)
            else:
                inst = self._sample_type(terrain, gx, gz, type_filter, rng)
            if inst is None:
                continue
            gpos, wpos, rot, scale, ftype = inst

            too_close = False
            if len(self.positions):
                d2 = (self.positions[:, 0] - wpos[0]) ** 2 + (self.positions[:, 2] - wpos[2]) ** 2
                if np.any(d2 < min_spacing * min_spacing):
                    too_close = True
            if not too_close:
                for p in new_pos:
                    if (p[0] - wpos[0]) ** 2 + (p[2] - wpos[2]) ** 2 < min_spacing * min_spacing:
                        too_close = True
                        break
            if too_close:
                continue

            new_grid.append(gpos)
            new_pos.append(wpos)
            new_rot.append(rot)
            new_scale.append(scale)
            new_type.append(ftype)

        if not new_pos:
            return
        self.grid_pos = np.concatenate([self.grid_pos, np.array(new_grid, dtype="float64").reshape(-1, 2)], axis=0)
        self.positions = np.concatenate([self.positions, np.array(new_pos, dtype="float32").reshape(-1, 3)], axis=0)
        self.rotations = np.concatenate([self.rotations, np.array(new_rot, dtype="float32")], axis=0)
        self.scales = np.concatenate([self.scales, np.array(new_scale, dtype="float32")], axis=0)
        self.types = np.concatenate([self.types, np.array(new_type, dtype="int32")], axis=0)

    def erase_instances(self, center, radius):
        """Remove any instances within `radius` of `center` (foliage eraser)."""
        if len(self.positions) == 0:
            return
        dx = self.positions[:, 0] - center[0]
        dz = self.positions[:, 2] - center[2]
        keep = (dx * dx + dz * dz) > (radius * radius)
        if not np.all(keep):
            self.grid_pos = self.grid_pos[keep]
            self.positions = self.positions[keep]
            self.rotations = self.rotations[keep]
            self.scales = self.scales[keep]
            self.types = self.types[keep]

    # -- keeping foliage glued to an edited terrain -----------------------

    def update_region(self, terrain, center, radius):
        """Call after a sculpt brush stroke. Re-samples height/slope for
        every instance inside the affected radius: instances that still
        satisfy their type's rule are re-glued to the new surface height,
        and ones that don't (buried underwater, now on a cliff, etc.) are
        removed."""
        if len(self.positions) == 0:
            return
        cx, cz = center[0], center[2]
        dx = self.positions[:, 0] - cx
        dz = self.positions[:, 2] - cz
        affected = np.nonzero((dx * dx + dz * dz) <= (radius * radius))[0]
        if len(affected) == 0:
            return
        self._reglue(terrain, affected)

    def update_all(self, terrain):
        """Call after a full terrain regeneration/flatten. Re-glues or
        culls every existing instance against the new heightmap, without
        adding any new foliage."""
        if len(self.positions) == 0:
            return
        self._reglue(terrain, np.arange(len(self.positions)))

    def _reglue(self, terrain, indices):
        keep_mask = np.ones(len(self.positions), dtype=bool)
        for i in indices:
            gx, gz = self.grid_pos[i]
            h = float(terrain.height_at(gx, gz))
            up_y = float(terrain.normal_at(gx, gz)[1])
            rule = FOLIAGE_RULES[int(self.types[i])]
            if rule["min_h"] <= h <= rule["max_h"] and up_y >= rule["min_up"]:
                self.positions[i, 1] = h
            else:
                keep_mask[i] = False

        if not np.all(keep_mask):
            self.grid_pos = self.grid_pos[keep_mask]
            self.positions = self.positions[keep_mask]
            self.rotations = self.rotations[keep_mask]
            self.scales = self.scales[keep_mask]
            self.types = self.types[keep_mask]


# -- foliage prototype meshes (built once as OpenGL display lists) --------

def _sphere_triangles(lat_segs, lon_segs):
    """Unit-sphere triangles as (p0, p1, p2) tuples of (x, y, z), y in [0,2]
    (i.e. a sphere of radius 1 centered at y=1, so it rests on y=0)."""
    tris = []

    def sph(theta, phi):
        x = math.sin(theta) * math.cos(phi)
        y = math.cos(theta)
        z = math.sin(theta) * math.sin(phi)
        return (x, y + 1.0, z)

    for i in range(lat_segs):
        theta0 = math.pi * i / lat_segs
        theta1 = math.pi * (i + 1) / lat_segs
        for j in range(lon_segs):
            phi0 = 2 * math.pi * j / lon_segs
            phi1 = 2 * math.pi * (j + 1) / lon_segs
            p00, p01 = sph(theta0, phi0), sph(theta0, phi1)
            p10, p11 = sph(theta1, phi0), sph(theta1, phi1)
            tris.append((p00, p10, p11))
            tris.append((p00, p11, p01))
    return tris


def _build_tree_list():
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)

    # trunk
    trunk_segments = 6
    trunk_h = 0.55
    trunk_r = 0.055
    glColor3f(0.36, 0.24, 0.14)
    glBegin(GL_TRIANGLES)
    for i in range(trunk_segments):
        a0 = 2 * math.pi * i / trunk_segments
        a1 = 2 * math.pi * (i + 1) / trunk_segments
        x0, z0 = math.cos(a0) * trunk_r, math.sin(a0) * trunk_r
        x1, z1 = math.cos(a1) * trunk_r, math.sin(a1) * trunk_r
        glNormal3f(math.cos(a0), 0.0, math.sin(a0))
        glVertex3f(x0, 0.0, z0)
        glVertex3f(x0, trunk_h, z0)
        glNormal3f(math.cos(a1), 0.0, math.sin(a1))
        glVertex3f(x1, 0.0, z1)

        glVertex3f(x1, 0.0, z1)
        glVertex3f(x0, trunk_h, z0)
        glVertex3f(x1, trunk_h, z1)
    glEnd()

    # three stacked cone layers of leaves
    glColor3f(0.15, 0.40, 0.17)
    cone_segments = 8
    layers = [(0.55, trunk_h, 0.65), (0.42, trunk_h + 0.42, 0.55), (0.28, trunk_h + 0.78, 0.42)]
    for radius, base_y, cone_h in layers:
        tip = (0.0, base_y + cone_h, 0.0)
        glBegin(GL_TRIANGLES)
        for i in range(cone_segments):
            a0 = 2 * math.pi * i / cone_segments
            a1 = 2 * math.pi * (i + 1) / cone_segments
            x0, z0 = math.cos(a0) * radius, math.sin(a0) * radius
            x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
            nx = math.cos((a0 + a1) / 2.0)
            nz = math.sin((a0 + a1) / 2.0)
            glNormal3f(nx, 0.5, nz)
            glVertex3f(x0, base_y, z0)
            glVertex3f(x1, base_y, z1)
            glVertex3f(*tip)
        glEnd()
    glEndList()
    return list_id


def _build_bush_list():
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    glColor3f(0.20, 0.44, 0.19)
    radius = 0.28
    glBegin(GL_TRIANGLES)
    for p0, p1, p2 in _sphere_triangles(5, 8):
        for p in (p0, p1, p2):
            nx, ny, nz = p[0], p[1] - 1.0, p[2]
            glNormal3f(nx, ny, nz)
            glVertex3f(p[0] * radius, p[1] * radius, p[2] * radius)
    glEnd()
    glEndList()
    return list_id


def _build_grass_list():
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    glColor3f(0.36, 0.62, 0.24)
    w, h = 0.28, 0.45
    glBegin(GL_TRIANGLES)
    for angle in (0.0, math.pi / 2.0):
        cx, cz = math.cos(angle) * w / 2.0, math.sin(angle) * w / 2.0
        p0 = (-cx, 0.0, -cz)
        p1 = (cx, 0.0, cz)
        p2 = (cx, h, cz)
        p3 = (-cx, h, -cz)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(*p0)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glVertex3f(*p0)
        glVertex3f(*p2)
        glVertex3f(*p3)
    glEnd()
    glEndList()
    return list_id


def _build_leaves_list():
    """A small scattered clump of flat fallen-leaf blades lying on the
    ground - used for the "leaves" foliage type (leaf litter)."""
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    colors = [(0.55, 0.35, 0.12), (0.62, 0.42, 0.10), (0.45, 0.28, 0.08), (0.58, 0.30, 0.09)]
    # fixed jittered offsets baked into the display list (compiled once)
    leaf_positions = [(0.0, 0.0), (0.12, 0.08), (-0.10, 0.10), (0.05, -0.12), (-0.13, -0.05), (0.15, -0.02)]
    y = 0.015
    glBegin(GL_TRIANGLES)
    for idx, (lx, lz) in enumerate(leaf_positions):
        glColor3f(*colors[idx % len(colors)])
        size = 0.10
        p0 = (lx, y, lz - size)
        p1 = (lx + size * 0.6, y, lz)
        p2 = (lx, y, lz + size)
        p3 = (lx - size * 0.6, y, lz)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(*p0)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glVertex3f(*p0)
        glVertex3f(*p2)
        glVertex3f(*p3)
    glEnd()
    glEndList()
    return list_id


def _build_mud_list():
    """A flat, irregular dark puddle/patch decal - used for the "mud"
    foliage type."""
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    glColor3f(0.30, 0.20, 0.12)
    segments = 12
    radius = 0.5
    y = 0.01
    glBegin(GL_TRIANGLES)
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        r0 = radius * (0.85 + 0.15 * math.sin(a0 * 3.0))
        r1 = radius * (0.85 + 0.15 * math.sin(a1 * 3.0))
        x0, z0 = math.cos(a0) * r0, math.sin(a0) * r0
        x1, z1 = math.cos(a1) * r1, math.sin(a1) * r1
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, y, 0.0)
        glVertex3f(x0, y, z0)
        glVertex3f(x1, y, z1)
    glEnd()
    glEndList()
    return list_id


def _build_sand_list():
    """A flat, roughly-circular tan patch decal - used for the "sand"
    foliage type."""
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    glColor3f(0.76, 0.70, 0.50)
    segments = 12
    radius = 0.6
    y = 0.008
    glBegin(GL_TRIANGLES)
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        r0 = radius * (0.9 + 0.1 * math.cos(a0 * 4.0))
        r1 = radius * (0.9 + 0.1 * math.cos(a1 * 4.0))
        x0, z0 = math.cos(a0) * r0, math.sin(a0) * r0
        x1, z1 = math.cos(a1) * r1, math.sin(a1) * r1
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, y, 0.0)
        glVertex3f(x0, y, z0)
        glVertex3f(x1, y, z1)
    glEnd()
    glEndList()
    return list_id


def _build_rock_list():
    """A squashed, low-poly boulder - used for the "rock" foliage type."""
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    glColor3f(0.46, 0.45, 0.43)
    sx, sy, sz = 0.36, 0.24, 0.32  # squash vertically so it sits like a boulder
    glBegin(GL_TRIANGLES)
    for p0, p1, p2 in _sphere_triangles(4, 6):
        for p in (p0, p1, p2):
            nx, ny, nz = p[0], p[1] - 1.0, p[2]
            glNormal3f(nx, ny, nz)
            glVertex3f(p[0] * sx, p[1] * sy, p[2] * sz)
    glEnd()
    # a couple of smaller companion pebbles beside the main boulder
    for ox, oz, s in [(0.32, 0.12, 0.5), (-0.22, -0.28, 0.4)]:
        glBegin(GL_TRIANGLES)
        for p0, p1, p2 in _sphere_triangles(3, 5):
            for p in (p0, p1, p2):
                nx, ny, nz = p[0], p[1] - 1.0, p[2]
                glNormal3f(nx, ny, nz)
                glVertex3f(ox + p[0] * sx * s, p[1] * sy * s, oz + p[2] * sz * s)
        glEnd()
    glEndList()
    return list_id


def _build_flower_list():
    """A thin green stem topped with a small cross of colored petal quads -
    used for the "flower" foliage type."""
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)

    stem_h = 0.22
    stem_w = 0.018
    glColor3f(0.30, 0.55, 0.20)
    glBegin(GL_TRIANGLES)
    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(-stem_w, 0.0, 0.0)
    glVertex3f(stem_w, 0.0, 0.0)
    glVertex3f(stem_w, stem_h, 0.0)
    glVertex3f(-stem_w, 0.0, 0.0)
    glVertex3f(stem_w, stem_h, 0.0)
    glVertex3f(-stem_w, stem_h, 0.0)
    glEnd()

    bloom_colors = [(0.85, 0.25, 0.35), (0.90, 0.55, 0.15), (0.80, 0.30, 0.75)]
    color = bloom_colors[0]
    glColor3f(*color)
    bloom_w = 0.12
    petal_h = 0.05
    glBegin(GL_TRIANGLES)
    for angle in (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0):
        cx, cz = math.cos(angle) * bloom_w / 2.0, math.sin(angle) * bloom_w / 2.0
        p0 = (-cx, stem_h, -cz)
        p1 = (cx, stem_h, cz)
        p2 = (cx, stem_h + petal_h, cz)
        p3 = (-cx, stem_h + petal_h, -cz)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(*p0)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glVertex3f(*p0)
        glVertex3f(*p2)
        glVertex3f(*p3)
    glEnd()
    glEndList()
    return list_id


def _build_deadtree_list():
    """A bare trunk with a few angled branch stubs and no leaves - used for
    the "dead tree" foliage type."""
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)

    trunk_segments = 6
    trunk_h = 0.75
    trunk_r = 0.05
    glColor3f(0.32, 0.27, 0.22)
    glBegin(GL_TRIANGLES)
    for i in range(trunk_segments):
        a0 = 2 * math.pi * i / trunk_segments
        a1 = 2 * math.pi * (i + 1) / trunk_segments
        x0, z0 = math.cos(a0) * trunk_r, math.sin(a0) * trunk_r
        x1, z1 = math.cos(a1) * trunk_r, math.sin(a1) * trunk_r
        glNormal3f(math.cos(a0), 0.0, math.sin(a0))
        glVertex3f(x0, 0.0, z0)
        glVertex3f(x0, trunk_h, z0)
        glNormal3f(math.cos(a1), 0.0, math.sin(a1))
        glVertex3f(x1, 0.0, z1)

        glVertex3f(x1, 0.0, z1)
        glVertex3f(x0, trunk_h, z0)
        glVertex3f(x1, trunk_h, z1)
    glEnd()

    # a few thin branch stubs jutting out at different heights/angles
    branch_specs = [(0.42, 0.35, 0.55), (0.60, -0.40, 0.35), (0.55, 0.10, -0.50), (0.68, -0.15, -0.30)]
    glColor3f(0.30, 0.25, 0.20)
    glBegin(GL_TRIANGLES)
    for base_y, dx, dz in branch_specs:
        length = 0.24
        r = 0.018
        ex = dx * length
        ez = dz * length
        ey = base_y + 0.10
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-r, base_y, 0.0)
        glVertex3f(r, base_y, 0.0)
        glVertex3f(ex, ey, ez)
    glEnd()
    glEndList()
    return list_id


def build_foliage_display_lists():
    """Must be called after the OpenGL context exists."""
    return {
        FOLIAGE_TREE: _build_tree_list(),
        FOLIAGE_BUSH: _build_bush_list(),
        FOLIAGE_GRASS: _build_grass_list(),
        FOLIAGE_LEAVES: _build_leaves_list(),
        FOLIAGE_MUD: _build_mud_list(),
        FOLIAGE_SAND: _build_sand_list(),
        FOLIAGE_ROCK: _build_rock_list(),
        FOLIAGE_FLOWER: _build_flower_list(),
        FOLIAGE_DEADTREE: _build_deadtree_list(),
    }


def draw_foliage(foliage, display_lists):
    n = len(foliage)
    if n == 0:
        return
    glEnable(GL_LIGHTING)
    positions = foliage.positions
    rotations = foliage.rotations
    scales = foliage.scales
    types = foliage.types
    for i in range(n):
        glPushMatrix()
        glTranslatef(float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]))
        glRotatef(math.degrees(float(rotations[i])), 0.0, 1.0, 0.0)
        s = float(scales[i])
        glScalef(s, s, s)
        glCallList(display_lists[int(types[i])])
        glPopMatrix()


# ---------------------------------------------------------------------------
# Rendering (client-side vertex arrays, no shaders needed)
# ---------------------------------------------------------------------------

def draw_terrain_solid(terrain):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, terrain.vertices)
    glNormalPointer(GL_FLOAT, 0, terrain.normals)
    glColorPointer(3, GL_FLOAT, 0, terrain.colors)
    glDrawElements(GL_TRIANGLES, len(terrain.tris) * 3, GL_UNSIGNED_INT, terrain.tris)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)


def draw_terrain_wireframe(terrain):
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, terrain.vertices)
    glDrawElements(GL_LINES, len(terrain.line_indices), GL_UNSIGNED_INT, terrain.line_indices)
    glDisableClientState(GL_VERTEX_ARRAY)


def draw_brush_ring(center, radius, color=(1.0, 1.0, 0.2), segments=48):
    """A flat ring on the XZ plane at the brush's hit point, for feedback.
    Color is used to distinguish tools/modes (sculpt vs foliage paint vs
    foliage erase)."""
    glDisable(GL_LIGHTING)
    glColor3f(*color)
    glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        glVertex3f(center[0] + math.cos(a) * radius, center[1] + 0.05, center[2] + math.sin(a) * radius)
    glEnd()
    glLineWidth(1.0)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    display = (1000, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("Terrain Editor")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, display[0] / display[1], 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 30.0, 10.0, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)
    # foliage instances are drawn with non-uniform world scale via glScalef;
    # GL_NORMALIZE keeps their lighting normals correct regardless of scale.
    glEnable(GL_NORMALIZE)

    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    imgui.create_context()
    impl = PygameRenderer()
    io = imgui.get_io()
    io.display_size = display

    terrain = Terrain(width=50, height=50, scale=5.0, seed=0)

    foliage_display_lists = build_foliage_display_lists()
    foliage = Foliage()  # starts empty - nothing is placed automatically
    rng_paint = np.random.default_rng()
    show_foliage = True

    cam_pos = np.array([0.0, 15.0, -45.0], dtype="float32")
    cam_yaw = 90.0
    cam_pitch = -15.0

    show_solid = True
    show_wireframe = False

    # -- tool selection: sculpt terrain, or paint foliage ------------------
    tool_idx = 0  # 0 = Sculpt Terrain, 1 = Paint Foliage
    tool_labels = ["Sculpt Terrain", "Paint Foliage"]

    brush_mode = 0  # 0=raise 1=lower 2=flatten 3=smooth
    brush_modes = ["Raise", "Lower", "Flatten", "Smooth"]
    brush_radius = 5.0
    brush_strength = 1.0

    foliage_type_idx = len(FOLIAGE_RULES)  # default to "All (mix)"
    foliage_paint_rate = 6.0  # attempted placements per ~16ms frame at full rate
    foliage_erase_mode = False
    foliage_soft_edge = True  # denser near brush center, thinning toward the rim
    foliage_scatter_density = 1.0  # only used by the explicit "Scatter Fill" button

    looking = False        # right mouse button held
    flatten_height = None  # captured at the start of a flatten stroke

    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60)
        for event in pygame.event.get():
            impl.process_event(event)

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == VIDEORESIZE:
                display = (event.w, event.h)
                pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45, display[0] / max(display[1], 1), 0.1, 200.0)
                glMatrixMode(GL_MODELVIEW)
                io.display_size = display
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == K_f:
                    show_wireframe = not show_wireframe
                if event.key == K_g:
                    show_solid = not show_solid
                if event.key == K_t:
                    show_foliage = not show_foliage
                if event.key == K_TAB:
                    tool_idx = 1 - tool_idx
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                looking = True
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                pygame.mouse.get_rel()  # clear jump
            if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                looking = False
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                flatten_height = None  # reset flatten reference for a new stroke

        imgui.new_frame()

        # ---- ImGui panel ----
        imgui.set_next_window_position(15, 15, imgui.ONCE)
        imgui.set_next_window_size(320, 0, imgui.ONCE)
        imgui.begin("Terrain Editor", True)

        imgui.text("Mode: {}".format("Look/Fly (RMB held)" if looking else "Edit (LMB paints)"))
        imgui.separator()

        _, tool_idx = imgui.combo("Tool (Tab)", tool_idx, tool_labels)
        imgui.separator()

        if tool_idx == 0:
            # -- terrain sculpting controls --
            imgui.text("Sculpt")
            clicked, brush_mode = imgui.combo("Brush", brush_mode, brush_modes)
            _, brush_radius = imgui.slider_float("Radius##sculpt", brush_radius, 1.0, 20.0)
            _, brush_strength = imgui.slider_float("Strength", brush_strength, 0.05, 5.0)
        else:
            # -- foliage painting controls --
            imgui.text("Foliage Brush")
            _, foliage_type_idx = imgui.combo("Type", foliage_type_idx, FOLIAGE_TYPE_LABELS)
            _, brush_radius = imgui.slider_float("Radius##foliage", brush_radius, 0.5, 15.0)
            _, foliage_paint_rate = imgui.slider_float("Density", foliage_paint_rate, 0.5, 30.0)
            _, foliage_soft_edge = imgui.checkbox("Soft edge falloff", foliage_soft_edge)
            _, foliage_erase_mode = imgui.checkbox("Erase mode", foliage_erase_mode)
            if imgui.button("Clear all foliage"):
                foliage.clear()

        imgui.separator()
        _, show_solid = imgui.checkbox("Show solid faces (G)", show_solid)
        _, show_wireframe = imgui.checkbox("Show wireframe (F)", show_wireframe)
        _, show_foliage = imgui.checkbox("Show foliage (T)", show_foliage)
        imgui.text("Foliage instances: {}".format(len(foliage)))

        imgui.separator()
        if imgui.button("Regenerate terrain"):
            terrain.regenerate(seed=int(np.random.randint(0, 1_000_000)))
            foliage.update_all(terrain)  # re-glue/cull existing foliage, add nothing new
        imgui.same_line()
        if imgui.button("Flatten to 0"):
            terrain.flatten_all(0.0)
            foliage.update_all(terrain)

        if imgui.tree_node("Advanced: random scatter fill"):
            imgui.text_wrapped(
                "Optional one-shot random fill, only runs when you press the "
                "button below - foliage is never added automatically."
            )
            _, foliage_scatter_density = imgui.slider_float("Fill density", foliage_scatter_density, 0.1, 3.0)
            if imgui.button("Scatter Fill (random)"):
                foliage.generate(terrain, density=foliage_scatter_density,
                                  seed=int(np.random.randint(0, 1_000_000)))
            imgui.tree_pop()

        imgui.separator()
        imgui.text_wrapped(
            "RMB: look + WASD fly | Q/E or Shift/Space: down/up | "
            "Tab: switch tool | LMB: paint with the active tool | T: toggle foliage"
        )
        imgui.end()

        # ---- camera look (only while RMB held) ----
        if looking:
            mouse_x, mouse_y = pygame.mouse.get_rel()
            cam_yaw += mouse_x * 0.15
            cam_pitch -= mouse_y * 0.15
            cam_pitch = max(-89.0, min(89.0, cam_pitch))
        else:
            pygame.mouse.get_rel()  # drain relative motion so it doesn't jump on next RMB press

        front = np.array(
            [
                math.cos(math.radians(cam_yaw)) * math.cos(math.radians(cam_pitch)),
                math.sin(math.radians(cam_pitch)),
                math.sin(math.radians(cam_yaw)) * math.cos(math.radians(cam_pitch)),
            ]
        )
        front = front / np.linalg.norm(front)
        up = np.array([0.0, 1.0, 0.0], dtype="float32")
        right = np.cross(front, up)
        right = right / np.linalg.norm(right)

        # ---- movement (blocked while typing/hovering imgui widgets) ----
        if not io.want_capture_keyboard:
            keys = pygame.key.get_pressed()
            speed = 0.02 * dt
            if keys[K_w]:
                cam_pos += front * speed
            if keys[K_s]:
                cam_pos -= front * speed
            if keys[K_a]:
                cam_pos -= right * speed
            if keys[K_d]:
                cam_pos += right * speed
            if keys[K_q] or keys[K_LSHIFT]:
                cam_pos -= up * speed
            if keys[K_e] or keys[K_SPACE]:
                cam_pos += up * speed

        # ---- brush picking + painting (blocked while over imgui panel) ----
        brush_hit = None
        if not io.want_capture_mouse:
            mx, my = pygame.mouse.get_pos()
            # unproject the mouse position into a world-space ray
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            gl_y = viewport[3] - my
            near = np.array(gluUnProject(mx, gl_y, 0.0, modelview, projection, viewport))
            far = np.array(gluUnProject(mx, gl_y, 1.0, modelview, projection, viewport))
            ray_dir = far - near
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            brush_hit = terrain.raycast(near.astype("float32"), ray_dir.astype("float32"))

            mouse_buttons = pygame.mouse.get_pressed()
            if brush_hit is not None and mouse_buttons[0] and not looking:
                if tool_idx == 0:
                    # -- terrain sculpting --
                    mode_name = brush_modes[brush_mode].lower()
                    if mode_name == "flatten" and flatten_height is None:
                        flatten_height = float(brush_hit[1])
                    terrain.apply_brush(
                        brush_hit, brush_radius, brush_strength * (dt / 16.0),
                        mode_name, flatten_height=flatten_height,
                    )
                    # keep foliage glued to (or removed from) the freshly edited area
                    foliage.update_region(terrain, brush_hit, brush_radius)
                else:
                    # -- foliage painting --
                    if foliage_erase_mode:
                        foliage.erase_instances(brush_hit, brush_radius)
                    else:
                        selected = FOLIAGE_ALL if foliage_type_idx >= len(FOLIAGE_RULES) else foliage_type_idx
                        rate = foliage_paint_rate * (dt / 16.0)
                        count = int(rate)
                        if rng_paint.random() < (rate - count):
                            count += 1
                        foliage.add_instances(terrain, brush_hit, brush_radius, count, selected, rng_paint,
                                               soft_edge=foliage_soft_edge)

        # ---- render scene ----
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            cam_pos[0], cam_pos[1], cam_pos[2],
            cam_pos[0] + front[0], cam_pos[1] + front[1], cam_pos[2] + front[2],
            0.0, 1.0, 0.0,
        )

        if show_solid:
            glEnable(GL_LIGHTING)
            draw_terrain_solid(terrain)
        if show_wireframe:
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)
            draw_terrain_wireframe(terrain)
        if show_foliage:
            draw_foliage(foliage, foliage_display_lists)
        if brush_hit is not None:
            if tool_idx == 0:
                ring_color = (1.0, 1.0, 0.2)       # yellow: sculpt
            elif foliage_erase_mode:
                ring_color = (1.0, 0.25, 0.25)     # red: foliage erase
            else:
                ring_color = (0.25, 1.0, 0.35)     # green: foliage paint
            draw_brush_ring(brush_hit, brush_radius, ring_color)
            glEnable(GL_LIGHTING)

        imgui.render()
        impl.render(imgui.get_draw_data())

        pygame.display.flip()


if __name__ == "__main__":
    main()