import math
import os
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
# Terrain texture painting (Unity-style splat mapping)
# ---------------------------------------------------------------------------
#
# Each vertex carries a 4-component weight (grass/dirt/rock/sand). The
# fragment shader normalizes these and blends the four tiled textures
# accordingly, exactly like Unity's terrain "Paint Texture" tool. Painting
# nudges the weights at the brushed vertices toward the selected layer
# (with the others shrinking proportionally) using the same cosine falloff
# as the sculpt brush, so strokes feel consistent across tools.

class TerrainTextures:
    LAYER_NAMES = ["Grass", "Dirt", "Rock", "Sand"]

    def __init__(self, terrain, tile_scale=6.0):
        self.tile_scale = tile_scale
        self.weights = np.zeros((len(terrain.vertices), 4), dtype="float32")
        self.initialize_from_height_slope(terrain)

    def initialize_from_height_slope(self, terrain):
        """Seed sensible starting weights so the terrain looks reasonable
        before any manual painting: grass on mid-height gentle slopes,
        rock on steep slopes, sand in low areas, dirt filling the rest."""
        y = terrain.vertices[:, 1]
        up = np.clip(terrain.normals[:, 1], 0.0, 1.0)
        y_min, y_max = float(y.min()), float(y.max())
        span = max(y_max - y_min, 1e-6)
        t = np.clip((y - y_min) / span, 0.0, 1.0)

        grass = np.clip(1.0 - np.abs(t - 0.45) * 2.2, 0.0, 1.0) * up
        sand = np.clip((0.25 - t) * 4.0, 0.0, 1.0)
        rock = np.clip((1.0 - up) * 2.0, 0.0, 1.0)
        dirt = np.clip(1.0 - grass - sand - rock, 0.05, 1.0)

        w = np.stack([grass, dirt, rock, sand], axis=1)
        wsum = w.sum(axis=1, keepdims=True)
        wsum[wsum < 1e-6] = 1.0
        self.weights = (w / wsum).astype("float32")

    def paint(self, terrain, hit_point, radius, strength, layer_idx):
        hx, hz = hit_point[0], hit_point[2]
        vx = terrain.vertices[:, 0]
        vz = terrain.vertices[:, 2]
        dist = np.sqrt((vx - hx) ** 2 + (vz - hz) ** 2)
        mask = dist <= radius
        if not np.any(mask):
            return

        falloff = 0.5 * (1.0 + np.cos(np.pi * (dist[mask] / radius)))
        amount = np.clip(strength * falloff, 0.0, 1.0)[:, None]

        target = np.zeros(4, dtype="float32")
        target[layer_idx] = 1.0

        w = self.weights[mask]
        self.weights[mask] = w + (target[None, :] - w) * amount


# ---------------------------------------------------------------------------
# Procedural textures (no external image assets required)
# ---------------------------------------------------------------------------

def _value_noise(size, cell, rng):
    """Smooth value noise: a coarse random grid, bilinearly upsampled."""
    grid_n = max(size // cell, 2)
    grid = rng.random((grid_n + 1, grid_n + 1)).astype("float32")
    ys = np.linspace(0, grid_n - 1e-6, size)
    xs = np.linspace(0, grid_n - 1e-6, size)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    y0 = gy.astype(int)
    x0 = gx.astype(int)
    y1 = np.clip(y0 + 1, 0, grid_n)
    x1 = np.clip(x0 + 1, 0, grid_n)
    ty = (gy - y0)[..., None]
    tx = (gx - x0)[..., None]
    v00 = grid[y0, x0][..., None]
    v10 = grid[y0, x1][..., None]
    v01 = grid[y1, x0][..., None]
    v11 = grid[y1, x1][..., None]
    v0 = v00 * (1 - tx) + v10 * tx
    v1 = v01 * (1 - tx) + v11 * tx
    return (v0 * (1 - ty) + v1 * ty)[..., 0]


def generate_texture(base_color, size=256, seed=0, streaks=False, speckle=False):
    """Builds a tileable-ish procedural albedo texture by layering value
    noise at a few octaves on top of a base color."""
    rng = np.random.default_rng(seed)
    n1 = _value_noise(size, 6, rng)
    n2 = _value_noise(size, 20, rng)
    n3 = _value_noise(size, 56, rng)
    noise = n1 * 0.5 + n2 * 0.3 + n3 * 0.2
    noise = (noise - noise.min()) / max(noise.max() - noise.min(), 1e-6)

    base = np.array(base_color, dtype="float32")
    variation = (noise[..., None] - 0.5) * 0.35
    img = np.clip(base[None, None, :] + variation, 0.0, 1.0)

    if streaks:
        xs = np.linspace(0, 30.0, size)
        stripe = (np.sin(xs) * 0.04)[None, :, None]
        img = np.clip(img + stripe, 0.0, 1.0)

    if speckle:
        speck = (rng.random((size, size)) > 0.985)[..., None]
        img = np.where(speck, np.clip(img + 0.25, 0.0, 1.0), img)

    return (img * 255).astype("uint8")


def create_gl_texture(image_rgb_uint8):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    h, w, _ = image_rgb_uint8.shape
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 np.ascontiguousarray(image_rgb_uint8))
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def build_terrain_textures():
    """Must be called after the OpenGL context exists."""
    return {
        "grass": create_gl_texture(generate_texture((0.20, 0.42, 0.16), seed=1)),
        "dirt": create_gl_texture(generate_texture((0.36, 0.24, 0.14), seed=2, speckle=True)),
        "rock": create_gl_texture(generate_texture((0.42, 0.41, 0.40), seed=3, streaks=True)),
        "sand": create_gl_texture(generate_texture((0.76, 0.69, 0.49), seed=4)),
    }


# ---------------------------------------------------------------------------
# Terrain shader (4-layer splat blend + per-pixel lighting + distance fog)
# ---------------------------------------------------------------------------

TERRAIN_VERT_SRC = """
#version 120
attribute vec4 aWeights;
varying vec3 vNormal;
varying vec3 vWorldPos;
varying vec4 vWeights;

void main() {
    vWorldPos = gl_Vertex.xyz;   // terrain has no model transform: object space == world space
    vNormal = gl_Normal;
    vWeights = aWeights;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""

TERRAIN_FRAG_SRC = """
#version 120
uniform sampler2D texGrass;
uniform sampler2D texDirt;
uniform sampler2D texRock;
uniform sampler2D texSand;
uniform float texScale;
uniform vec3 lightDir;
uniform vec3 camPos;
uniform vec3 fogColor;
uniform float fogDensity;

varying vec3 vNormal;
varying vec3 vWorldPos;
varying vec4 vWeights;

void main() {
    vec2 uv = vWorldPos.xz / texScale;
    vec3 grass = texture2D(texGrass, uv).rgb;
    vec3 dirt  = texture2D(texDirt,  uv).rgb;
    vec3 rock  = texture2D(texRock,  uv).rgb;
    vec3 sand  = texture2D(texSand,  uv).rgb;

    vec4 w = max(vWeights, 0.0);
    float wsum = w.x + w.y + w.z + w.w;
    w = wsum > 0.0001 ? w / wsum : vec4(0.25, 0.25, 0.25, 0.25);

    vec3 albedo = grass * w.x + dirt * w.y + rock * w.z + sand * w.w;

    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightDir);
    float diff = max(dot(N, L), 0.0);

    // cheap ambient-occlusion proxy: steeper slopes read a touch darker
    float ao = clamp(0.55 + 0.45 * N.y, 0.0, 1.0);

    // subtle sheen on rock/sand fraction to sell a "wet/mineral" look
    float sheen = (w.z + w.w * 0.5) * pow(diff, 2.0) * 0.15;

    vec3 color = albedo * (0.32 * ao + 0.85 * diff) + sheen;

    float dist = length(vWorldPos - camPos);
    float fog = clamp(exp(-fogDensity * fogDensity * dist * dist), 0.0, 1.0);
    color = mix(fogColor, color, fog);

    gl_FragColor = vec4(color, 1.0);
}
"""


def compile_shader(src, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader


def build_shader_program(vert_src, frag_src):
    vs = compile_shader(vert_src, GL_VERTEX_SHADER)
    fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


def draw_terrain_solid_textured(terrain, tex_ids, weights, program, uniforms,
                                 cam_pos, tile_scale, light_dir, fog_color, fog_density):
    glUseProgram(program)
    glUniform1i(uniforms["texGrass"], 0)
    glUniform1i(uniforms["texDirt"], 1)
    glUniform1i(uniforms["texRock"], 2)
    glUniform1i(uniforms["texSand"], 3)
    glUniform1f(uniforms["texScale"], tile_scale)
    glUniform3f(uniforms["lightDir"], *light_dir)
    glUniform3f(uniforms["camPos"], *cam_pos)
    glUniform3f(uniforms["fogColor"], *fog_color)
    glUniform1f(uniforms["fogDensity"], fog_density)

    for i, key in enumerate(("grass", "dirt", "rock", "sand")):
        glActiveTexture(GL_TEXTURE0 + i)
        glBindTexture(GL_TEXTURE_2D, tex_ids[key])

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, terrain.vertices)
    glNormalPointer(GL_FLOAT, 0, terrain.normals)

    loc = uniforms["aWeights"]
    glEnableVertexAttribArray(loc)
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, weights)

    glDrawElements(GL_TRIANGLES, len(terrain.tris) * 3, GL_UNSIGNED_INT, terrain.tris)

    glDisableVertexAttribArray(loc)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glActiveTexture(GL_TEXTURE0)
    glUseProgram(0)


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
# Custom foliage import (Sketchfab downloads / your own Blender exports)
# ---------------------------------------------------------------------------
#
# Anyone can drop a .obj model into the "Import custom foliage" panel and
# it becomes a brand-new paintable foliage type alongside the built-in
# tree/bush/grass/etc, with its own placement rules (height band, min
# slope) and scale range set right there in the UI.
#
# Why .obj: it's the one format every relevant tool agrees on -
# Sketchfab offers an "Original Format" / OBJ download for CC-licensed
# models, and Blender exports it natively (File > Export > Wavefront
# (.obj)). If a model only comes as .fbx/.gltf/.glb, open it in Blender
# once and re-export as .obj - free, and it keeps this file dependency-free
# (no fbx/gltf parser needed).

def load_obj_mesh(path):
    """Minimal Wavefront OBJ parser. Returns (verts, norms, faces) where
    verts/norms are lists of (x, y, z) tuples and faces is a list of
    faces, each a list of (vertex_index, normal_index_or_None). Polygons
    with more than 3 vertices are left as-is here and fan-triangulated by
    the caller. UVs and materials are intentionally ignored - imported
    foliage renders as a flat tinted color, same as the built-in props."""
    verts, norms, faces = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vn" and len(parts) >= 4:
                norms.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "f":
                face = []
                for token in parts[1:]:
                    idxs = token.split("/")
                    vi = int(idxs[0])
                    vi = vi - 1 if vi > 0 else len(verts) + vi
                    ni = None
                    if len(idxs) >= 3 and idxs[2]:
                        ni = int(idxs[2])
                        ni = ni - 1 if ni > 0 else len(norms) + ni
                    face.append((vi, ni))
                if len(face) >= 3:
                    faces.append(face)
    if not verts:
        raise ValueError("No vertices found - is this a valid .obj file?")
    if not faces:
        raise ValueError("No faces found - the .obj has points but no geometry")
    return verts, norms, faces


def build_display_list_from_obj(path, target_height=1.0, color=(0.35, 0.45, 0.22)):
    """Loads an .obj, recenters it on X/Z, drops it onto Y=0, uniformly
    rescales it to `target_height` tall, and compiles it into a GL display
    list exactly like the built-in _build_*_list() functions - so it can
    be dropped straight into the same foliage_display_lists dict and
    drawn/instanced identically to trees/bushes/rocks/etc."""
    verts, norms, faces = load_obj_mesh(path)
    v_arr = np.array(verts, dtype="float64")

    min_y, max_y = float(v_arr[:, 1].min()), float(v_arr[:, 1].max())
    span = max(max_y - min_y, 1e-6)
    scale = target_height / span
    cx = (float(v_arr[:, 0].min()) + float(v_arr[:, 0].max())) / 2.0
    cz = (float(v_arr[:, 2].min()) + float(v_arr[:, 2].max())) / 2.0

    def xform(v):
        return ((v[0] - cx) * scale, (v[1] - min_y) * scale, (v[2] - cz) * scale)

    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    glColor3f(*color)
    glBegin(GL_TRIANGLES)
    for face in faces:
        # fan-triangulate n-gons
        for i in range(1, len(face) - 1):
            tri = (face[0], face[i], face[i + 1])
            pts = [xform(verts[vi]) for vi, _ in tri]
            if all(ni is not None for _, ni in tri):
                for (vi, ni), p in zip(tri, pts):
                    n = norms[ni]
                    glNormal3f(n[0], n[1], n[2])
                    glVertex3f(*p)
            else:
                # no vertex normals in the file - use the flat face normal
                p0, p1, p2 = (np.array(p) for p in pts)
                n = np.cross(p1 - p0, p2 - p0)
                nl = np.linalg.norm(n)
                n = (n / nl) if nl > 1e-8 else np.array([0.0, 1.0, 0.0])
                for p in pts:
                    glNormal3f(float(n[0]), float(n[1]), float(n[2]))
                    glVertex3f(*p)
    glEnd()
    glEndList()
    return list_id


def register_custom_foliage(display_lists, name, model_path, min_h, max_h, min_up,
                             scale_range, color=(0.35, 0.45, 0.22), target_height=1.0):
    """Loads `model_path` as a new foliage type and appends it to the
    global FOLIAGE_RULES / FOLIAGE_TYPE_LABELS so it shows up in the
    "Type" dropdown of the foliage brush right alongside the built-ins.
    Returns the new type's index (to select it automatically)."""
    list_id = build_display_list_from_obj(model_path, target_height=target_height, color=color)
    new_idx = len(FOLIAGE_RULES)
    display_lists[new_idx] = list_id
    FOLIAGE_RULES.append(dict(
        name=name, min_h=min_h, max_h=max_h, min_up=min_up, weight=0.1, scale=scale_range,
    ))
    # mutate in place (slice assignment) so every existing reference to
    # this list - e.g. the imgui.combo() call in main() - sees the update
    FOLIAGE_TYPE_LABELS[:] = [r["name"].capitalize() for r in FOLIAGE_RULES] + ["All (mix)"]
    return new_idx


def browse_for_model_file():
    """Tries to open a native "Open File" dialog via Tkinter. Returns
    (True, path) if the dialog opened normally (path is "" if the person
    cancelled it), or (False, "") if Tkinter isn't available/working on
    this system - in which case the caller should fall back to the
    built-in in-app browser (open_inapp_browser / draw_inapp_browser
    below), which has no dependency on Tkinter at all."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a foliage model",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")],
        )
        root.destroy()
        return True, path
    except Exception as e:
        print("Native file dialog unavailable ({}); using the in-app browser instead.".format(e))
        return False, ""


# -- in-app (Tkinter-free) file browser, used automatically whenever the ---
# -- native dialog above isn't available (e.g. Tkinter not installed) -----

def list_dir_entries(directory, extension=".obj"):
    """Returns (subdirs, matching_files), both sorted, for `directory`.
    Returns ([], []) if the directory can't be read (bad path, permissions,
    a drive that no longer exists, etc.) so the browser can show an
    error instead of crashing."""
    try:
        entries = os.listdir(directory)
    except Exception:
        return [], []
    subdirs = sorted(
        e for e in entries
        if not e.startswith(".") and os.path.isdir(os.path.join(directory, e))
    )
    files = sorted(e for e in entries if e.lower().endswith(extension))
    return subdirs, files


def default_browse_start_dir():
    """Starts in ~/Downloads when it exists (where a Sketchfab .zip is
    usually extracted to), otherwise the home directory."""
    home = os.path.expanduser("~")
    downloads = os.path.join(home, "Downloads")
    return downloads if os.path.isdir(downloads) else home


def draw_inapp_browser(browse_dir):
    """Renders the fallback in-app file browser window. Sizes and centers
    itself against the *current* window/display size every time it's
    opened (imgui.APPEARING), so it always fits on screen instead of
    being clipped when the app window is small. Returns
    (still_open, new_browse_dir, picked_file_path_or_None)."""
    picked = None
    io = imgui.get_io()
    disp_w, disp_h = io.display_size
    # fit inside the current window with a margin, but don't go below a
    # usable minimum - if the window is *very* small the browser will be
    # tight, but it will never be pushed off-screen or unreadable
    margin = 40
    win_w = max(260, min(460, disp_w - margin))
    win_h = max(220, min(420, disp_h - margin))
    imgui.set_next_window_size(win_w, win_h, imgui.APPEARING)
    imgui.set_next_window_position(
        max(0, (disp_w - win_w) / 2.0), max(0, (disp_h - win_h) / 2.0), imgui.APPEARING
    )
    expanded, still_open = imgui.begin("Select Foliage Model (.obj)", True)
    if expanded:
        imgui.text_wrapped("Folder: " + browse_dir)
        imgui.push_item_width(-70)
        changed, new_path = imgui.input_text("##browser_path", browse_dir, 512)
        imgui.pop_item_width()
        if changed:
            browse_dir = new_path
        imgui.same_line()
        if imgui.button("Go"):
            if not os.path.isdir(browse_dir):
                browse_dir = default_browse_start_dir()

        if imgui.button(".. (up one level)"):
            parent = os.path.dirname(browse_dir.rstrip(os.sep) or os.sep)
            browse_dir = parent if parent else browse_dir

        imgui.separator()
        subdirs, files = list_dir_entries(browse_dir, ".obj")
        # let the list fill whatever vertical space is left in the window
        list_h = max(100, imgui.get_content_region_available()[1])
        imgui.begin_child("browser_entries", 0, list_h, border=True)
        if not subdirs and not files:
            imgui.text_disabled("(no subfolders or .obj files here)")
        for d in subdirs:
            clicked, _ = imgui.selectable("[folder] " + d, False)
            if clicked:
                browse_dir = os.path.join(browse_dir, d)
        for fname in files:
            clicked, _ = imgui.selectable(fname, False)
            if clicked:
                picked = os.path.join(browse_dir, fname)
        imgui.end_child()
    imgui.end()
    return still_open, browse_dir, picked


# ---------------------------------------------------------------------------
# Road / spline system (Unreal-style spline tool)
# ---------------------------------------------------------------------------
#
# A RoadSpline is a chain of editable control points (world-space x/y/z).
# A smooth Catmull-Rom curve is evaluated through them to build:
#   1. a textured ribbon mesh (the road surface itself)
#   2. a terrain deformation mask that flattens the ground under the road
#      and blends it smoothly into the surrounding terrain over a
#      separately-adjustable margin ("affect width")
#   3. a set of modular prop instances (fences, lamp posts, ...) scattered
#      along the road edges at a configurable spacing/offset, exactly like
#      the foliage brush but constrained to run parallel to the spline.
#
# Everything derived (mesh/props/terrain target heights) is cached and
# only recomputed when the spline is marked dirty (a point moved/added/
# removed, or a shape parameter changed), so editing stays responsive even
# with a fairly dense terrain grid.

def catmull_rom_point(p0, p1, p2, p3, t):
    """Standard centripetal-ish (uniform) Catmull-Rom interpolation between
    p1 and p2, using p0/p3 as the tangent-defining neighbors."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


class RoadSpline:
    """A single editable road: control points -> smooth curve -> mesh +
    terrain carve + modular props. Multiple independent RoadSpline
    instances can coexist (see the "Road Editor" tool in main())."""

    STEPS_PER_SEGMENT = 12  # curve smoothness between consecutive control points

    def __init__(self, name="Road"):
        self.name = name
        self.points = []  # list of float64 np.array([x, y, z])

        # -- shape / terrain interaction --
        self.half_width = 2.5
        self.affect_width = 3.0     # extra terrain blend margin beyond the road edge
        self.height_offset = 0.05   # mesh sits slightly above the carved terrain (avoids z-fighting)
        self.uv_tile_length = 4.0   # world units per texture repeat along the road

        # -- modular props (fence, lamp, ...) --
        self.props_enabled = False
        self.prop_type = "fence"
        self.prop_spacing = 3.0
        self.prop_side = "both"     # "left" | "right" | "both"
        self.prop_offset = 0.3      # extra gap beyond half_width before placing props

        self.selected_point = -1

        self._dirty = True
        self._samples = []
        self._mesh = None            # (verts, normals, uvs, tris) or None
        self._prop_instances = []    # list of (pos, yaw_radians, side)

    # -- editing ----------------------------------------------------------

    def add_point(self, pos):
        self.points.append(np.array(pos, dtype="float64"))
        self._dirty = True
        return len(self.points) - 1

    def move_point(self, idx, pos):
        if 0 <= idx < len(self.points):
            self.points[idx] = np.array(pos, dtype="float64")
            self._dirty = True

    def remove_point(self, idx):
        if 0 <= idx < len(self.points):
            del self.points[idx]
            self._dirty = True
        self.selected_point = -1

    def clear(self):
        self.points = []
        self.selected_point = -1
        self._dirty = True

    def mark_dirty(self):
        self._dirty = True

    def pick_point(self, ray_origin, ray_dir, pick_radius=0.6, max_dist=200.0):
        """Ray-sphere test against every control point (world space).
        Returns the index of the closest hit point along the ray, or -1."""
        best_idx = -1
        best_t = max_dist
        for i, p in enumerate(self.points):
            oc = ray_origin - p
            b = float(np.dot(oc, ray_dir))
            c = float(np.dot(oc, oc)) - pick_radius * pick_radius
            disc = b * b - c
            if disc < 0.0:
                continue
            sqrt_disc = math.sqrt(disc)
            t = -b - sqrt_disc
            if t < 0.0:
                t = -b + sqrt_disc
            if 0.0 <= t < best_t:
                best_t = t
                best_idx = i
        return best_idx

    def resnap_points_to_terrain(self, terrain):
        """Re-sample each point's height from the terrain (call after a
        full terrain regenerate/flatten so the road doesn't float or
        clip through the new ground before apply_to_terrain re-carves it)."""
        for p in self.points:
            gx, gz = terrain.world_to_grid(p[0], p[2])
            if terrain.in_bounds_grid(gx, gz):
                p[1] = terrain.height_at(gx, gz)
        self._dirty = True

    # -- sampling / derived data -------------------------------------------

    def get_samples(self):
        if self._dirty:
            self._rebuild()
        return self._samples

    def get_mesh(self):
        if self._dirty:
            self._rebuild()
        return self._mesh

    def get_prop_instances(self):
        if self._dirty:
            self._rebuild()
        return self._prop_instances

    def _rebuild(self):
        self._samples = self._compute_samples()
        self._mesh = self._compute_mesh(self._samples)
        self._prop_instances = self._compute_prop_instances(self._samples)
        self._dirty = False

    def _compute_samples(self):
        pts = self.points
        n = len(pts)
        if n < 2:
            return []
        samples = []
        cum = 0.0
        prev_pos = None
        for seg in range(n - 1):
            p0 = pts[seg - 1] if seg - 1 >= 0 else pts[seg]
            p1 = pts[seg]
            p2 = pts[seg + 1]
            p3 = pts[seg + 2] if seg + 2 < n else pts[seg + 1]
            steps = self.STEPS_PER_SEGMENT
            last_segment = (seg == n - 2)
            count = steps + 1 if last_segment else steps
            for i in range(count):
                t = i / steps
                pos = catmull_rom_point(p0, p1, p2, p3, t)
                if prev_pos is not None:
                    cum += float(np.linalg.norm(pos - prev_pos))
                samples.append({"pos": pos, "dist": cum})
                prev_pos = pos

        m = len(samples)
        for i in range(m):
            if i == 0:
                tangent = samples[1]["pos"] - samples[0]["pos"]
            elif i == m - 1:
                tangent = samples[i]["pos"] - samples[i - 1]["pos"]
            else:
                tangent = samples[i + 1]["pos"] - samples[i - 1]["pos"]
            tl = np.linalg.norm(tangent)
            tangent = tangent / tl if tl > 1e-8 else np.array([1.0, 0.0, 0.0])
            right = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
            rl = np.linalg.norm(right)
            right = right / rl if rl > 1e-8 else np.array([0.0, 0.0, 1.0])
            samples[i]["tangent"] = tangent
            samples[i]["right"] = right
        return samples

    def _compute_mesh(self, samples):
        n = len(samples)
        if n < 2:
            return None
        verts = np.zeros((n * 2, 3), dtype="float32")
        normals = np.zeros((n * 2, 3), dtype="float32")
        uvs = np.zeros((n * 2, 2), dtype="float32")
        for i, s in enumerate(samples):
            left = (s["pos"] - s["right"] * self.half_width).copy()
            right = (s["pos"] + s["right"] * self.half_width).copy()
            left[1] += self.height_offset
            right[1] += self.height_offset
            verts[i * 2] = left
            verts[i * 2 + 1] = right
            normals[i * 2] = (0.0, 1.0, 0.0)
            normals[i * 2 + 1] = (0.0, 1.0, 0.0)
            v = s["dist"] / max(self.uv_tile_length, 0.01)
            uvs[i * 2] = (0.0, v)
            uvs[i * 2 + 1] = (1.0, v)
        tris = []
        for i in range(n - 1):
            i0, i1, i2, i3 = i * 2, i * 2 + 1, (i + 1) * 2, (i + 1) * 2 + 1
            tris.append((i0, i2, i1))
            tris.append((i1, i2, i3))
        return verts, normals, uvs, np.array(tris, dtype="uint32")

    def _compute_prop_instances(self, samples):
        if not self.props_enabled or len(samples) < 2:
            return []
        instances = []
        last_dist = -1e9
        for s in samples:
            if s["dist"] - last_dist < self.prop_spacing:
                continue
            last_dist = s["dist"]
            yaw = math.atan2(s["tangent"][2], s["tangent"][0])
            off = self.half_width + self.prop_offset
            if self.prop_side in ("left", "both"):
                instances.append((s["pos"] - s["right"] * off, yaw, "left"))
            if self.prop_side in ("right", "both"):
                instances.append((s["pos"] + s["right"] * off, yaw, "right"))
        return instances

    # -- terrain deformation ------------------------------------------------

    def apply_to_terrain(self, terrain):
        """Flattens the terrain under the road (full snap within half_width,
        cosine falloff blend for `affect_width` beyond that), matching the
        feel of the sculpt brush's falloff so the carve reads naturally."""
        samples = self.get_samples()
        if len(samples) < 2:
            return
        positions = np.array([s["pos"] for s in samples], dtype="float64")
        A = positions[:-1]
        B = positions[1:]
        seg_len2 = (B[:, 0] - A[:, 0]) ** 2 + (B[:, 2] - A[:, 2]) ** 2
        seg_len2[seg_len2 < 1e-9] = 1e-9

        vx = terrain.vertices[:, 0].astype("float64")
        vz = terrain.vertices[:, 2].astype("float64")
        n = len(vx)
        min_dist = np.full(n, np.inf)
        target_h = np.zeros(n)

        for i in range(len(A)):
            ax, az, ay = A[i, 0], A[i, 2], A[i, 1]
            bx, bz, by = B[i, 0], B[i, 2], B[i, 1]
            t = ((vx - ax) * (bx - ax) + (vz - az) * (bz - az)) / seg_len2[i]
            t = np.clip(t, 0.0, 1.0)
            cx = ax + t * (bx - ax)
            cz = az + t * (bz - az)
            ch = ay + t * (by - ay)
            d = np.sqrt((vx - cx) ** 2 + (vz - cz) ** 2)
            better = d < min_dist
            min_dist[better] = d[better]
            target_h[better] = ch[better]

        total_radius = self.half_width + self.affect_width
        affected = min_dist <= total_radius
        if not np.any(affected):
            return
        d = min_dist[affected]
        inside = d <= self.half_width
        edge_t = np.zeros_like(d)
        if self.affect_width > 1e-6:
            edge_t = np.clip((d - self.half_width) / self.affect_width, 0.0, 1.0)
        falloff = 0.5 * (1.0 + np.cos(np.pi * edge_t))
        blend = np.where(inside, 1.0, falloff)

        cur = terrain.vertices[affected, 1]
        th = target_h[affected]
        terrain.vertices[affected, 1] = cur + (th - cur) * blend
        terrain.recompute_derived()


# -- modular road props (fence, lamp post - easy to extend) ---------------

ROAD_PROP_TYPE_KEYS = ["fence", "lamp"]
ROAD_PROP_TYPE_LABELS = ["Fence", "Lamp"]


def _build_fence_prop_list():
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    post_h, post_r = 0.7, 0.045
    segs = 6
    glColor3f(0.42, 0.30, 0.16)
    glBegin(GL_TRIANGLES)
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i + 1) / segs
        x0, z0 = math.cos(a0) * post_r, math.sin(a0) * post_r
        x1, z1 = math.cos(a1) * post_r, math.sin(a1) * post_r
        glNormal3f(math.cos(a0), 0.0, math.sin(a0))
        glVertex3f(x0, 0.0, z0)
        glVertex3f(x0, post_h, z0)
        glNormal3f(math.cos(a1), 0.0, math.sin(a1))
        glVertex3f(x1, 0.0, z1)
        glVertex3f(x1, 0.0, z1)
        glVertex3f(x0, post_h, z0)
        glVertex3f(x1, post_h, z1)
    glEnd()
    # two horizontal rail stubs so repeated posts read as a fence line
    rail_len, rail_thick = 0.9, 0.04
    glColor3f(0.46, 0.34, 0.18)
    for rail_y in (0.30, 0.55):
        glBegin(GL_TRIANGLES)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-rail_len * 0.5, rail_y, -rail_thick)
        glVertex3f(rail_len * 0.5, rail_y, -rail_thick)
        glVertex3f(rail_len * 0.5, rail_y, rail_thick)
        glVertex3f(-rail_len * 0.5, rail_y, -rail_thick)
        glVertex3f(rail_len * 0.5, rail_y, rail_thick)
        glVertex3f(-rail_len * 0.5, rail_y, rail_thick)
        glEnd()
    glEndList()
    return list_id


def _build_lamp_prop_list():
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    pole_h, pole_r = 2.2, 0.05
    segs = 6
    glColor3f(0.20, 0.20, 0.22)
    glBegin(GL_TRIANGLES)
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i + 1) / segs
        x0, z0 = math.cos(a0) * pole_r, math.sin(a0) * pole_r
        x1, z1 = math.cos(a1) * pole_r, math.sin(a1) * pole_r
        glNormal3f(math.cos(a0), 0.0, math.sin(a0))
        glVertex3f(x0, 0.0, z0)
        glVertex3f(x0, pole_h, z0)
        glNormal3f(math.cos(a1), 0.0, math.sin(a1))
        glVertex3f(x1, 0.0, z1)
        glVertex3f(x1, 0.0, z1)
        glVertex3f(x0, pole_h, z0)
        glVertex3f(x1, pole_h, z1)
    glEnd()
    glColor3f(0.95, 0.85, 0.45)
    r = 0.16
    glBegin(GL_TRIANGLES)
    for p0, p1, p2 in _sphere_triangles(4, 6):
        for p in (p0, p1, p2):
            nx, ny, nz = p[0], p[1] - 1.0, p[2]
            glNormal3f(nx, ny, nz)
            glVertex3f(p[0] * r, pole_h + (p[1] - 1.0) * r, p[2] * r)
    glEnd()
    glEndList()
    return list_id


ROAD_PROP_LIST_BUILDERS = {"fence": _build_fence_prop_list, "lamp": _build_lamp_prop_list}


def build_road_prop_display_lists():
    """Must be called after the OpenGL context exists."""
    return {name: builder() for name, builder in ROAD_PROP_LIST_BUILDERS.items()}


def build_road_texture():
    """Must be called after the OpenGL context exists."""
    return create_gl_texture(generate_texture((0.22, 0.22, 0.24), seed=7, speckle=True))


def draw_road_props(road, display_lists):
    glEnable(GL_LIGHTING)
    for pos, yaw, _side in road.get_prop_instances():
        glPushMatrix()
        glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
        glRotatef(math.degrees(yaw), 0.0, 1.0, 0.0)
        glCallList(display_lists[road.prop_type])
        glPopMatrix()


def draw_road_mesh(road, texture_id):
    mesh = road.get_mesh()
    if mesh is None:
        return
    verts, normals, uvs, tris = mesh
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glColor3f(1.0, 1.0, 1.0)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glNormalPointer(GL_FLOAT, 0, normals)
    glTexCoordPointer(2, GL_FLOAT, 0, uvs)
    glDrawElements(GL_TRIANGLES, len(tris) * 3, GL_UNSIGNED_INT, tris)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    draw_road_centerline(road)


def draw_road_centerline(road, color=(0.92, 0.85, 0.20)):
    """A simple raised line marking down the middle of the road, for
    visual read of the spline path. (A dashed lane-marking mesh would be
    a nice follow-up, but a solid line keeps this fast and simple.)"""
    samples = road.get_samples()
    if len(samples) < 2:
        return
    glDisable(GL_LIGHTING)
    glColor3f(*color)
    glLineWidth(3.0)
    glBegin(GL_LINE_STRIP)
    for s in samples:
        p = s["pos"]
        glVertex3f(float(p[0]), float(p[1]) + road.height_offset + 0.02, float(p[2]))
    glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def draw_road_control_points(road):
    """Small ring + vertical tick gizmo at each control point, highlighted
    red when selected, for the Road Editor tool."""
    glDisable(GL_LIGHTING)
    for i, p in enumerate(road.points):
        selected = (i == road.selected_point)
        if selected:
            glColor3f(1.0, 0.3, 0.3)
        else:
            glColor3f(0.2, 0.8, 1.0)
        radius = 0.35 if selected else 0.25
        glBegin(GL_LINE_LOOP)
        for j in range(16):
            a = 2.0 * math.pi * j / 16
            glVertex3f(p[0] + math.cos(a) * radius, p[1] + 0.05, p[2] + math.sin(a) * radius)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(p[0], p[1] - 0.3, p[2])
        glVertex3f(p[0], p[1] + 0.3, p[2])
        glEnd()
    glEnable(GL_LIGHTING)


# ---------------------------------------------------------------------------
# Rendering (client-side vertex arrays, no shaders needed for wireframe)
# ---------------------------------------------------------------------------

def draw_terrain_wireframe(terrain):
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, terrain.vertices)
    glDrawElements(GL_LINES, len(terrain.line_indices), GL_UNSIGNED_INT, terrain.line_indices)
    glDisableClientState(GL_VERTEX_ARRAY)


def draw_brush_ring(center, radius, color=(1.0, 1.0, 0.2), segments=48):
    """A flat ring on the XZ plane at the brush's hit point, for feedback.
    Color is used to distinguish tools/modes (sculpt vs foliage paint vs
    foliage erase vs texture paint)."""
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

    # -- realistic terrain texturing (splat-mapped, Unity style) -----------
    terrain_textures = TerrainTextures(terrain, tile_scale=6.0)
    terrain_tex_ids = build_terrain_textures()
    terrain_shader = build_shader_program(TERRAIN_VERT_SRC, TERRAIN_FRAG_SRC)
    terrain_uniforms = {
        name: glGetUniformLocation(terrain_shader, name)
        for name in ("texGrass", "texDirt", "texRock", "texSand",
                     "texScale", "lightDir", "camPos", "fogColor", "fogDensity")
    }
    terrain_uniforms["aWeights"] = glGetAttribLocation(terrain_shader, "aWeights")

    light_dir = (0.35, 0.85, 0.35)
    fog_color = (0.55, 0.62, 0.70)
    fog_density = 0.012

    # -- road / spline system ------------------------------------------------
    road_prop_lists = build_road_prop_display_lists()
    road_texture = build_road_texture()
    roads = []            # list of RoadSpline
    active_road_idx = 0
    dragging_point = False

    cam_pos = np.array([0.0, 15.0, -45.0], dtype="float32")
    cam_yaw = 90.0
    cam_pitch = -15.0

    show_solid = True
    show_wireframe = False

    # -- tool selection: sculpt terrain, paint foliage, paint texture, or edit roads --
    tool_idx = 0
    tool_labels = ["Sculpt Terrain", "Paint Foliage", "Paint Texture", "Road Editor"]

    brush_mode = 0  # 0=raise 1=lower 2=flatten 3=smooth
    brush_modes = ["Raise", "Lower", "Flatten", "Smooth"]
    brush_radius = 5.0
    brush_strength = 1.0

    foliage_type_idx = len(FOLIAGE_RULES)  # default to "All (mix)"
    foliage_paint_rate = 6.0  # attempted placements per ~16ms frame at full rate
    foliage_erase_mode = False
    foliage_soft_edge = True  # denser near brush center, thinning toward the rim
    foliage_scatter_density = 1.0  # only used by the explicit "Scatter Fill" button

    # -- custom foliage import (Sketchfab downloads / your own Blender exports) --
    import_path = ""
    import_name = "My Foliage"
    import_min_h = -1.0
    import_max_h = 2.0
    import_min_up = 0.5
    import_scale_range = (0.6, 1.2)
    import_color = (0.35, 0.45, 0.22)
    import_status = ""
    show_inapp_browser = False
    browse_dir = default_browse_start_dir()

    texture_layer_idx = 0  # index into TerrainTextures.LAYER_NAMES
    texture_strength = 0.6
    texture_tile_scale = terrain_textures.tile_scale

    looking = False        # right mouse button held
    flatten_height = None  # captured at the start of a flatten stroke

    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60)
        lmb_just_pressed = False
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
                    tool_idx = (tool_idx + 1) % len(tool_labels)
                if event.key == K_DELETE and tool_idx == 3 and roads:
                    road = roads[active_road_idx]
                    if road.selected_point >= 0:
                        road.remove_point(road.selected_point)
                        road.apply_to_terrain(terrain)
                        foliage.update_all(terrain)
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
                lmb_just_pressed = True

        imgui.new_frame()

        # ---- ImGui panel ----
        imgui.set_next_window_position(15, 15, imgui.ONCE)
        imgui.set_next_window_size(380, 0, imgui.ONCE)
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
        elif tool_idx == 1:
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
            if imgui.tree_node("Import custom foliage (.obj)"):
                imgui.text_wrapped(
                    "Get a model from Sketchfab (pick the OBJ / Original "
                    "Format download for CC-licensed models) or make your "
                    "own in Blender and export via File > Export > "
                    "Wavefront (.obj). Other formats (.fbx/.gltf/.glb)? "
                    "Open them in Blender once and re-export as .obj."
                )
                imgui.text("File path:")
                imgui.push_item_width(-1)
                _, import_path = imgui.input_text("##foliage_import_path", import_path, 512)
                imgui.pop_item_width()
                if imgui.button("Browse..."):
                    dialog_worked, picked = browse_for_model_file()
                    if dialog_worked:
                        if picked:
                            import_path = picked
                    else:
                        show_inapp_browser = True
                imgui.same_line()
                if imgui.button("Browse (in-app)"):
                    show_inapp_browser = True
                _, import_name = imgui.input_text("Name", import_name, 64)
                _, import_min_h = imgui.slider_float("Min height", import_min_h, -2.0, 2.0)
                _, import_max_h = imgui.slider_float("Max height", import_max_h, -2.0, 2.0)
                _, import_min_up = imgui.slider_float("Min flatness", import_min_up, 0.0, 1.0)
                _, import_scale_range = imgui.slider_float2(
                    "Scale range", import_scale_range[0], import_scale_range[1], 0.1, 3.0
                )
                _, import_color = imgui.color_edit3("Tint color", *import_color)
                if imgui.button("Import as new foliage type"):
                    if import_path and os.path.isfile(import_path):
                        try:
                            was_all = (foliage_type_idx >= len(FOLIAGE_RULES))
                            new_idx = register_custom_foliage(
                                foliage_display_lists,
                                import_name or "Custom",
                                import_path,
                                min(import_min_h, import_max_h),
                                max(import_min_h, import_max_h),
                                import_min_up,
                                (min(import_scale_range), max(import_scale_range)),
                                color=import_color,
                            )
                            # keep "All (mix)" selected if it was selected before,
                            # otherwise auto-select the freshly imported type
                            foliage_type_idx = len(FOLIAGE_RULES) if was_all else new_idx
                            import_status = "Imported '{}' as a new foliage type.".format(import_name)
                        except Exception as e:
                            import_status = "Import failed: {}".format(e)
                    else:
                        import_status = "That file path doesn't exist - check it or use Browse..."
                if import_status:
                    imgui.text_wrapped(import_status)
                imgui.tree_pop()
        elif tool_idx == 2:
            # -- terrain texture (splat) painting controls --
            imgui.text("Texture Brush")
            _, texture_layer_idx = imgui.combo("Layer", texture_layer_idx, TerrainTextures.LAYER_NAMES)
            _, brush_radius = imgui.slider_float("Radius##texture", brush_radius, 0.5, 20.0)
            _, texture_strength = imgui.slider_float("Strength##texture", texture_strength, 0.02, 1.0)
            changed_scale, texture_tile_scale = imgui.slider_float("Tile scale", texture_tile_scale, 1.0, 20.0)
            if changed_scale:
                terrain_textures.tile_scale = texture_tile_scale
            if imgui.button("Reset from height/slope"):
                terrain_textures.initialize_from_height_slope(terrain)
        else:
            # -- road / spline editor controls --
            imgui.text("Road Editor")

            if roads:
                road_names = [r.name for r in roads]
                _, active_road_idx = imgui.combo("Active Road", active_road_idx, road_names)
                active_road_idx = max(0, min(active_road_idx, len(roads) - 1))
                imgui.same_line()
            if imgui.button("New Road"):
                roads.append(RoadSpline("Road {}".format(len(roads) + 1)))
                active_road_idx = len(roads) - 1

            if roads:
                if imgui.button("Delete Road"):
                    roads.pop(active_road_idx)
                    active_road_idx = max(0, active_road_idx - 1)

            road = roads[active_road_idx] if roads else None

            if road is not None:
                if imgui.button("Delete Selected Point"):
                    road.remove_point(road.selected_point)
                    road.apply_to_terrain(terrain)
                    foliage.update_all(terrain)
                imgui.same_line()
                if imgui.button("Clear Points"):
                    road.clear()

                imgui.text_wrapped(
                    "{} control point(s). Click empty ground to add a point, "
                    "click + drag an existing point to move it, Delete key "
                    "or the button above to remove the selected one.".format(len(road.points))
                )

                imgui.separator()
                imgui.text("Road Shape")
                changed_w, road.half_width = imgui.slider_float("Width", road.half_width, 0.5, 10.0)
                changed_a, road.affect_width = imgui.slider_float("Terrain blend margin", road.affect_width, 0.0, 10.0)
                changed_h, road.height_offset = imgui.slider_float("Mesh height offset", road.height_offset, 0.0, 0.5)
                changed_u, road.uv_tile_length = imgui.slider_float("Texture tile length", road.uv_tile_length, 0.5, 10.0)
                if changed_w or changed_a or changed_h or changed_u:
                    road.mark_dirty()

                if imgui.button("Apply Road to Terrain"):
                    road.apply_to_terrain(terrain)
                    foliage.update_all(terrain)
                imgui.text_wrapped(
                    "Terrain carves automatically as you add/move/delete points. "
                    "Press this after changing Width or Blend margin to re-carve."
                )

                imgui.separator()
                imgui.text("Modular Props")
                _, road.props_enabled = imgui.checkbox("Enable props", road.props_enabled)
                prop_type_idx = ROAD_PROP_TYPE_KEYS.index(road.prop_type)
                changed_pt, prop_type_idx = imgui.combo("Prop type", prop_type_idx, ROAD_PROP_TYPE_LABELS)
                if changed_pt:
                    road.prop_type = ROAD_PROP_TYPE_KEYS[prop_type_idx]
                    road.mark_dirty()
                side_options = ["left", "right", "both"]
                side_idx = side_options.index(road.prop_side)
                changed_side, side_idx = imgui.combo("Side", side_idx, ["Left", "Right", "Both"])
                if changed_side:
                    road.prop_side = side_options[side_idx]
                    road.mark_dirty()
                changed_sp, road.prop_spacing = imgui.slider_float("Spacing", road.prop_spacing, 0.5, 15.0)
                changed_of, road.prop_offset = imgui.slider_float("Offset from edge", road.prop_offset, 0.0, 3.0)
                if changed_sp or changed_of:
                    road.mark_dirty()
            else:
                imgui.text_wrapped("No road yet - click 'New Road' to start placing points.")

        imgui.separator()
        _, show_solid = imgui.checkbox("Show solid faces (G)", show_solid)
        _, show_wireframe = imgui.checkbox("Show wireframe (F)", show_wireframe)
        _, show_foliage = imgui.checkbox("Show foliage (T)", show_foliage)
        imgui.text("Foliage instances: {}".format(len(foliage)))

        imgui.separator()
        if imgui.button("Regenerate terrain"):
            terrain.regenerate(seed=int(np.random.randint(0, 1_000_000)))
            terrain_textures.initialize_from_height_slope(terrain)  # fresh terrain -> fresh splat seed
            for road in roads:
                road.resnap_points_to_terrain(terrain)
                road.apply_to_terrain(terrain)
            foliage.update_all(terrain)  # re-glue/cull existing foliage, add nothing new
        imgui.same_line()
        if imgui.button("Flatten to 0"):
            terrain.flatten_all(0.0)
            for road in roads:
                road.resnap_points_to_terrain(terrain)
                road.apply_to_terrain(terrain)
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

        if imgui.tree_node("Atmosphere"):
            _, fog_density = imgui.slider_float("Fog density", fog_density, 0.0, 0.05)
            changed_fog, fog_color = imgui.color_edit3("Fog / sky color", *fog_color)
            imgui.tree_pop()

        imgui.separator()
        imgui.text_wrapped(
            "RMB: look + WASD fly | Q/E or Shift/Space: down/up | "
            "Tab: switch tool | LMB: paint/edit with the active tool | "
            "T: toggle foliage | Delete: remove selected road point"
        )
        imgui.end()

        if show_inapp_browser:
            show_inapp_browser, browse_dir, picked_file = draw_inapp_browser(browse_dir)
            if picked_file:
                import_path = picked_file
                import_status = "Selected: " + picked_file
                show_inapp_browser = False

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

            if tool_idx == 3 and roads:
                # -- road / spline editing --
                road = roads[active_road_idx]
                if mouse_buttons[0] and not looking:
                    if not dragging_point:
                        if lmb_just_pressed:
                            idx = road.pick_point(near, ray_dir)
                            if idx >= 0:
                                road.selected_point = idx
                                dragging_point = True
                            elif brush_hit is not None:
                                road.add_point(brush_hit)
                                road.selected_point = len(road.points) - 1
                    elif brush_hit is not None:
                        road.move_point(road.selected_point, brush_hit)
                else:
                    if dragging_point:
                        road.apply_to_terrain(terrain)
                        foliage.update_all(terrain)
                    dragging_point = False
            elif brush_hit is not None and mouse_buttons[0] and not looking:
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
                elif tool_idx == 1:
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
                else:
                    # -- terrain texture (splat) painting --
                    terrain_textures.paint(
                        terrain, brush_hit, brush_radius,
                        texture_strength * (dt / 16.0), texture_layer_idx,
                    )

        # ---- render scene ----
        glClearColor(fog_color[0], fog_color[1], fog_color[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            cam_pos[0], cam_pos[1], cam_pos[2],
            cam_pos[0] + front[0], cam_pos[1] + front[1], cam_pos[2] + front[2],
            0.0, 1.0, 0.0,
        )

        if show_solid:
            draw_terrain_solid_textured(
                terrain, terrain_tex_ids, terrain_textures.weights,
                terrain_shader, terrain_uniforms,
                cam_pos, terrain_textures.tile_scale, light_dir, fog_color, fog_density,
            )
        if show_wireframe:
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)
            draw_terrain_wireframe(terrain)
            glEnable(GL_LIGHTING)
        if show_foliage:
            draw_foliage(foliage, foliage_display_lists)

        for road in roads:
            draw_road_mesh(road, road_texture)
            if road.props_enabled:
                draw_road_props(road, road_prop_lists)

        if tool_idx == 3 and roads:
            draw_road_control_points(roads[active_road_idx])

        if brush_hit is not None and tool_idx != 3:
            if tool_idx == 0:
                ring_color = (1.0, 1.0, 0.2)       # yellow: sculpt
            elif tool_idx == 1:
                ring_color = (1.0, 0.25, 0.25) if foliage_erase_mode else (0.25, 1.0, 0.35)
            else:
                ring_color = (0.3, 0.6, 1.0)       # blue: texture paint
            draw_brush_ring(brush_hit, brush_radius, ring_color)
            glEnable(GL_LIGHTING)

        imgui.render()
        impl.render(imgui.get_draw_data())

        pygame.display.flip()


