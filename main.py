

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


def draw_brush_ring(center, radius, segments=48):
    """A flat ring on the XZ plane at the brush's hit point, for feedback."""
    glDisable(GL_LIGHTING)
    glColor3f(1.0, 1.0, 0.2)
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

    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    imgui.create_context()
    impl = PygameRenderer()
    io = imgui.get_io()
    io.display_size = display

    terrain = Terrain(width=50, height=50, scale=5.0, seed=0)

    cam_pos = np.array([0.0, 15.0, -45.0], dtype="float32")
    cam_yaw = 90.0
    cam_pitch = -15.0

    show_solid = True
    show_wireframe = False

    brush_mode = 0  # 0=raise 1=lower 2=flatten 3=smooth
    brush_modes = ["Raise", "Lower", "Flatten", "Smooth"]
    brush_radius = 5.0
    brush_strength = 1.0

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
        imgui.set_next_window_size(300, 0, imgui.ONCE)
        imgui.begin("Terrain Brush", True)

        imgui.text("Mode: {}".format("Look/Fly (RMB held)" if looking else "Edit (LMB paints)"))
        imgui.separator()

        clicked, brush_mode = imgui.combo("Brush", brush_mode, brush_modes)
        _, brush_radius = imgui.slider_float("Radius", brush_radius, 1.0, 20.0)
        _, brush_strength = imgui.slider_float("Strength", brush_strength, 0.05, 5.0)

        imgui.separator()
        _, show_solid = imgui.checkbox("Show solid faces (G)", show_solid)
        _, show_wireframe = imgui.checkbox("Show wireframe (F)", show_wireframe)

        imgui.separator()
        if imgui.button("Regenerate terrain"):
            terrain.regenerate(seed=np.random.randint(0, 1_000_000))
        imgui.same_line()
        if imgui.button("Flatten to 0"):
            terrain.flatten_all(0.0)

        imgui.separator()
        imgui.text_wrapped(
            "RMB: look + WASD fly | Q/E or Shift/Space: down/up | "
            "LMB: paint with brush"
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
                mode_name = brush_modes[brush_mode].lower()
                if mode_name == "flatten" and flatten_height is None:
                    flatten_height = float(brush_hit[1])
                terrain.apply_brush(
                    brush_hit, brush_radius, brush_strength * (dt / 16.0),
                    mode_name, flatten_height=flatten_height,
                )

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
        if brush_hit is not None:
            draw_brush_ring(brush_hit, brush_radius)
            glEnable(GL_LIGHTING)

        imgui.render()
        impl.render(imgui.get_draw_data())

        pygame.display.flip()


if __name__ == "__main__":
    main()