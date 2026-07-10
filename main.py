import math
import sys
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *


# ---------- Terrain generation ----------

def generate_terrain(width, height, scale):
    vertices = []
    for z in range(height):
        for x in range(width):
            nx = x / scale
            nz = z / scale
            y = (math.sin(nx) * math.cos(nz) * 2.0) + (math.sin(nx * 2) * 0.5)
            vertices.append([float(x - width / 2), y, float(z - height / 2)])
    return np.array(vertices, dtype="float32")


def build_triangles(width, height):
    """Two triangles per grid cell, as index triples."""
    tris = []
    for z in range(height - 1):
        for x in range(width - 1):
            i0 = x + z * width
            i1 = x + (z + 1) * width
            i2 = (x + 1) + z * width
            i3 = (x + 1) + (z + 1) * width
            tris.append((i0, i1, i2))
            tris.append((i2, i1, i3))
    return tris


def compute_face_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(n)
    if length < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype="float32")
    return n / length


def height_to_color(y, y_min, y_max):
    """Simple low->high colour ramp: brown -> green -> white."""
    t = 0.0 if y_max == y_min else (y - y_min) / (y_max - y_min)
    if t < 0.5:
        k = t / 0.5
        low = np.array([0.45, 0.32, 0.18])
        mid = np.array([0.25, 0.55, 0.2])
        return low * (1 - k) + mid * k
    else:
        k = (t - 0.5) / 0.5
        mid = np.array([0.25, 0.55, 0.2])
        high = np.array([0.9, 0.9, 0.85])
        return mid * (1 - k) + high * k


def build_terrain_geometry(vertices, width, height):
    """Precompute everything needed to draw solid faces once, since the
    terrain itself never changes shape at runtime."""
    tris = build_triangles(width, height)
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())

    faces = []  # (v0, v1, v2, normal, color0, color1, color2)
    for (i0, i1, i2) in tris:
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        normal = compute_face_normal(v0, v1, v2)
        c0 = height_to_color(v0[1], y_min, y_max)
        c1 = height_to_color(v1[1], y_min, y_max)
        c2 = height_to_color(v2[1], y_min, y_max)
        faces.append((v0, v1, v2, normal, c0, c1, c2))
    return faces


def draw_terrain_solid(faces):
    glBegin(GL_TRIANGLES)
    for v0, v1, v2, normal, c0, c1, c2 in faces:
        glNormal3fv(normal)
        glColor3fv(c0)
        glVertex3fv(v0)
        glColor3fv(c1)
        glVertex3fv(v1)
        glColor3fv(c2)
        glVertex3fv(v2)
    glEnd()


def draw_terrain_wireframe(vertices, width, height):
    glBegin(GL_LINES)
    for z in range(height - 1):
        for x in range(width - 1):
            v1 = vertices[x + z * width]
            v2 = vertices[x + (z + 1) * width]
            v3 = vertices[x + 1 + (z + 1) * width]

            glVertex3fv(v1)
            glVertex3fv(v2)

            glVertex3fv(v2)
            glVertex3fv(v3)

            glVertex3fv(v3)
            glVertex3fv(v1)
    glEnd()


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        "Terrain viewer  |  F = toggle wireframe, G = toggle solid faces"
    )

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    # Basic lighting so the solid faces actually show shape, not just flat color
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 30.0, 10.0, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)

    # Nudge wireframe slightly toward the camera in depth so it doesn't
    # z-fight with the solid faces when both are shown at once.
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    # Camera variables
    cam_pos = np.array([0.0, 15.0, -45.0], dtype="float32")
    cam_yaw = 90.0
    cam_pitch = -15.0

    terrain_size = 50
    vertices = generate_terrain(terrain_size, terrain_size, 5.0)
    faces = build_terrain_geometry(vertices, terrain_size, terrain_size)

    show_solid = True
    show_wireframe = False

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    center = (display[0] // 2, display[1] // 2)
    pygame.mouse.set_pos(center)
    pygame.mouse.get_rel()

    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == K_f:
                    show_wireframe = not show_wireframe
                if event.key == K_g:
                    show_solid = not show_solid

        mouse_x, mouse_y = pygame.mouse.get_rel()
        cam_yaw += mouse_x * 0.1
        cam_pitch -= mouse_y * 0.1
        cam_pitch = max(-89.0, min(89.0, cam_pitch))
        pygame.mouse.set_pos(center)

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

        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(
            cam_pos[0], cam_pos[1], cam_pos[2],
            cam_pos[0] + front[0], cam_pos[1] + front[1], cam_pos[2] + front[2],
            0.0, 1.0, 0.0,
        )

        if show_solid:
            glEnable(GL_LIGHTING)
            draw_terrain_solid(faces)

        if show_wireframe:
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)
            draw_terrain_wireframe(vertices, terrain_size, terrain_size)

        pygame.display.flip()


if __name__ == "__main__":
    main()