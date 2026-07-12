# 🌍 PSX -  Terrain Editor

> A lightweight **3D Terrain Editor** built completely in **Python + OpenGL** for creating game environments, roads, forests, props, and exporting complete scenes directly to Blender.

<img width="100%" height="1000" alt="Gemini_Generated_Image_sudn85sudn85sudn" src="https://github.com/user-attachments/assets/3e5603f0-5442-4387-953f-e790a425adb6" />

---


# ✨ Features

- 🏔️ Interactive Terrain Generation
- 🌲 Procedural Tree Painting
- 🪨 Rock & Prop Placement
- 🛣️ Easy Road Creation Tool
- 🚧 Fence Generator
- 💡 Light Object Placement
- 🌫️ Adjustable Atmosphere & Fog
- 🎨 Terrain Texture Painting
- 📦 Export entire scene to Blender (.OBJ + .MTL)
- 🖼️ Automatic Texture Export
- ⚡ Real-time OpenGL Rendering
- 🖱️ Modern ImGui Editor Interface
- 🎮 Camera Controls for Easy Navigation

---


### 🏔️ Terrain Editing
- Create and edit terrains in real-time
- Adjustable terrain resolution
- Terrain texture painting
- Texture tiling controls
- Multiple terrain materials

### 🌿 Foliage System
- Paint thousands of trees with ease
- Place grass, bushes, and vegetation
- Customize foliage density
- Scale and rotation randomization
- Replace foliage with your own models

### 🪨 Props & Assets
- Import your own **OBJ** models
- Use custom rocks, houses, trees, props, and decorations
- Organize reusable assets
- Rotate, scale, and position objects freely

### 🛣️ Road System
- Interactive road creation
- Curved road editing
- Adjustable road width
- Custom road materials
- Easily replace road textures

### 🚧 Fence System
- Automatically generate fences along roads
- Adjustable spacing
- Supports custom fence models

### 🌄 Environment
- Adjustable fog density
- Sky color controls
- Real-time lighting preview
- Outdoor environment settings

### 🎨 Texture System
- Import your own terrain textures
- Tile textures to any scale
- Mix different terrain materials
- Supports high-resolution textures

### 📦 Scene Export
- Export the complete scene to Blender
- Generates:
  - OBJ
  - MTL
  - Baked textures
- Every object is exported separately for easy editing

### ⚡ Performance
- GPU accelerated rendering using OpenGL
- Real-time editing
- Lightweight Python application
- Immediate viewport updates

### 🖥️ Modern Editor
- Dear ImGui based editor
- Easy-to-use interface
- Fast workflow
- Designed for rapid environment creation

---


# 📸 Screenshots

### Terrain Editor

![Editor](docs/images/editor.png)

### Exported Scene in Blender

![Blender](docs/images/blender.png)

---

# 🚀 Built With

This project was developed completely in **Python** using the following libraries.

| Library | Purpose |
|----------|---------|
| 🐍 Python | Main Programming Language |
| OpenGL (PyOpenGL) | Real-time Rendering |
| Pygame | Window & Input Handling |
| Pillow | Texture Loading & Image Processing & Export Texture |
| NumPy | Math Operations |
| Dear ImGui | Editor User Interface |

---

# 🎯 What Can You Create?

Using this editor you can build scenes like:

- Forests
- Hills
- Mountains
- Dirt Roads
- Racing Tracks
- Nature Environments
- Low Poly Worlds
- Open World Prototypes
- Game Levels

---

# 📦 Export

The editor exports your scene into Blender compatible files.

Supported output:

```
Scene.obj
Scene.mtl
Textures/
```

Everything including:

- Terrain
- Trees
- Roads
- Fences
- Rocks
- Props

is exported as meshes.

Simply import the OBJ into Blender and continue editing.

---

# 🎮 Controls

| Key | Action |
|------|--------|
| W A S D | Fly forward / left / back / right (while RMB held) |
| Right Mouse Button (hold) | Look around and unlock WASD flying |
| Q / Left Shift | Fly down |
| E / Space | Fly up |
| Tab | Cycle Sculpt → Foliage → Texture → Road Editor |
| Left Mouse Button | Paint or edit with whichever tool is active |
| F | Toggle wireframe overlay |
| G | Toggle solid terrain faces |
| T | Toggle foliage visibility |
| Delete | Remove the selected road point (Road Editor only) |
| Esc | Quit |

---

# 📂 Project Structure

```
TerrainEditor/
│
├── main.py
│
├── Terrian.py
│
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/sohamkoltecc-ai/PyTerrianEditor
```

Go into the project

```bash
cd PyTerrianEditor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

# 📦 Requirements

- Python 3.10+
- OpenGL Compatible GPU
- Windows (Currently Tested)

---

# 🌱 Future Plans

- Heightmap Import
- Terrain Sculpting Brushes
- Water System
- River Generator
- Better Material Editor
- Terrain Layers
- Vegetation System
- Undo / Redo
- FBX Export
- GLTF Export
- Object Gizmos
- Scene Saving (.terrain)
- Terrain LOD Generation
- Shadow Improvements
- Skybox Editor
- Weather System

---

# ❤️ Open Source

This project is open source and contributions are always welcome.

You can help by:

- Fixing Bugs
- Improving Performance
- Adding New Features
- Refactoring Code
- Improving Documentation
- Creating Better UI
- Testing on Different Platforms

---

# 🤝 Contributing

1. Fork this repository
2. Create your feature branch

```bash
git checkout -b feature/NewFeature
```

3. Commit your changes

```bash
git commit -m "Add amazing feature"
```

4. Push

```bash
git push origin feature/NewFeature
```

5. Open a Pull Request

---

# ⭐ Support

If you find this project useful, please consider giving it a **Star ⭐** on GitHub.

It really helps the project grow!

---

# 📄 License

This project is licensed under the **MIT License**.

Feel free to use it for personal and commercial projects.

---

# 💙 Made with Python & OpenGL

> Building lightweight game development tools for everyone.

---
