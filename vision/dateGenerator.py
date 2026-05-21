# Run inside Blender: Scripting tab → paste → Run Script
import bpy, csv, random, math, os

DESKTOP     = os.path.join(os.path.expanduser('~'), 'Desktop')
OUTPUT_DIR  = os.path.join(DESKTOP, 'dataset', 'images')
TXT_PATH    = os.path.join(DESKTOP, 'dataset', 'labels.txt')
NUM_SAMPLES = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

SHAPES = {
    'cube':     lambda: bpy.ops.mesh.primitive_cube_add(),
    'sphere':   lambda: bpy.ops.mesh.primitive_uv_sphere_add(),
    'cylinder': lambda: bpy.ops.mesh.primitive_cylinder_add(),
    'cone':     lambda: bpy.ops.mesh.primitive_cone_add(),
    'torus':    lambda: bpy.ops.mesh.primitive_torus_add(),
}

COLORS = {
    'red':    (1.0, 0.0, 0.0, 1.0),
    'yellow': (1.0, 1.0, 0.0, 1.0),
    'green':  (0.0, 0.8, 0.0, 1.0),
    'blue':   (0.0, 0.0, 1.0, 1.0),
}

# ── White ground plane ──────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.01))
ground = bpy.context.active_object
ground.name = 'Ground'
mat_ground = bpy.data.materials.new(name='GroundWhite')
mat_ground.use_nodes = True
nodes_g = mat_ground.node_tree.nodes
nodes_g.clear()
bsdf_g = nodes_g.new(type='ShaderNodeBsdfPrincipled')
bsdf_g.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
bsdf_g.inputs['Roughness'].default_value = 0.9
output_g = nodes_g.new(type='ShaderNodeOutputMaterial')
mat_ground.node_tree.links.new(bsdf_g.outputs['BSDF'], output_g.inputs['Surface'])
ground.data.materials.append(mat_ground)

# ── Camera: slightly tilted ─────────────────────────────────────
cam = bpy.data.objects['Camera']
TILT_DEG = 15
cam.rotation_euler = (math.radians(TILT_DEG), 0, 0)
CAM_Z = 5.0
CAM_Y_OFFSET = -CAM_Z * math.tan(math.radians(TILT_DEG))
cam.location = (0, CAM_Y_OFFSET, CAM_Z)

# ── Resolution: 640 × 360 RGB ───────────────────────────────────
scene = bpy.context.scene
scene.render.resolution_x = 640
scene.render.resolution_y = 360
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.quality = 95

# ── Compute safe placement bounds from camera FOV ───────────────
cam_data = cam.data
ASPECT   = 640 / 360
HALF_H   = math.tan(cam_data.angle_y / 2) * CAM_Z
HALF_W   = HALF_H * ASPECT

print(f"Camera FOV vertical : {math.degrees(cam_data.angle_y):.1f}°")
print(f"Visible area        : X ±{HALF_W:.2f}  Y ±{HALF_H:.2f}")

HALF_EXTENT = 0.233 * 0.6
MARGIN      = 0.4
X_LIMIT     = HALF_W - HALF_EXTENT - MARGIN
Y_LIMIT     = HALF_H - HALF_EXTENT - MARGIN

print(f"Placement limits    : X ±{X_LIMIT:.2f}  Y ±{Y_LIMIT:.2f}")

# ── Blender camera projection matrix ───────────────────────────
# Used to convert 3D world position → exact 2D pixel coordinate
# This is the most accurate way to get center pixel from 3D center
def world_to_pixel(scene, cam, world_pos):
    from bpy_extras.object_utils import world_to_camera_view
    co = world_to_camera_view(scene, cam, world_pos)
    px = co.x * scene.render.resolution_x
    py = (1.0 - co.y) * scene.render.resolution_y  # flip y: top-left origin
    return round(px, 1), round(py, 1)

with open(TXT_PATH, 'w') as f:
    f.write(f"{'Index':<8} {'Filename':<20} {'Shape':<12} {'Color':<10} {'X (px)':<10} {'Y (px)'}\n")
    f.write('-' * 72 + '\n')

    for i in range(NUM_SAMPLES):
        # ── Clear previous mesh objects (keep Ground) ───────────
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.name != 'Ground':
                obj.select_set(True)
        bpy.ops.object.delete()

        # ── Random shape ────────────────────────────────────────
        label = random.choice(list(SHAPES.keys()))
        SHAPES[label]()
        obj = bpy.context.active_object

        # ── Random scale and rotation ───────────────────────────
        s = random.uniform(0.1, 0.233)
        obj.scale = (s, s, s)
        obj.rotation_euler = (0, 0, random.uniform(0, math.pi * 2))

        # Apply scale so bounding box reflects true size
        bpy.ops.object.transform_apply(scale=True)

        # Set origin to geometry center so location = visual center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        # ── Position guaranteed inside frame ────────────────────
        rx = random.uniform(-X_LIMIT, X_LIMIT)
        ry = random.uniform(-Y_LIMIT, Y_LIMIT)

        # Raise object so it sits on the ground (z = half its height)
        obj_height = obj.dimensions.z
        obj.location = (rx, ry, obj_height / 2)

        # ── Random color with nodes (works in EEVEE render) ─────
        color_name, color_rgba = random.choice(list(COLORS.items()))
        mat = bpy.data.materials.new(name='mat')
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = color_rgba
        bsdf.inputs['Roughness'].default_value = 0.5
        bsdf.inputs['Metallic'].default_value = 0.0
        output = nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        obj.data.materials.append(mat)

        # ── Random light intensity ──────────────────────────────
        for light in bpy.data.lights:
            light.energy = random.uniform(300, 1200)

        # ── Render ──────────────────────────────────────────────
        fname = f'img_{i:05d}.jpg'
        scene.render.filepath = os.path.join(OUTPUT_DIR, fname)
        bpy.ops.render.render(write_still=True)

        # ── Project 3D object center → exact pixel coordinate ───
        # Uses Blender's own camera projection so tilt is accounted for
        center_3d = obj.location
        px, py = world_to_pixel(scene, cam, center_3d)

        # ── Write to .txt ────────────────────────────────────────
        f.write(f"{i:<8} {fname:<20} {label:<12} {color_name:<10} {px:<10} {py}\n")
        f.flush()

        print(f"[{i+1}/{NUM_SAMPLES}] {label} ({color_name}) at ({px}, {py})")