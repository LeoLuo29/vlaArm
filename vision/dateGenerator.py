# Run inside Blender: Scripting tab → paste → Run Script
import bpy, csv, random, math, os

DESKTOP     = os.path.join(os.path.expanduser('~'), 'Desktop')
OUTPUT_DIR  = os.path.join(DESKTOP, 'dataset', 'images')
TXT_PATH    = os.path.join(DESKTOP, 'dataset', 'labels.txt')
NUM_SAMPLES = 3000

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
    'orange': (1.0, 0.4, 0.0, 1.0),
    'purple': (0.6, 0.0, 1.0, 1.0),
}

# ── Helper: node-based material ─────────────────────────────────
def make_material(name, color_rgba, roughness=0.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = color_rgba
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = 0.0
    out = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

# ── Helper: assign different color per face group ───────────────
def assign_face_colors(obj):
    import bmesh
    mesh = obj.data
    color_list = list(COLORS.items())
    obj.data.materials.clear()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    normal_groups = {}
    for face in bm.faces:
        key = (round(face.normal.x, 1),
               round(face.normal.y, 1),
               round(face.normal.z, 1))
        if key not in normal_groups:
            normal_groups[key] = []
        normal_groups[key].append(face.index)
    bm.free()
    for group_idx, (normal_key, face_indices) in enumerate(normal_groups.items()):
        color_name, color_rgba = color_list[group_idx % len(color_list)]
        # Slight roughness variation per face for visual interest
        roughness = random.uniform(0.3, 0.7)
        mat = make_material(f'mat_{group_idx}_{color_name}', color_rgba, roughness)
        obj.data.materials.append(mat)
        for fi in face_indices:
            mesh.polygons[fi].material_index = group_idx

# ── Helper: noisy ground material ───────────────────────────────
def make_ground_material():
    mat = bpy.data.materials.new(name='GroundNoisy')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Texture coordinate and mapping
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    mapping   = nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (8.0, 8.0, 8.0)
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])

    # Noise texture for surface variation
    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value     = 12.0
    noise.inputs['Detail'].default_value    = 8.0
    noise.inputs['Roughness'].default_value = 0.65
    noise.inputs['Distortion'].default_value = 0.3
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])

    # Color ramp: white with subtle warm/cool variation
    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.interpolation = 'LINEAR'
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color    = (0.88, 0.86, 0.84, 1.0)  # warm off-white
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color    = (0.97, 0.97, 0.98, 1.0)  # cool white
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])

    # Bump node for surface micro-relief
    bump = nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.4
    bump.inputs['Distance'].default_value = 0.05
    links.new(noise.outputs['Fac'], bump.inputs['Height'])

    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.85
    links.new(ramp.outputs['Color'],  bsdf.inputs['Base Color'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    out = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

# ── Helper: build the lighting rig ──────────────────────────────
def setup_lights():
    # Remove all existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    def add_light(name, light_type, location, energy, color=(1,1,1), size=1.0):
        bpy.ops.object.light_add(type=light_type, location=location)
        light = bpy.context.active_object
        light.name = name
        light.data.energy = energy
        light.data.color  = color
        if light_type == 'AREA':
            light.data.size = size
        elif light_type == 'SPOT':
            light.data.spot_size  = math.radians(45)
            light.data.spot_blend = 0.3
        return light

    # Key light — strong, slightly warm, from upper left
    key = add_light('KeyLight', 'AREA',
                    location=(-3, -3, 6),
                    energy=800,
                    color=(1.0, 0.95, 0.88),
                    size=2.0)
    key.rotation_euler = (math.radians(50), 0, math.radians(-45))

    # Fill light — soft, cool, from right to fill shadows
    fill = add_light('FillLight', 'AREA',
                     location=(4, -1, 4),
                     energy=250,
                     color=(0.88, 0.93, 1.0),
                     size=3.0)
    fill.rotation_euler = (math.radians(55), 0, math.radians(60))

    # Rim light — from behind to create highlight on back edges
    rim = add_light('RimLight', 'SPOT',
                    location=(0, 4, 5),
                    energy=400,
                    color=(1.0, 1.0, 1.0))
    rim.rotation_euler = (math.radians(-40), 0, 0)

    # Ambient — very low energy point light at camera to reduce
    # completely black shadows
    add_light('AmbientFill', 'POINT',
              location=(0, -1.3, 5),
              energy=80,
              color=(1.0, 1.0, 1.0))

    return key, fill, rim

# ── White ground plane with noise ───────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.01))
ground = bpy.context.active_object
ground.name = 'Ground'
ground.data.materials.append(make_ground_material())

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

# ── Blender camera projection → exact pixel coordinate ──────────
def world_to_pixel(scene, cam, world_pos):
    from bpy_extras.object_utils import world_to_camera_view
    co = world_to_camera_view(scene, cam, world_pos)
    px = co.x * scene.render.resolution_x
    py = (1.0 - co.y) * scene.render.resolution_y
    return round(px, 1), round(py, 1)

# ── Build light rig once ─────────────────────────────────────────
key_light, fill_light, rim_light = setup_lights()

with open(TXT_PATH, 'w') as f:
    f.write(f"{'Index':<8} {'Filename':<20} {'Shape':<12} {'X (px)':<10} {'Y (px)'}\n")
    f.write('-' * 65 + '\n')

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

        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        # ── Position guaranteed inside frame ────────────────────
        rx = random.uniform(-X_LIMIT, X_LIMIT)
        ry = random.uniform(-Y_LIMIT, Y_LIMIT)
        obj_height = obj.dimensions.z
        obj.location = (rx, ry, obj_height / 2)

        # ── Assign different color to each face group ───────────
        assign_face_colors(obj)

        # ── Randomise light energies per frame ──────────────────
        key_light.data.energy  = random.uniform(600,  1000)
        fill_light.data.energy = random.uniform(150,  350)
        rim_light.data.energy  = random.uniform(250,  550)

        # Slightly shift key light position for varied shadows
        key_light.location = (
            random.uniform(-4, -2),
            random.uniform(-4, -2),
            random.uniform(5, 7)
        )

        # ── Render ──────────────────────────────────────────────
        fname = f'img_{i:05d}.jpg'
        scene.render.filepath = os.path.join(OUTPUT_DIR, fname)
        bpy.ops.render.render(write_still=True)

        # ── Project 3D object center → exact pixel coordinate ───
        px, py = world_to_pixel(scene, cam, obj.location)

        # ── Write to .txt ────────────────────────────────────────
        f.write(f"{i:<8} {fname:<20} {label:<12} {px:<10} {py}\n")
        f.flush()

        print(f"[{i+1}/{NUM_SAMPLES}] {label} at ({px}, {py})")