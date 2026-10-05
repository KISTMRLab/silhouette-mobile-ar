"""Optional Blender generator of toy-like training renders with exact per-object masks.

Run headless (Blender 4.2+; checked with 5.2):

    blender -b --factory-startup -P scripts/blender_synthetic.py -- --out outputs/blender --count 200
    blender -b --factory-startup -P scripts/blender_synthetic.py -- --out outputs/blender --models models.json

Without ``--models`` it builds procedural toy figures (``toy-bear``, ``toy-rabbit``)
from primitives; they are authored here and released as CC0. ``models.json`` lists
your own free-to-distribute assets::

    [{"path": "zebra.glb", "class": "zebra", "license": "CC0-1.0", "source": "https://...", "author": "..."}]

Entries without ``license`` and ``source`` are refused. Each render writes
``<id>.png`` (RGBA, transparent background), ``<id>_obj<k>.png`` (alpha = visible
pixels of object k, others held out) and ``<id>.json``; ``metadata.json`` records
classes and asset provenance. Then composite over DTD textures with
``silhouette-seg prepare-synthetic --renders <out> --textures <dtd>/images --out <dataset>``.
"""
import json
import math
import random
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector

PROCEDURAL = ("toy-bear", "toy-rabbit")


def parse_args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render toy-like objects with exact masks")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--models", help="JSON list of {path, class, license, source}")
    parser.add_argument("--samples", type=int, default=16)
    return parser.parse_args(argv)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    if getattr(mat, "node_tree", None):
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Base Color"].default_value = (*color, 1.0)
                node.inputs["Roughness"].default_value = .85
    return mat


def sphere(name, collection, location, scale, mat):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=1.0)
    bm.to_mesh(mesh)
    bm.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    obj.location, obj.scale = location, scale
    collection.objects.link(obj)
    return obj


def procedural_toy(kind, index, rng):
    """Toy-like figure from ellipsoids; height ~0.2-0.35 m before scaling."""
    collection = bpy.data.collections.new(f"{kind}-{index}")
    bpy.context.scene.collection.children.link(collection)
    base = [rng.uniform(.25, .9) for _ in range(3)] if kind == "toy-bear" else [rng.uniform(.75, 1.0)] * 3
    if kind == "toy-bear":
        base = [base[0] * .6 + .2, base[1] * .4 + .1, base[2] * .25]
    body, accent = material(f"{kind}-{index}-body", base), material(f"{kind}-{index}-accent", [min(1, c * 1.3 + .1) for c in base])
    parts = [sphere("body", collection, (0, 0, .11), (.075, .065, .095), body),
             sphere("head", collection, (0, 0, .235), (.06, .055, .055), body)]
    if kind == "toy-bear":
        parts += [sphere("ear", collection, (s * .045, 0, .285), (.022, .012, .022), body) for s in (-1, 1)]
        parts += [sphere("snout", collection, (0, -.05, .225), (.025, .018, .02), accent)]
        parts += [sphere("arm", collection, (s * .075, -.01, .13), (.022, .022, .05), body) for s in (-1, 1)]
        parts += [sphere("leg", collection, (s * .04, -.04, .03), (.03, .04, .028), body) for s in (-1, 1)]
    else:
        parts += [sphere("ear", collection, (s * .022, 0, .33), (.014, .01, .065), body) for s in (-1, 1)]
        parts += [sphere("tail", collection, (0, .07, .07), (.02, .02, .02), accent)]
        parts += [sphere("foot", collection, (s * .035, -.05, .015), (.022, .045, .015), accent) for s in (-1, 1)]
    return collection, parts


def imported_toy(entry, index):
    collection = bpy.data.collections.new(f"{entry['class']}-{index}")
    bpy.context.scene.collection.children.link(collection)
    before = set(bpy.data.objects)
    path = str(entry["path"])
    if path.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif path.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"Unsupported model format: {path}")
    parts = [obj for obj in set(bpy.data.objects) - before]
    for obj in parts:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
    meshes = [obj for obj in parts if obj.type == "MESH"]
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    scale = .3 / max(1e-6, high.z - low.z)
    centre = (low + high) / 2
    for obj in parts:
        if obj.parent is None:
            obj.location = (obj.location - Vector((centre.x, centre.y, low.z))) * scale
            obj.scale = obj.scale * scale
    return collection, meshes


def place(parts, location, yaw, scale):
    for obj in parts:
        if obj.parent is None:
            local = obj.location.copy()
            rotated = Vector((local.x * math.cos(yaw) - local.y * math.sin(yaw), local.x * math.sin(yaw) + local.y * math.cos(yaw), local.z))
            obj.location = location + rotated * scale
            obj.rotation_euler.z += yaw
            obj.scale = obj.scale * scale


def look_at(obj, target):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure(scene, size, samples):
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    scene.render.engine = next(e for e in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES") if e in engines)
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (.5, .5, .5)


def render(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(args.models).read_text(encoding="utf-8")) if args.models else []
    for entry in entries:
        if not entry.get("license") or not entry.get("source"):
            raise SystemExit(f"Model entry lacks license/source provenance: {entry}")
        entry["path"] = str((Path(args.models).parent / entry["path"]).resolve())
    classes = sorted({entry["class"] for entry in entries}) if entries else list(PROCEDURAL)
    assets = ([{k: entry.get(k) for k in ("path", "class", "license", "source", "author")} for entry in entries] if entries else
              [{"name": f"procedural {name}", "class": name, "license": "CC0-1.0", "author": "scripts/blender_synthetic.py"} for name in PROCEDURAL])
    scene = bpy.context.scene
    for index in range(args.count):
        clear_scene()
        for collection in list(bpy.data.collections):
            bpy.data.collections.remove(collection)
        configure(scene, args.size, args.samples)
        toys = []
        for k in range(rng.randint(1, args.max_objects)):
            if entries:
                entry = rng.choice(entries)
                collection, parts = imported_toy(entry, k)
                label = entry["class"]
            else:
                label = rng.choice(PROCEDURAL)
                collection, parts = procedural_toy(label, k, rng)
            angle = 2 * math.pi * k / args.max_objects + rng.uniform(-.4, .4)
            radius = 0 if k == 0 else rng.uniform(.2, .35)
            place(parts, Vector((radius * math.cos(angle), radius * math.sin(angle), 0)), rng.uniform(0, 2 * math.pi), rng.uniform(.8, 1.3))
            toys.append((label, parts))
        camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
        scene.collection.objects.link(camera)
        scene.camera = camera
        distance, elevation, azimuth = rng.uniform(.8, 1.5), math.radians(rng.uniform(8, 55)), rng.uniform(0, 2 * math.pi)
        camera.location = Vector((distance * math.cos(elevation) * math.cos(azimuth), distance * math.cos(elevation) * math.sin(azimuth), distance * math.sin(elevation)))
        look_at(camera, Vector((rng.uniform(-.05, .05), rng.uniform(-.05, .05), .1)))
        camera.data.lens = rng.uniform(22, 35)
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        sun.data.energy = rng.uniform(2, 5)
        sun.rotation_euler = (rng.uniform(.2, 1.0), 0, rng.uniform(0, 2 * math.pi))
        scene.collection.objects.link(sun)
        stem = f"render-{index:05d}"
        render(scene, out / f"{stem}.png")
        objects = []
        for k, (label, parts) in enumerate(toys):
            others = [obj for j, (_, other) in enumerate(toys) if j != k for obj in other]
            for obj in others:
                obj.is_holdout = True
            render(scene, out / f"{stem}_obj{k}.png")
            for obj in others:
                obj.is_holdout = False
            objects.append({"class": label, "mask": f"{stem}_obj{k}.png"})
        (out / f"{stem}.json").write_text(json.dumps({"image": f"{stem}.png", "objects": objects, "camera_distance": distance}, indent=2))
    (out / "metadata.json").write_text(json.dumps({"classes": classes, "assets": assets, "generator": {"script": "scripts/blender_synthetic.py",
                                                    "blender": bpy.app.version_string, "seed": args.seed, "count": args.count}}, indent=2))
    print(f"wrote {args.count} renders with exact masks to {out}")


main()
