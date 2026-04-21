import os
import sys

import numpy as np
import open3d as o3d
import pytorch3d
import torch
import torch.nn as nn
from imageio.v2 import imread
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from scipy import ndimage
from torchvision import transforms

from config import get_sensat_cfg
from mesh_init.mesh_init_linear_solver import init_mesh_sparse
from mesh_init.mesh_renderer import mesh_render_depth, render_mesh_texture
from model.models import VoxMeshHead
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified
from vis.vis import (
    pseudo_color_map,
    pseudo_color_map_sparse,
    texture_mesh,
    texture_mesh_vertices,
)


torch.cuda.empty_cache()

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())


PATH = "demo_data"
DEPTH_SCALE = 100
IMAGE_SIZE = 512
FOCAL_LENGTH = -2
FOCAL_LENGTH_TEXTURE = 2
NUM_MESH_VERTICES = 1024
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RGB_FILE = f"{PATH}/RGB.png"
SPARSE_DEPTH_FILE = f"{PATH}/SparseDepth.png"
GT_DEPTH_FILE = f"{PATH}/GT_Depth.png"
GT_SEM_FILE = f"{PATH}/GT_Semantics.png"

SEG_MODEL_PATH = "checkpoints/deeplab/model_best_semantic.tar"
SEG_CFG_FILE = "Sensat_basic.yaml"
REFINE_MODEL_PATH = "checkpoints/Sem_Dice/model_best_depth.tar"
REFINE_CFG_FILE = os.path.join(REFINE_MODEL_PATH, "..", "Sensat_basic.yaml")


## TerrainMesh: Metric-Semantic Terrain Reconstruction from Aerial Images Using Joint 2D-3D Learning
### Load and visualize the demo data
# calculate depth scale / min and max
gt_depth = imread(GT_DEPTH_FILE) / DEPTH_SCALE
depth_min, depth_max = np.percentile(gt_depth[gt_depth > 0], [5, 95])
print(f"Depth min: {depth_min:.2f} m, Depth max: {depth_max:.2f} m")

rgb = imread(RGB_FILE)
sparsedepth = imread(SPARSE_DEPTH_FILE) / DEPTH_SCALE
gt_depth = imread(GT_DEPTH_FILE) / DEPTH_SCALE
gt_sem = imread(GT_SEM_FILE)


### Depth sanity checks
gt = imread(GT_DEPTH_FILE)
sp = imread(SPARSE_DEPTH_FILE)

print("GT dtype/min/max:", gt.dtype, gt.min(), gt.max())
print("SP dtype/min/max:", sp.dtype, sp.min(), sp.max())

print("GT nonzero:", np.count_nonzero(gt))
print("SP nonzero:", np.count_nonzero(sp))

gt_m = gt / DEPTH_SCALE
sp_m = sp / DEPTH_SCALE

mask = sp > 0
print("Sparse points:", mask.sum())

if mask.sum() > 0:
    print("Sparse depth range in meters:", sp_m[mask].min(), sp_m[mask].max())
    print("GT depth at sparse points:", gt_m[mask].min(), gt_m[mask].max())


### Get the 2D segmentation
seg_cfg = get_sensat_cfg()
seg_cfg.merge_from_file(SEG_CFG_FILE)

# Load a trained deeplabv3 2D semantic segmentation model
model_2dseg = torch.hub.load("pytorch/vision:v0.8.0", "deeplabv3_resnet50", pretrained=True)
model_2dseg.classifier[4] = nn.Conv2d(
    256,
    seg_cfg.MODEL.DEEPLAB.NUM_CLASSES,
    kernel_size=1,
    stride=1,
)
model_2dseg.to(DEVICE)
checkpoint = torch.load(SEG_MODEL_PATH, map_location=DEVICE, weights_only=True)
model_2dseg.load_state_dict(checkpoint["model_state_dict"])
model_2dseg.eval()

# The input is an RGB image
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(rgb)
input_img = input_tensor.unsqueeze(0).to(DEVICE)
pred_semantic_features = model_2dseg(input_img)["out"]
pred_semantic = pred_semantic_features.detach().max(dim=1)[1].cpu().numpy()[0, ::]
pred_semantic_vis = convert_class_to_rgb_sensat_simplified(pred_semantic)


## Mesh initialization using sparse depth measurements
# Initialize the mesh using only the sparse depth
init_mesh_vertices, init_mesh_faces = init_mesh_sparse(
    sparsedepth,
    NUM_MESH_VERTICES,
    w_laplacian=0.5,
    device=DEVICE,
)
init_mesh_vertices_tensor = torch.tensor(init_mesh_vertices, dtype=torch.float32, device=DEVICE)
init_mesh_faces_tensor = torch.tensor(init_mesh_faces, dtype=torch.int64, device=DEVICE)
init_mesh = Meshes(verts=[init_mesh_vertices_tensor], faces=[init_mesh_faces_tensor])

init_mesh_depth = mesh_render_depth(
    init_mesh,
    image_size=IMAGE_SIZE,
    focal_length=FOCAL_LENGTH,
    device=DEVICE,
)
init_mesh_depth_vis = pseudo_color_map(init_mesh_depth, depth_min, depth_max)


## Mesh refinement
# Load a trained 2D-3D model for mesh refinement
refine_cfg = get_sensat_cfg()
refine_cfg.merge_from_file(REFINE_CFG_FILE)
model = VoxMeshHead(refine_cfg)
checkpoint = torch.load(REFINE_MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)

# Combine the 2D inputs, including the RGB, the initial mesh rendered depth and the sparse depth EDT.
rgb_tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(DEVICE)
init_mesh_depth_tensor = transforms.ToTensor()(init_mesh_depth / DEPTH_SCALE).unsqueeze(0).to(DEVICE)
sparsedepth_mask = (sparsedepth <= 0) * 1
depth_edt = ndimage.distance_transform_edt(sparsedepth_mask)
depth_edt_tensor = transforms.ToTensor()(depth_edt).unsqueeze(0).to(DEVICE)
input_img = torch.cat((rgb_tensor, init_mesh_depth_tensor, depth_edt_tensor), dim=1).to(torch.float)

# Normalized the mesh
init_mesh_scale = torch.mean(init_mesh_vertices_tensor[:, 2])
init_mesh_vertices_norm_tensor = init_mesh_vertices_tensor / init_mesh_scale
init_mesh = Meshes(verts=[init_mesh_vertices_norm_tensor], faces=[init_mesh_faces_tensor])

# The model takes in the concatenated 2D inputs, the initial mesh and the 2D semantic segmentation
refine_mesh = model(input_img, init_mesh, pred_semantic_features)
refine_mesh = refine_mesh[1].scale_verts(init_mesh_scale.unsqueeze(0).to(DEVICE))


## Render refined outputs
refine_mesh_semantic, refine_mesh_depth = render_mesh_texture(
    refine_mesh,
    image_size=IMAGE_SIZE,
    focal_length=FOCAL_LENGTH,
    device=DEVICE,
)
refine_mesh_semantic_vis = convert_class_to_rgb_sensat_simplified(refine_mesh_semantic)
refine_mesh_depth_vis = pseudo_color_map(refine_mesh_depth, depth_min, depth_max)

refine_mesh_vertices, refine_mesh_faces = refine_mesh.get_mesh_verts_faces(0)
refine_mesh_vertices = refine_mesh_vertices.detach().cpu().numpy()
refine_mesh_faces = refine_mesh_faces.detach().cpu().numpy()
refine_mesh_height_color = texture_mesh_vertices(refine_mesh_vertices, depth_min, depth_max)
rotate_matrix = np.array(
    [
        [1, 0, 0],
        [0, -np.sqrt(3) / 2, 1 / 2],
        [0, -1 / 2, -np.sqrt(3) / 2],
    ]
)


## Save the refined mesh with different textures (Color, Elevation, Semantics)
os.makedirs(f"{PATH}/refined_mesh", exist_ok=True)

print("A import open3d ok", flush=True)

verts = np.ascontiguousarray(refine_mesh_vertices, dtype=np.float64)
faces = np.ascontiguousarray(refine_mesh_faces, dtype=np.int32)

print("B verts/faces cast ok", verts.shape, verts.dtype, faces.shape, faces.dtype, flush=True)
print("C finite verts:", np.isfinite(verts).all(), flush=True)
print("D faces min/max:", faces.min(), faces.max(), "nverts:", len(verts), flush=True)

assert np.isfinite(verts).all()
assert faces.ndim == 2 and faces.shape[1] == 3
assert faces.min() >= 0
assert faces.max() < len(verts)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

rgb_tex = texture_mesh(rgb, mesh, FOCAL_LENGTH_TEXTURE)
ok = o3d.io.write_triangle_mesh(f"{PATH}/refined_mesh/test_rgb.obj", rgb_tex)

depth_tex = texture_mesh(refine_mesh_depth_vis, mesh, FOCAL_LENGTH_TEXTURE)
ok = o3d.io.write_triangle_mesh(f"{PATH}/refined_mesh/test_depth.obj", depth_tex)

sem_tex = texture_mesh(refine_mesh_semantic_vis, mesh, FOCAL_LENGTH_TEXTURE)
ok = o3d.io.write_triangle_mesh(f"{PATH}/refined_mesh/test_sem.obj", sem_tex)

print("bad verts:", np.isnan(verts).sum(), np.isinf(verts).sum())
print("bad faces <0:", (faces < 0).sum())
print("bad faces >= nverts:", (faces >= len(verts)).sum())