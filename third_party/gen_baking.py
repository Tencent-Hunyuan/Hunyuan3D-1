import os, sys, time
import shutil
from typing import List, Optional
from iopath.common.file_io import PathManager

import cv2
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

import trimesh
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.common.datatypes import Device
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    camera_position_from_spherical_angles,
    BlendParams,
)


def erode_mask(src_mask, p=1 / 20.0):
    monoMaskImage = cv2.split(src_mask)[0]
    br = cv2.boundingRect(monoMaskImage)
    k = int(min(br[2], br[3]) * p)
    kernel = np.ones((k, k), dtype=np.uint8)
    dst_mask = cv2.erode(src_mask, kernel, 1)
    return dst_mask

def load_objs_as_meshes_fast(
    verts,
    faces,
    aux,
    device: Optional[Device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
):
    tex = None
    tex_maps = aux.texture_images
    if tex_maps is not None and len(tex_maps) > 0:
        verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        faces_uvs = faces.textures_idx.to(device)  # (F, 3)
        image = list(tex_maps.values())[0].to(device)[None]
        tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
    mesh = Meshes( verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex)
    return mesh


def get_triangle_to_triangle(tri_1, tri_2, img_refined):
    r1 = cv2.boundingRect(tri_1)
    r2 = cv2.boundingRect(tri_2)
    
    tri_1_cropped = []
    tri_2_cropped = []
    for i in range(0, 3):
        tri_1_cropped.append(((tri_1[i][1] - r1[1]), (tri_1[i][0] - r1[0])))
        tri_2_cropped.append(((tri_2[i][1] - r2[1]), (tri_2[i][0] - r2[0])))

    trans = cv2.getAffineTransform(np.float32(tri_1_cropped), np.float32(tri_2_cropped))

    img_1_cropped = np.float32(img_refined[r1[0]:r1[0] + r1[2], r1[1]:r1[1] + r1[3]])
    
    mask = np.zeros((r2[2], r2[3], 3), dtype=np.float32)
    
    cv2.fillConvexPoly(mask, np.int32(tri_2_cropped), (1.0, 1.0, 1.0), 16, 0)
    
    img_2_cropped = cv2.warpAffine(
        img_1_cropped, trans, (r2[3], r2[2]), None, 
        flags = cv2.INTER_LINEAR,
        borderMode = cv2.BORDER_REFLECT_101
    )
    return mask, img_2_cropped, r2


def back_projection(
    obj_file, 
    init_texture_file, 
    front_view_file, 
    dst_dir, 
    render_resolution=512, 
    uv_resolution=600, 
    normalThreshold=0.3, # 0.3 
    rgb_thresh=820, # 520
    views=None, 
    camera_dist=1.5, 
    erode_scale=1/100.0, 
    device="cuda:0"
):
    # obj_file: obj with uv
    # init_texture_file： uv texture image

    os.makedirs(dst_dir, exist_ok=True)

    if isinstance(front_view_file, str):
        src = np.array(Image.open(front_view_file).convert("RGB"))
    elif isinstance(front_view_file, Image.Image):
        src = np.array(front_view_file.convert("RGB"))
    else:
        raise "need file_path or pil"
    
    image_size = (render_resolution, render_resolution)

    init_texture = Image.open(init_texture_file)
    init_texture = init_texture.convert("RGB")
    # init_texture = init_texture.resize((uv_resolution, uv_resolution))
    init_texture = np.array(init_texture).astype(np.float32)  
    
    print("load obj", obj_file)
    verts, faces, aux = load_obj(obj_file, device=device)
    mesh = load_objs_as_meshes_fast(verts, faces, aux, device=device)


    t0 = time.time()
    verts_uvs = aux.verts_uvs
    triangle_uvs = verts_uvs[faces.textures_idx]
    triangle_uvs = torch.cat([
        ((1 - triangle_uvs[..., 1]) * uv_resolution).unsqueeze(2),
        (triangle_uvs[..., 0] * uv_resolution).unsqueeze(2),
    ], dim=-1)
    triangle_uvs = np.clip(np.round(np.float32(triangle_uvs.cpu())).astype(np.int64), 0, uv_resolution-1)

    
    R0, T0 = look_at_view_transform(camera_dist, views[0][0], views[0][1])

    cameras = FoVPerspectiveCameras(device=device, R=R0, T=T0, fov=49.1)
    
    camera_normal = camera_position_from_spherical_angles(1, views[0][0], views[0][1]).to(device)
    screen_coords = cameras.transform_points_screen(verts, image_size=image_size)[:, :2]
    screen_coords = torch.cat([screen_coords[..., 1, None], screen_coords[..., 0, None]], dim=-1)
    triangle_screen_coords = np.round(np.float32(screen_coords[faces.verts_idx].cpu())) # numpy.ndarray (90000, 3, 2)
    triangle_screen_coords = np.clip(triangle_screen_coords.astype(np.int64), 0, render_resolution-1)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings= RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            ),
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights= AmbientLights(device=device),
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )
    )

    dst = renderer(mesh)
    dst = (dst[..., :3] * 255).squeeze(0).cpu().numpy().astype(np.uint8)

    src_mask = np.ones((src.shape[0], src.shape[1]), dst.dtype)
    ids = np.where(dst.sum(-1) > 253 * 3)
    ids2 = np.where(src.sum(-1) > 250 * 3)
    src_mask[ids[0], ids[1]] = 0
    src_mask[ids2[0], ids2[1]] = 0
    src_mask = (src_mask > 0).astype(np.uint8) * 255
    
    monoMaskImage = cv2.split(src_mask)[0] # reducing the mask to a monochrome
    br = cv2.boundingRect(monoMaskImage) # bounding rect (x,y,width,height)
    center = (br[0] + br[2] // 2, br[1] + br[3] // 2)
 
    # seamlessClone
    try:
        images = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE) # better 
        # images = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)
    except Exception as err:
        print(f"\n\n Warning seamlessClone error: {err} \n\n")
        images = src

    Image.fromarray(src_mask).save(os.path.join(dst_dir, 'mask.jpeg'))
    Image.fromarray(src).save(os.path.join(dst_dir, 'src.jpeg'))
    Image.fromarray(dst).save(os.path.join(dst_dir, 'dst.jpeg'))
    Image.fromarray(images).save(os.path.join(dst_dir, 'blend.jpeg'))

    fragments_scaled = renderer.rasterizer(mesh)  # pytorch3d.renderer.mesh.rasterizer.Fragments
    faces_covered = fragments_scaled.pix_to_face.unique()[1:] # torch.Tensor torch.Size([30025])
    face_normals = mesh.faces_normals_packed().to(device) # torch.Tensor torch.Size([90000, 3]) cuda:0

    # faces:              pytorch3d.io.obj_io.Faces
    # faces.textures_idx: torch.Tensor torch.Size([90000, 3])
    # verts_uvs:          torch.Tensor torch.Size([49554, 2])
    triangle_uvs = verts_uvs[faces.textures_idx]
    triangle_uvs = [
        ((1 - triangle_uvs[..., 1]) * uv_resolution).unsqueeze(2),
        (triangle_uvs[..., 0] * uv_resolution).unsqueeze(2),
    ]
    triangle_uvs = torch.cat(triangle_uvs, dim=-1) # numpy.ndarray (90000, 3, 2)
    triangle_uvs = np.clip(np.round(np.float32(triangle_uvs.cpu())).astype(np.int64), 0, uv_resolution-1)
    
    t0 = time.time()
    
    SOFT_NORM = True # process big angle-diff face, true:flase? coeff:skip
    
    for k in faces_covered:
        # todo: accelerate this for-loop
        # if cosine between face-camera is too low, skip current face baking
        face_normal = face_normals[k]
        cosine = torch.sum((face_normal * camera_normal) ** 2)
        if not SOFT_NORM and cosine < normalThreshold: continue

        # if coord in screen out of subject, skip current face baking
        out_of_subject = src_mask[triangle_screen_coords[k][0][0], triangle_screen_coords[k][0][1]]==0
        if out_of_subject: continue
            
        coeff, img_2_cropped, r2 = get_triangle_to_triangle(triangle_screen_coords[k], triangle_uvs[k], images)
        
        # if color difference between new-old, skip current face baking
        err = np.abs(init_texture[r2[0]:r2[0]+r2[2], r2[1]:r2[1]+r2[3]]- img_2_cropped)
        err = (err * coeff).sum(-1)
        
        # print(err.shape, np.max(err))
        if (np.max(err) > rgb_thresh): continue
        
        color_for_debug = None
        # if (np.max(err) > 400): color_for_debug = [255, 0, 0]
        # if (np.max(err) > 450): color_for_debug = [0, 255, 0]
        # if (np.max(err) > 500): color_for_debug = [0, 0, 255]

        coeff = coeff.clip(0, 1)
        
        if SOFT_NORM:
            coeff *= ((cosine.detach().cpu().numpy() - normalThreshold) / normalThreshold).clip(0,1)

        coeff *= (((rgb_thresh - err[...,None]) / rgb_thresh)**0.4).clip(0,1)

        if color_for_debug is None:
            init_texture[r2[0]:r2[0]+r2[2], r2[1]:r2[1]+r2[3]] = \
                init_texture[r2[0]:r2[0]+r2[2], r2[1]:r2[1]+r2[3]] * ((1.0,1.0,1.0)-coeff) + img_2_cropped * coeff
        else:
            init_texture[r2[0]:r2[0]+r2[2], r2[1]:r2[1]+r2[3]] = color_for_debug

    print(f'View baking time: {time.time() - t0}')

    bake_dir = os.path.join(dst_dir, 'bake')
    os.makedirs(bake_dir, exist_ok=True)
    shutil.copy2(obj_file, bake_dir)
    
    textute_img = Image.fromarray(init_texture.astype(np.uint8))
    textute_img.save(os.path.join(bake_dir, init_texture_file.split("/")[-1]))
    
    mtl_dir = obj_file.replace('.obj', '.mtl')
    if not os.path.exists(mtl_dir): mtl_dir = obj_file.replace("mesh.obj" ,"material.mtl")
    if not os.path.exists(mtl_dir): mtl_dir = obj_file.replace("mesh.obj" ,"texture.mtl")
    if not os.path.exists(mtl_dir):
        raise FileNotFoundError(f"No material file found for {obj_file}")
    shutil.copy2(mtl_dir, bake_dir)

    # convert .obj to .glb file
    new_obj_pth = os.path.join(bake_dir, obj_file.split('/')[-1])
    new_glb_path = new_obj_pth.replace('.obj', '.glb')
    mesh = trimesh.load_mesh(new_obj_pth)
    mesh.export(new_glb_path, file_type='glb')
