import os
import json
import torch
import numpy as np
from PIL import Image
import utils3d
import imageio
import torch.nn.functional as F

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.utils import render_utils, postprocessing_utils

def get_data(model_dir, view):
    image_path = os.path.join(model_dir, view['file_path'])
    image = Image.open(image_path)
    image = image.resize((518, 518), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255
    image = image[:, :, :3] * image[:, :, 3:]
    image = torch.from_numpy(image).permute(2, 0, 1).float()

    c2w = torch.tensor(view['transform_matrix'])
    c2w[:3, 1:3] *= -1
    extrinsics = torch.inverse(c2w)
    fov = view['camera_angle_x']
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

    return {
        'image': image,
        'extrinsics': extrinsics,
        'intrinsics': intrinsics
    }

@torch.no_grad()
def extract_feature(output_dir, dinov2_model, transform, n_patch=518 // 14, batch_size=8, feature_name='dinov2_vitl14_reg'):
    dinov2_model.eval().cuda()

    with open(os.path.join(output_dir, 'app_renders', 'transforms.json'), 'r') as f:
        metadata = json.load(f)

    frames = metadata['frames']
    data = []

    for view in frames:
        datum = get_data(os.path.join(output_dir, 'app_renders'), view)
        datum['image'] = transform(datum['image'])
        data.append(datum)
    
    positions = utils3d.io.read_ply(os.path.join(output_dir, 'voxels', 'app_voxels.ply'))[0]
    positions = torch.from_numpy(positions).float().cuda()
    indices = ((positions + 0.5) * 64).long()
    assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
    
    n_views = len(data)
    N = positions.shape[0]
    pack = {
        'indices': indices.cpu().numpy().astype(np.uint8),
    }
    
    patchtokens_lst = []
    uv_lst = []
    
    with torch.no_grad():
        for i in range(0, n_views, batch_size):
            batch_data = data[i:i+batch_size]
            bs = len(batch_data)
            batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
            batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
            batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
            features = dinov2_model(batch_images, is_training=True)
            uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
            patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
            patchtokens_lst.append(patchtokens)
            uv_lst.append(uv)
    
    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)
    
    pack['patchtokens'] = F.grid_sample(
                    patchtokens.type(torch.float16),
                    uv.unsqueeze(1).type(torch.float16),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
    
    assert not torch.isnan(patchtokens.type(torch.float16)).any(), "NaNs in patchtokens"
    assert not np.isnan(pack['patchtokens']).any(), "NaNs in pack patchtokens"
    assert not torch.isnan(uv.unsqueeze(1).type(torch.float16)).any(), "NaNs in uv"
    
    pack['patchtokens'] = np.mean(pack['patchtokens'], axis=0).astype(np.float16)
    
    save_path = os.path.join(output_dir, 'features', feature_name, 'appearance.npz')
    np.savez_compressed(save_path, **pack)
    
    del patchtokens
    del pack
    
@torch.no_grad()
def get_latent(output_dir, feature_name, latent_name, encoder):
    feats = np.load(os.path.join(output_dir, 'features', feature_name, 'appearance.npz'))
    feats = sp.SparseTensor(
        feats = torch.from_numpy(feats['patchtokens']).type(torch.float32),
        coords = torch.cat([
            torch.zeros(feats['patchtokens'].shape[0], 1).int(),
            torch.from_numpy(feats['indices']).int(),
        ], dim=1),
    ).cuda()
    latent = encoder(feats, sample_posterior=False)
    assert torch.isfinite(latent.feats).all(), "Non-finite latent"
    pack = {
        'feats': latent.feats.cpu().numpy().astype(np.float32),
        'coords': latent.coords[:, :].cpu().numpy().astype(np.uint8),
    }
    
    save_path = os.path.join(output_dir, 'latents', latent_name, 'appearance.npz')
    np.savez_compressed(save_path, **pack)
    
    del latent
    del pack

def decode_slat(generation_pipeline, feats, coords, out_meshpath, out_gspath):
    # Decode Output SLAT
    slat = sp.SparseTensor(
            feats = feats.float(),
            coords = coords.int(),
        ).cuda()            
    formats = ['mesh', 'gaussian']
    with torch.no_grad():
        outputs = generation_pipeline.decode_slat(slat, formats)

    _, mesh_textured = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    # Optional parameters
                    simplify=0.95,          # Ratio of triangles to remove in the simplification process
                    texture_size=1024,      # Size of the texture used for the GLB
                    verbose=False,          # Print logs
                )
    mesh_textured.export(out_meshpath)

    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0], bg_color=[255, 255, 255])['color']
    imageio.mimsave(out_gspath, video, fps=30)