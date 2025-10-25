import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image
import logging

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from lib.util import partfield, generation

# Global logger
log = logging.getLogger(__name__)

def optimize_appearance(cfg, output_dir):
    log.info("Starting appearance optimization...")
    
    generation_pipeline = TrellisImageTo3DPipeline.from_pretrained(cfg.trellis_img_model_name)
    generation_pipeline.cuda()
    
    # load appearance and structure data
    path = osp.join(output_dir, 'latents', cfg.latent_name, "appearance.npz")
    data = np.load(path)
    app_feats = torch.from_numpy(data['feats']).cuda()
    app_coords = torch.from_numpy(data['coords']).cuda()
    
    struct_coords = utils3d.io.read_ply(osp.join(output_dir, 'voxels', 'struct_voxels.ply'))[0]
    struct_coords = torch.from_numpy(struct_coords).float().cuda()
    struct_coords = ((struct_coords + 0.5) * 64).long()
    
    app_image = Image.open(osp.join(output_dir, 'app_image.png')).convert('RGB')
    
    zeros = torch.zeros((struct_coords.size(0), 1), dtype=struct_coords.dtype, device=struct_coords.device)
    struct_coords = torch.cat([zeros, struct_coords], dim=1)
    
    # Load partfield planes
    path = osp.join(output_dir, 'partfield', 'part_feat_struct_mesh_zup_batch_part_plane.npy')
    struct_part_planes = torch.from_numpy(np.load(path, allow_pickle=True)).cuda()

    path = osp.join(output_dir, 'partfield', 'part_feat_app_mesh_zup_batch_part_plane.npy')
    app_part_planes = torch.from_numpy(np.load(path, allow_pickle=True)).cuda()

    app_labels, struct_labels, point_feat1, point_feat2 = partfield.cosegment_part(app_coords, app_part_planes, struct_coords, struct_part_planes, cfg.app_guidance.num_part_clusters)
        
    # Optimization Starts
    app_labels = torch.from_numpy(app_labels.flatten()).cuda()
    struct_labels = torch.from_numpy(struct_labels.flatten()).cuda()

    point_feat1 = torch.from_numpy(point_feat1).cuda()
    point_feat2 = torch.from_numpy(point_feat2).cuda()

    struct_feats_params = torch.nn.Parameter(torch.randn((struct_coords.shape[0], cfg.flow_model_in_channels)), requires_grad=True)

    param_list = [struct_feats_params]
    optimizer = torch.optim.AdamW(param_list, lr=cfg.app_guidance.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    best_loss = float('inf')
    feats = None

    image = generation_pipeline.preprocess_image(app_image)
    cond = generation_pipeline.get_cond([image])

    flow_model = generation_pipeline.models['slat_flow_model']

    sampler_params={
        "cfg_strength": cfg.app_guidance.cfg_strength,
        "cfg_interval": cfg.app_guidance.cfg_interval,
    }

    t_seq = np.linspace(1, 0, cfg.app_guidance.steps + 1)
    t_seq = cfg.app_guidance.rescale_t * t_seq / (1 + (cfg.app_guidance.rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(cfg.app_guidance.steps))

    std = torch.tensor(generation_pipeline.slat_normalization['std'])[None].cuda()
    mean = torch.tensor(generation_pipeline.slat_normalization['mean'])[None].cuda()
    
    log.info(f"Beginning guidance + flow sampling loop for {len(t_pairs)} steps...")
    for iteration, (t, t_prev) in enumerate(t_pairs):
        optimizer.zero_grad()

        # Diffusion
        struct_feats_params_clone = struct_feats_params.clone().cuda()
        noise = sp.SparseTensor(
                    feats = struct_feats_params_clone,
                    coords = struct_coords.int(),
                    ).cuda()
    
        with torch.no_grad():
            out = generation_pipeline.slat_sampler.sample_once(flow_model, noise, t, t_prev, **cond, **sampler_params)

        sample = out.pred_x_prev
        struct_feats_params.data = sample.feats

        # Optimization
        if iteration < len(t_pairs) - 1:
            app_loss, num_labels = torch.tensor(0.0, requires_grad=True).cuda(), 0.0
            for label in torch.unique(app_labels):
                app_mask = (app_labels == label)
                struct_mask = (struct_labels == label)
                
                if app_mask.sum() == 0 or struct_mask.sum() == 0:
                    continue
                
                # Appearance Loss
                cos_sim = torch.matmul(point_feat2[struct_mask], point_feat1[app_mask].T)
                cos_dist = (1 - cos_sim) / 2.
                nearest = torch.argmin(cos_dist, dim=1)
                
                matched = app_feats[app_mask][nearest]
                curr_loss = F.mse_loss(struct_feats_params[struct_mask], matched)
                
                app_loss += curr_loss
                num_labels += 1

            app_loss = cfg.app_guidance.loss_weight * (app_loss / num_labels)

            total_loss = app_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if (iteration == 0) or (iteration + 1) % cfg.log_every == 0:
                message = f"Step: {iteration}, Appearance Loss: {app_loss.item():.4f}, Total Loss: {total_loss.item():.4f}"
                log.info(message)

            if total_loss < best_loss:
                best_loss = total_loss.item()
        
        feats = struct_feats_params.detach() * std + mean

    # Decode SLAT
    log.info("Decoding output SLAT...")
    out_meshpath = osp.join(output_dir,  'out_app.glb')
    out_gspath = osp.join(output_dir,  'out_gaussian_app.mp4')
    generation.decode_slat(generation_pipeline, feats, struct_coords, out_meshpath, out_gspath)