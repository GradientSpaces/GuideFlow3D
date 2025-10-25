import os.path as osp
import numpy as np
import torch
import utils3d
import logging

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.pipelines import TrellisTextTo3DPipeline
from lib.util import generation, partfield

# Global logger
log = logging.getLogger(__name__)

def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix

def optimize_self_similarity(cfg, app_text, output_dir):
    log.info("Starting self-similarity optimization...")
    
    generation_pipeline = TrellisTextTo3DPipeline.from_pretrained(cfg.trellis_text_model_name)
    generation_pipeline.cuda()
    
    # Load Structure Data
    struct_coords = utils3d.io.read_ply(osp.join(output_dir, 'voxels', 'struct_voxels.ply'))[0]
    struct_coords = torch.from_numpy(struct_coords).float().cuda()
    struct_coords = ((struct_coords + 0.5) * 64).long()
    
    zeros = torch.zeros((struct_coords.size(0), 1), dtype=struct_coords.dtype, device=struct_coords.device)
    struct_coords = torch.cat([zeros, struct_coords], dim=1)
    
    # Load partfield planes
    path = osp.join(output_dir, "partfield", "part_feat_struct_mesh_zup_batch_part_plane.npy")
    struct_part_planes = torch.from_numpy(np.load(path, allow_pickle=True)).cuda()

    struct_labels = partfield.cluster_geoms(struct_coords, struct_part_planes, num_clusters=cfg.sim_guidance.num_part_clusters)

    # Optimization Starts...
    struct_labels = torch.from_numpy(struct_labels.flatten()).cuda()
    struct_feats_params = torch.nn.Parameter(torch.randn((struct_coords.shape[0], cfg.flow_model_in_channels)), requires_grad=True)
    
    param_list = [struct_feats_params]
    optimizer = torch.optim.AdamW(param_list, lr=cfg.sim_guidance.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    
    best_loss = float('inf')
    feats = None
    
    cond = generation_pipeline.get_cond([app_text])
    
    flow_model = generation_pipeline.models['slat_flow_model']
    sampler_params={
        "cfg_strength": cfg.sim_guidance.cfg_strength,
        "cfg_interval": cfg.sim_guidance.cfg_interval,
    }

    t_seq = np.linspace(1, 0, cfg.sim_guidance.steps + 1)
    t_seq = cfg.sim_guidance.rescale_t * t_seq / (1 + (cfg.sim_guidance.rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(cfg.sim_guidance.steps))
    
    std = torch.tensor(generation_pipeline.slat_normalization['std'])[None].cuda()
    mean = torch.tensor(generation_pipeline.slat_normalization['mean'])[None].cuda()

    log.info(f"Beginning self-similarity guidance + flow sampling loop for {len(t_pairs)} steps...")
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
        
        # Optimization - Structure Loss
        if iteration < len(t_pairs) - 1:
            labels = struct_labels.view(-1,1)
            sim = attn_cosine_sim(struct_feats_params[None, None, ...])[0]
        
            mask = (labels == labels.T).float()
            
            logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=struct_feats_params.device)
            mask = mask * logits_mask
            
            exp_sim = torch.exp(sim) * logits_mask
            numerator = (exp_sim * mask).sum(dim=1)
            denominator = exp_sim.sum(dim=1)
            
            struct_loss = -torch.log(numerator / (denominator + 1e-8))
            struct_loss = struct_loss[mask.sum(dim=1) > 0].mean()

            total_loss = cfg.sim_guidance.loss_weight * struct_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iteration == 0) or (iteration + 1) % cfg.log_every == 0:
                message = f"Step: {iteration}, Structure Loss: {struct_loss.item():.4f}, Total Loss: {total_loss.item():.4f}"
                log.info(message)
                
            if total_loss < best_loss:
                best_loss = total_loss.item()
                feats = struct_feats_params.detach() * std + mean
    
    # Decode SLAT
    log.info("Decoding output SLAT...")
    out_meshpath = osp.join(output_dir,  'out_sim.glb')
    out_gspath = osp.join(output_dir,  'out_gaussian_sim.mp4')
    generation.decode_slat(generation_pipeline, feats, struct_coords, out_meshpath, out_gspath)