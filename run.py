import os.path as osp
import gc
import trimesh
from PIL import Image
import logging as log
from omegaconf import OmegaConf
import argparse
import random
import numpy as np

import torch
from torchvision import transforms
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

import sys
sys.path.append('.')

from third_party.PartField.partfield.model_trainer_pvcnn_only_demo import Model
from lib.opt import appearance, self_similarity
from lib.util import common
import third_party.TRELLIS.trellis.models as models

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def init_args():
    parser = argparse.ArgumentParser(description='GuideFlow3D - 3D Shape Generation')
    parser.add_argument('--appearance_mesh', type=str, required=True, 
                        help='Path to appearance mesh (.glb format)')
    parser.add_argument('--structure_mesh', type=str, required=True,
                        help='Path to structure mesh (.glb format)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--appearance_image', type=str, required=True,
                        help='Path to appearance reference image')
    parser.add_argument('--appearance_text', type=str, default='',
                        help='Optional appearance text description')
    parser.add_argument('--convert_yup_to_zup', action='store_true',
                        help='Convert Y-up coordinate system to Z-up')
    return parser.parse_args()

def predict_part(obj_path, output_dir):
    log.info("Extracting PartField feature planes...")
    partfield_config = 'third_party/PartField/config.yaml'
    partfield_cfg = OmegaConf.load(partfield_config)
    
    seed_everything(partfield_cfg.seed)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    checkpoint_callbacks = [ModelCheckpoint(
        monitor="train/current_epoch",
        dirpath=partfield_cfg.output_dir,
        filename="{epoch:02d}",
        save_top_k=100,
        save_last=True,
        every_n_epochs=partfield_cfg.save_every_epoch,
        mode="max",
        verbose=True
    )]

    trainer = Trainer(devices=-1,
                      accelerator="gpu",
                      precision="16-mixed",
                      strategy=DDPStrategy(find_unused_parameters=True),
                      max_epochs=partfield_cfg.training_epochs,
                      log_every_n_steps=1,
                      limit_train_batches=3500,
                      limit_val_batches=None,
                      callbacks=checkpoint_callbacks
                     )

    partfield_model = Model(partfield_cfg, obj_path)
    output = trainer.predict(partfield_model, ckpt_path=partfield_cfg.continue_ckpt)
    part_planes, uid = output[0]
    np.save(f'{output_dir}/part_feat_{uid}_batch_part_plane.npy', part_planes)
    
    del partfield_model
    gc.collect() # Free up memory

def main():
    args = init_args()
    cfg = OmegaConf.load('config/default.yaml')

    if not args.appearance_mesh.endswith('.glb') and not args.structure_mesh.endswith('.glb'):
        log.error("Meshes must be in .glb format")
        return

    if not osp.exists(args.appearance_mesh):
        log.error(f"Appearance mesh not found: {args.appearance_mesh}")
        return

    common.ensure_dir(args.output_dir)

    # Loading Mesh and image
    log.info("Loading mesh...")
    app_mesh = trimesh.load(args.appearance_mesh, force='mesh')
    app_mesh.export(osp.join(args.output_dir, 'app_mesh.glb'))
    
    struct_mesh = trimesh.load(args.structure_mesh, force='mesh')
    struct_mesh.export(osp.join(args.output_dir, 'struct_mesh.glb'))
    
    # Load appearance image/text
    app_image = Image.open(args.appearance_image).convert('RGB')
    app_image.save(osp.join(args.output_dir, 'app_image.png'))
    app_text = args.appearance_text
    
    # # Convert Y-up to Z-up if needed
    # if args.convert_yup_to_zup:
    #     app_mesh = pointcloud.convert_mesh_yup_to_zup(app_mesh)
    #     struct_mesh = pointcloud.convert_mesh_yup_to_zup(struct_mesh)
    
    # app_mesh.export(osp.join(args.output_dir, 'app_mesh_zup.glb'))
    # struct_mesh.export(osp.join(args.output_dir, 'struct_mesh_zup.glb'))

    # # Render views for DinoV2 feature extraction
    # log.info(f"Rendering appearance mesh for {cfg.num_views} views...")
    # app_render_dir = osp.join(args.output_dir, 'app_renders')
    # common.ensure_dir(app_render_dir)
    # out_renderviews = render.render_all_views(osp.join(args.output_dir, 'app_mesh.glb'), app_render_dir, num_views=cfg.num_views)
    # if not out_renderviews:
    #     log.info("Appearance rendering failed!")
    #     return
    
    # log.info(f"Rendering structure mesh for {cfg.num_views // 10} views...")
    # struct_render_dir = osp.join(args.output_dir, 'struct_renders')
    # common.ensure_dir(struct_render_dir)
    # out_renderviews = render.render_all_views(osp.join(args.output_dir, 'struct_mesh.glb'), struct_render_dir, num_views=cfg.num_views // 10)
    # if not out_renderviews:
    #     log.info("Structure rendering failed!")

    # # Voxelise mesh
    # voxel_dir = osp.join(args.output_dir, 'voxels')
    # common.ensure_dir(voxel_dir)
    # log.info("Voxelizing mesh...")
    # pointcloud.voxelize_mesh(osp.join(app_render_dir, 'mesh.ply'), save_path=osp.join(voxel_dir, 'app_voxels.ply'))
    # pointcloud.voxelize_mesh(osp.join(struct_render_dir, 'mesh.ply'), save_path=osp.join(voxel_dir, 'struct_voxels.ply'))

    # Extract DinoV2 Features
    # log.info("Extracting DinoV2 features...")
    # dinov2_model = torch.hub.load(cfg.dinov2_repo, cfg.feature_name)
    # dinov2_model.eval().cuda()
    # transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # common.ensure_dir(osp.join(args.output_dir, 'features', cfg.feature_name))
    # generation.extract_feature(args.output_dir, dinov2_model, transform)
    # torch.cuda.empty_cache()
    
    # del dinov2_model
    # gc.collect() # Free up memory
    
    # # assert False
    
    # # Extract SLAT Latent
    # log.info("Extracting SLAT latent...")
    # encoder = models.from_pretrained(cfg.enc_pretrained).eval().cuda()
    
    # common.ensure_dir(osp.join(args.output_dir, 'latents', cfg.latent_name))
    # generation.get_latent(args.output_dir, cfg.feature_name, cfg.latent_name, encoder)

    # del encoder
    # gc.collect() # Free up memory
    
    # TODO: PartField Feature Plane Extraction
    # log.info("Extracting PartField feature planes...")

    # partfield_dir = osp.join(args.output_dir, 'partfield')
    # common.ensure_dir(partfield_dir)
    # predict_part(osp.join(args.output_dir, 'app_mesh_zup.glb'), partfield_dir)
    # predict_part(osp.join(args.output_dir, 'struct_mesh_zup.glb'), partfield_dir)

    # assert False
    
    # TODO: Appearance Optimization
    appearance.optimize_appearance(cfg, args.output_dir)

    # TODO: Self-Similarity Optimization
    # self_similarity.optimize_self_similarity(args.output_dir, app_text, cfg)

if __name__ == "__main__":
    main()