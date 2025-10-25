import torch
import lightning.pytorch as pl
import torch.nn as nn
import os
import trimesh
import numpy as np

from torch.utils.data import DataLoader

from third_party.PartField.partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
from third_party.PartField.partfield.model.triplane import TriplaneTransformer, get_grid_coord #, sample_from_planes, Voxel2Triplane
from third_party.PartField.partfield.model.model_utils import VanillaMLP
from third_party.PartField.partfield.dataloader import Demo_Dataset

class Model(pl.LightningModule):
    def __init__(self, cfg, obj_path):
        super().__init__()
        self.obj_path = obj_path

        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=cfg.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(input_dim=64,
                                      output_dim=1, 
                                      out_activation="tanh", 
                                      n_neurons=64, #64
                                      n_hidden_layers=6) #6
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device="cuda",
                shape_min=-1, 
                shape_length=2,
                use_2d_feat=self.use_2d_feat) #.cuda()
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(input_dim=64,
                                output_dim=192, 
                                out_activation="GELU", 
                                n_neurons=64, #64
                                n_hidden_layers=6) #6   
        
    def predict_dataloader(self):
        dataset = Demo_Dataset(self.obj_path)
        dataloader = DataLoader(dataset, 
                            num_workers=self.cfg.dataset.val_num_workers,
                            batch_size=self.cfg.dataset.val_batch_size,
                            shuffle=False, 
                            pin_memory=True,
                            drop_last=False)
        
        return dataloader            

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        N = batch['pc'].shape[0]
        
        assert N == 1

        pc_feat = self.pvcnn(batch['pc'], batch['pc'])

        planes = pc_feat
        planes = self.triplane_transformer(planes)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
        
        def sample_points(vertices, faces, n_point_per_face):
            # Generate random barycentric coordinates
            # borrowed from Kaolin https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/mesh/trianglemesh.py#L43
            n_f = faces.shape[0]
            u = torch.sqrt(torch.rand((n_f, n_point_per_face, 1),
                                        device=vertices.device,
                                        dtype=vertices.dtype))
            v = torch.rand((n_f, n_point_per_face, 1),
                            device=vertices.device,
                            dtype=vertices.dtype)
            w0 = 1 - u
            w1 = u * (1 - v)
            w2 = u * v

            face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
            face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
            face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
            points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
            return points

        def sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face):
            n_sample_each = self.cfg.n_sample_each # we iterate over this to avoid OOM
            n_v = tensor_vertices.shape[1]
            n_sample = n_v // n_sample_each + 1
            all_sample = []
            for i_sample in range(n_sample):
                sampled_feature = sample_triplane_feat(part_planes, tensor_vertices[:, i_sample * n_sample_each: i_sample * n_sample_each + n_sample_each,])
                assert sampled_feature.shape[1] % n_point_per_face == 0
                sampled_feature = sampled_feature.reshape(1, -1, n_point_per_face, sampled_feature.shape[-1])
                sampled_feature = torch.mean(sampled_feature, axis=-2)
                all_sample.append(sampled_feature)
            return torch.cat(all_sample, dim=1)

        part_planes = part_planes.cpu().numpy()
        return part_planes, batch['uid'][0]