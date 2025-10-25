import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

import torch
import torch.nn.functional as F

def get_voxel_partfeats(voxel_coords, part_planes):
    voxel_coords = ((voxel_coords[:, 1:] + 0.5) / 64 - 0.5).cpu().numpy()
    bbmin = voxel_coords.min(0)
    bbmax = voxel_coords.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.9 / (bbmax - bbmin).max()
    voxel_coords = (voxel_coords - center) * scale

    tensor_vertices = torch.from_numpy(voxel_coords).unsqueeze(0).reshape(1, -1, 3).cuda().to(torch.float16)
    part_feats = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C
    part_feats = part_feats.cpu().numpy().reshape(-1, 448)
    
    return part_feats

def sample_triplane_feat(feature_triplane, normalized_pos):
    '''
        normalized_pos [-1, 1]
    '''
    tri_plane = torch.unbind(feature_triplane, dim=1)

    x_feat = F.grid_sample(
        tri_plane[0],
        torch.cat(
            [normalized_pos[:, :, 0:1], normalized_pos[:, :, 1:2]],
            dim=-1).unsqueeze(dim=1), padding_mode='border',
        align_corners=True)
    y_feat = F.grid_sample(
        tri_plane[1],
        torch.cat(
            [normalized_pos[:, :, 1:2], normalized_pos[:, :, 2:3]],
            dim=-1).unsqueeze(dim=1), padding_mode='border',
        align_corners=True)

    z_feat = F.grid_sample(
        tri_plane[2],
        torch.cat(
            [normalized_pos[:, :, 0:1], normalized_pos[:, :, 2:3]],
            dim=-1).unsqueeze(dim=1), padding_mode='border',
        align_corners=True)
    final_feat = (x_feat + y_feat + z_feat)
    final_feat = final_feat.squeeze(dim=2).permute(0, 2, 1)  # 32dimension
    return final_feat

def cosegment_part(app_coords, app_part_planes, struct_coords, struct_part_planes, num_clusters=30):
    struct_partfield_feats = get_voxel_partfeats(struct_coords, struct_part_planes)
    app_partfield_feats = get_voxel_partfeats(app_coords, app_part_planes)
    
    point_feat1 = app_partfield_feats
    point_feat2 = struct_partfield_feats
    
    point_feat1 = point_feat1 / np.linalg.norm(point_feat1, axis=-1, keepdims=True)
    point_feat2 = point_feat2 / np.linalg.norm(point_feat2, axis=-1, keepdims=True)
    
    clustering1 = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(point_feat1)
    # Get feature means per cluster
    feature_means1 = []
    for j in range(num_clusters):
        all_cluster_feat = point_feat1[clustering1.labels_==j]
        mean_feat = np.mean(all_cluster_feat, axis=0)
        feature_means1.append(mean_feat)
    
    labels1 = clustering1.labels_

    feature_means1 = np.array(feature_means1)
    tree = KDTree(feature_means1)
    
    init_mode = np.array(feature_means1)
    
    point_feat2 = point_feat2 / np.linalg.norm(point_feat2, axis=-1, keepdims=True)
    clustering2 = KMeans(n_clusters=num_clusters, random_state=0, init=init_mode).fit(point_feat2)

    ### Get feature means per cluster
    feature_means2 = []
    for j in range(num_clusters):
        all_cluster_feat = point_feat2[clustering2.labels_==j]
        mean_feat = np.mean(all_cluster_feat, axis=0)
        feature_means2.append(mean_feat)

    feature_means2 = np.array(feature_means2)
    _, nn_idx = tree.query(feature_means2, k=1)
    relabelled_2 = nn_idx[clustering2.labels_]
    
    return labels1, relabelled_2, point_feat1, point_feat2

def cluster_geoms(struct_coords, struct_part_planes, num_clusters=10):
    struct_partfield_feats = get_voxel_partfeats(struct_coords, struct_part_planes)
    
    point_feat = struct_partfield_feats
    point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
    
    
    clustering = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(point_feat)
    # Get feature means per cluster
    feature_means = []
    for j in range(num_clusters):
        all_cluster_feat = point_feat[clustering.labels_==j]
        mean_feat = np.mean(all_cluster_feat, axis=0)
        feature_means.append(mean_feat)
    
    labels = clustering.labels_
    return labels