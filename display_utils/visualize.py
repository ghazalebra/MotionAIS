import open3d as o3d
import numpy as np
import json
import cv2
from tqdm import tqdm


# the image dimensions
w = 1936
h = 1176
header_size = 512

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def read_single_xyz_raw_file(file_path):

    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h, w, 3))
    # first index is horizental, the seocnd index in inverted
    data_array_modified = data_array.reshape((-1, 3))

    return data_array_modified

# modified it to work with point clouds
def remove_bg(pc):
    keep =  np.where(pc[:, 2]>0)
    pc_removed_0 = pc[keep[0]]
    z_body = np.median(pc_removed_0[:, 2])
    z_bg = np.max(pc_removed_0[:, 2])
    z_front = np.min(pc_removed_0[:, 2])
    print(z_front)
    non_bg = np.where(pc_removed_0[:, 2] < (z_body + z_bg) / 2.)
    pc_no_bg = pc_removed_0[non_bg]
    return pc_no_bg

path = '/home/travail/ghebr/project/Data/Participant3/BG/Libre/Prise01/xyz_removed_bg/pt3_BG_libre_01_001649_XYZ_35.ply'
intensity_path = '/home/travail/ghebr/project/Data/Participant1/autocorrection/Prise01/intensity/auto_01_014762_I_0.jpg'

pc = o3d.io.read_point_cloud(path)

view_status_file = open('view_status.json')
view_status = json.load(view_status_file)
trajectory = view_status['trajectory']
trajectory = trajectory[0]

# removing the outliers
def display_inlier_outlier(cloud, ind, trajectory):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                    #   field_of_view=trajectory['field_of_view'],
                                  zoom=trajectory['zoom'],
                                  front=trajectory['front'],
                                  lookat=trajectory['lookat'],
                                  up=trajectory['up'])






o3d.visualization.draw_geometries([pc],
                                #   field_of_view=trajectory['field_of_view'],
                                  zoom=trajectory['zoom'],
                                  front=trajectory['front'],
                                  lookat=trajectory['lookat'],
                                  up=trajectory['up']
                                )

