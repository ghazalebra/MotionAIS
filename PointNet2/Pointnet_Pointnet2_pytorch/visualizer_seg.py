import open3d as o3d
import numpy as np
import json

def visualize_point_cloud(ply_file, json_file):
    # Read the PLY file
    cloud = o3d.io.read_point_cloud(ply_file)

    # Load the segmentation labels from JSON
    with open(json_file, 'r') as file:
        labels = json.load(file)

    # Extract label information
    point_ids = np.array(cloud.colors)  # Assuming point IDs are stored as RGB values
    label_ids = np.array(labels['labels'])

    # Create a colormap for the labels
    unique_labels = np.unique(label_ids)
    colormap = plt.cm.get_cmap('plasma', len(unique_labels))

    # Assign colors based on the labels
    colors = np.zeros((len(point_ids), 3))
    for i, label in enumerate(unique_labels):
        mask = label_ids == label
        colors[mask] = colormap(i)[:3]

    # Assign the colors to the point cloud
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the colored point cloud
    o3d.visualization.draw_geometries([cloud])

visualize_point_cloud('')