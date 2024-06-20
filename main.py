import torch
import numpy as np
import importlib
import json
import os
from PointNet2.data_utils.ScoliosisDataLoader import ScoliosisDataset
import random
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import io
# import open3d as o3d
# import faulthandler
# faulthandler.enable()
# from gui.seg_vis import visualize_point_cloud

# print("here")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/PointNet2/'
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def visualize_sequence_flow(markers_flow_points_target=[], markers_flow_points=[], frame_first=None, frame_last=None, color_first=None, color_last=None, frame_last_=[], offset=50):

    flow_pc_target = o3d.geometry.PointCloud()
    try:
        flow_pc_target.points = o3d.utility.Vector3dVector(np.array(markers_flow_points_target).reshape(-1, 3))
    # if it's a dict
    except:
        markers = []
        for frame in markers_flow_points_target:
            markers.append([marker for marker in frame.values()])
        # markers_flow_points_target = np.array([marker for marker in markers_flow_points_target[j].values() for j in range(len(markers_flow_points_target))])
        flow_pc_target.points = o3d.utility.Vector3dVector(np.array(markers).reshape(-1, 3))
    flow_pc_target.paint_uniform_color([0, 1, 0])
    
    flow_pc = o3d.geometry.PointCloud()
    try:
        flow_pc.points = o3d.utility.Vector3dVector(np.array(markers_flow_points).reshape(-1, 3))
    except:
        markers = []
        for frame in markers_flow_points:
            markers.append([marker for marker in frame.values()])
        # markers_flow_points = np.array([marker for marker in markers_flow_points.values()])
        flow_pc.points = o3d.utility.Vector3dVector(np.array(markers).reshape(-1, 3))
    flow_pc.paint_uniform_color([1, 0, 0])
    
    # frame_pc = o3d.io.read_point_cloud(sequence + 'pt6_auto_01_007367_XYZ_0.ply')
    frame_pc_first = o3d.geometry.PointCloud()
    if frame_first is not None:
        frame_pc_first.points = o3d.utility.Vector3dVector(np.array([point + [0, 0, offset] for point in frame_first]))
    if color_first is not None:
        # color_first = (color_first - np.min(color_first)) / (np.max(color_first) - np.min(color_first))
        # color_first = np.repeat(color_first, 3).reshape(-1, 3)

        # Assign color to each point in the point cloud
        frame_pc_first.colors = o3d.utility.Vector3dVector(np.array(color_first))
    else:
        frame_pc_first.paint_uniform_color([0.8, 0.8, 1])

    # frame_pc_last = o3d.io.read_point_cloud(sequence + 'pt6_auto_01_007481_XYZ_72.ply')
    frame_pc_last = o3d.geometry.PointCloud()
    if frame_last is not None:
        frame_pc_last.points = o3d.utility.Vector3dVector(np.array([point + [0, 0, offset] for point in frame_last]))
    if color_last is not None:
        # color_last = (color_last- np.min(color_last)) / (np.max(color_last) - np.min(color_last))
        # color_last= np.repeat(color_last, 3).reshape(-1, 3)

        # Assign color to each point in the point cloud
        frame_pc_last.colors = o3d.utility.Vector3dVector(np.array(color_last))
    else:
        frame_pc_last.paint_uniform_color([0.8, 1, 0.8])

    # frame_pc_last_ = o3d.geometry.PointCloud()
    # frame_pc_last_.points = o3d.utility.Vector3dVector(np.array([point + [0, 0, offset] for point in frame_last_]))
    # frame_pc_last_.paint_uniform_color([1, 0.8, 0.8])

    all_pc = frame_pc_last + frame_pc_first + flow_pc_target + flow_pc

    all_pc.rotate(all_pc.get_rotation_matrix_from_xyz((0, np.pi, np.pi / 2)))
    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(all_pc)
    # vis.update_geometry(all_pc)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image('/home/travail/ghebr/project/flownet_test')
    # vis.destroy_window()
  

    o3d.visualization.draw_geometries([all_pc])

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def find_centers(points, labels, n=9):
    centers = torch.zeros((len(points), n, 3))
    num_labels = torch.zeros((len(points), n))
    points = points[:, :, :3].cpu()
    # print(centers.shape)
    

    for j in range(len(points)):
        for i in range(len(points[j])):
            centers[j][labels[j][i]] += points[j][i]
            num_labels[j][labels[j][i]] += 1

    # print(centers.size(), num_labels.size())
    
    return (centers / num_labels[:, :, None]).tolist()

# Find anatomical segments and their middle points (estimated landmarks)for a given frame of a motion sequence
def find_segments_frame(data, net, device='cpu', landmarks=True, num_part=7):

    num_classes = 1
    num_part = num_part
    num_votes = 3

    # with open(data_path, 'rb') as f:
    #     data = json.load(f)
    points = data['points']
    colors = points[:, :, 3:]
    points_original = data['points_original']
    target = data['segments']
    # markers = data['markers_original']
    label = data['class']


    points, label = points.float().to(device), label.long().to(device)
    points = points.transpose(2, 1)

    vote_pool = torch.zeros(points.size()[0], points.size()[2], num_part).to(device)

    for _ in range(num_votes):
        seg_pred, _ = net(points, to_categorical(label, num_classes))
        # if num_part == 7:
        #     seg_pred = seg_pred[:, :, [0, 1, 2, 5, 6, 7, 8]]
        vote_pool += seg_pred
    
    points = points.transpose(2, 1).cpu().data.numpy()

    seg_pred = vote_pool / num_votes
    cur_pred_val = seg_pred.cpu().data.numpy()
    labels = np.argmax(cur_pred_val, 2)

    centers = []
    if landmarks:
        centers = find_centers(points_original, labels, n=num_part)

    return [{'labels': labels[i].tolist(), 'centers': centers[i]} for i in range(len(points))]

def find_segments_sequence(sequence, method='', publish_progress=None):

    # reproducibility
    manual_seed = 0
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2**32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(manual_seed)

    project_path = '/Users/ghebr/Desktop/MotionAIS'
    pretrained_dir = project_path + '/PointNet2/Pointnet_Pointnet2_pytorch/log/back_segmentation/pointnet2_part_seg_msg_circles/10000/checkpoints'
    # data_path = project_path + '/Data/'
    # method = 'pointnet2_part_seg_msg_circles/10000/middle_points/'

    device = 'cpu'
    save = True
    npoint = 10000
    normal = True
    batch_size = 1
    landmarks = 8
    
    # output_dir = sequence_path + '/results/'
    # input_dir = sequence_path + '/uploads/'
    # os.makedirs(output_dir, exist_ok=True)
    # output_files = os.listdir(output_dir)
    # Check if output files already exist
    outputs = []
    # if len(output_files):
    #     for output_file in output_files:
    #         output_path = output_dir + output_file
    #         with open(output_path) as f:
                # outputs.append(json.load(f))
    # else:
    # Find segments and landmarks (centers method)
    Sequence = ScoliosisDataset(data=sequence, npoints=npoint, augmentation=False, normal_channel=normal, task='seg', mode='inference')
    sequenceDataLoader = torch.utils.data.DataLoader(Sequence, batch_size=batch_size, shuffle=False, num_workers=4)
                                                    #  , worker_init_fn=seed_worker, generator=g)
    # Load the model
    num_classes = 1
    num_part = 9
    model_name = 'pointnet2_part_seg_msg'
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(9, normal_channel=normal, num_points=npoint).to(device)

    # Load the model parameters
    checkpoint = torch.load(str(pretrained_dir) + '/best_model.pth', map_location=torch.device(device))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(sequenceDataLoader), total=len(sequenceDataLoader),
                                                    smoothing=0.9):
            # Input frames' names which is used for writing the outout
            frames = data['frame']
            output_batch = find_segments_frame(data, classifier, device, num_part=num_part)
            for output in output_batch:
                outputs.append(output)
            
            # if save:
            #     for output, frame in zip(output_batch, frames):
            #         with open(output_dir + f'{frame[:-4]}.json', 'w') as f:
            #             json.dump(output, f)

            if publish_progress is not None:
                publish_progress(batch_id, len(sequenceDataLoader))
    print("Done!")
    return outputs

# Calculate the metrics
def calculate_metrics(sequence=[], path=None):
    if not len(sequence):
        if "results" in os.listdir(path):
            path = path + "/results/"
            print(path)
        frames = sorted(os.listdir(path))
        for frame in frames:
            frame_path = path + str(frame)
            with open(frame_path) as frame_file:
                sequence.append(json.load(frame_file))

    # landmarks = [1: C7, 2: ScL, 3: ScR, 4: IL, 5: IR, 6: TUp, 7: TAp, 8: TDown]
    metrics = {
        'Scapulae Angle y-axis': [], 
        'Scapulae Angle z-axis': [],
        'Scapulae Assymmetry': [],
        'Scoliosis Angle': [],
        'Pelvis Movement': [],
        }
    for i, frame in tqdm(enumerate(sequence)):
        landmarks = frame['centers']
        scap_y = np.degrees(np.arctan((landmarks[2][1] - landmarks[1][1])/(landmarks[2][0] - landmarks[1][0])))
        metrics['Scapulae Angle y-axis'].append(scap_y)
        scap_z = np.degrees(np.arctan((landmarks[2][2] - landmarks[1][2])/(landmarks[2][0] - landmarks[1][0]))) #ajouter avec z
        metrics['Scapulae Angle z-axis'].append(scap_z)

        a = (landmarks[7][0]-landmarks[0][0])/(landmarks[7][1]-landmarks[0][1])
        b = landmarks[0][0] - a*landmarks[0][1]
        x1 = landmarks[1][0]
        x2 = (a*landmarks[1][1])+b
        d1 = abs(x1 - x2)
        x3 = landmarks[2][0]
        x4 = a*landmarks[2][1]+b
        d2 = abs(x3 - x4)
        diff_d1d2 = abs(d1 - d2)
        metrics['Scapulae Assymmetry'].append(diff_d1d2)

        # movement of the pelvis in the x-axis
        pelvis_movement = (landmarks[4][0]+landmarks[3][0])/2 - landmarks[0][0]
        metrics['Pelvis Movement'].append(pelvis_movement)
        
        a = np.sqrt((landmarks[5][0]-landmarks[6][0])**2+(landmarks[5][1]-landmarks[6][1])**2)
        b = np.sqrt((landmarks[6][0]-landmarks[7][0])**2+(landmarks[6][1]-landmarks[7][1])**2)
        c = np.sqrt((landmarks[5][0]-landmarks[7][0])**2+(landmarks[5][1]-landmarks[7][1])**2)
        scoliosis_angle = 180 - np.degrees(np.arccos((a**2+b**2-c**2)/(2*a*b)))
        metrics['Scoliosis Angle'].append(scoliosis_angle)

    return metrics

def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def plot_metrics(metrics, metrics_names = [
        'Scapulae Angle y-axis', 
        'Scapulae Angle z-axis',
        'Scapulae Assymmetry',
        'Scoliosis Angle',
        'Pelvis Movement',
        ], window_size=5):
    num_metrics = len(metrics_names)
    num_cols = min(num_metrics, 3)  # Adjust the number of columns as needed (e.g., 3 columns)

    num_rows = math.ceil(num_metrics / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    for i, metric_name in enumerate(metrics_names):
        row = i // num_cols
        col = i % num_cols
        values = metrics[metric_name]

        # Apply smoothing
        if len(values) >= window_size:
            smoothed_values = moving_average(values, window_size)
            # Extend the range to match the original values for plotting
            x = range(1 + window_size//2, len(values) - window_size//2 + 1)
        else:
            smoothed_values = values
            x = range(1, len(values) + 1)

        ax = axs[row, col] if num_metrics > 1 else axs  # Handle single subplot case

        ax.plot(range(1, len(values) + 1), values, linestyle='-', color='grey', alpha=0.5, label='Original Values')
        ax.plot(x, smoothed_values, linestyle='-', color='green', label='Smoothed Values')
        
        ax.set_title(f'Metric "{metric_name}"')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        # ax.grid(True)
        ax.legend()

    # Remove empty subplots if any
    for i in range(num_metrics, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        if num_metrics > 1:
            fig.delaxes(axs[row, col])
        else:
            fig.delaxes(axs)

    plt.tight_layout()
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig)
    
    return img_bytes
        
def main(args):
    sequence = []
    path = "/Users/ghebr/Desktop/MotionAIS/ais-interface-backend/data/pt23-bending-left-free-02"
    metrics = calculate_metrics(path=path)
    # print(metrics)
    plot_metrics(metrics=metrics)

#     frames = os.listdir(path + "uploads/")
#     for frame in frames:
#         frame_path = path + "uploads/" + str(frame)
#         with open(frame_path) as frame_file:
#             sequence.append(json.load(frame_file))
#     outpath = path + "results/"
#     outputs = find_segments_sequence(sequence)
#     for output, frame in zip(outputs, frames):
#         with open(outpath + f'{frame[:-5]}.json', 'w') as f:
#             json.dump(output, f)
    
#     # reproducibility
#     manual_seed = 0
#     random.seed(manual_seed)
#     np.random.seed(manual_seed)
#     torch.manual_seed(manual_seed)
#     torch.cuda.manual_seed_all(manual_seed)
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     g = torch.Generator()
#     g.manual_seed(manual_seed)

#     project_path = '/Users/ghebr/Desktop/MotionAIS'
#     pretrained_dir = project_path + '/PointNet2/Pointnet_Pointnet2_pytorch/log/back_segmentation/pointnet2_part_seg_msg_circles/10000/checkpoints'
#     data_path = project_path + '/Data/'
#     method = 'pointnet2_part_seg_msg_circles/10000/middle_points/'
#     sequence_paths = ['Participant23_BD_Libre_Prise01']

#     device = 'cpu'
#     save = True
#     npoint = 10000
#     normal = True
#     batch_size = 1
#     landmarks = 8

    
#     # the full path to the directory with a sub-directory called input including all the input frames
#     sequence_path = args[1]
#     task = args[2] # seg, track, analyze

#     output_dir = sequence_path + '/output/' + method + '/'
#     input_dir = sequence_path + '/input/'
#     os.makedirs(output_dir, exist_ok=True)
#     output_files = os.listdir(output_dir)
#     # Check if output files already exist
#     outputs = []
#     if len(output_files):
#         for output_file in output_files:
#             output_path = output_dir + output_file
#             with open(output_path) as f:
#                 outputs.append(json.load(f))
#     else:
#         # Find segments and landmarks (centers method)
#         Sequence = ScoliosisDataset(root=input_dir, npoints=npoint, augmentation=False, normal_channel=normal, task='seg', mode='inference', num_landmarks=landmarks)
#         sequenceDataLoader = torch.utils.data.DataLoader(Sequence, batch_size=batch_size, shuffle=False, num_workers=4)
#                                                         #  , worker_init_fn=seed_worker, generator=g)
#         # Load the model
#         num_classes = 1
#         num_part = 9
#         model_name = 'pointnet2_part_seg_msg'
#         MODEL = importlib.import_module(model_name)
#         classifier = MODEL.get_model(9, normal_channel=normal, num_points=npoint).to(device)

#         # Load the model parameters
#         checkpoint = torch.load(str(pretrained_dir) + '/best_model.pth', map_location=torch.device(device))
#         classifier.load_state_dict(checkpoint['model_state_dict'])

#         with torch.no_grad():
#             for batch_id, data in tqdm(enumerate(sequenceDataLoader), total=len(sequenceDataLoader),
#                                                         smoothing=0.9):
#                 # Input frames' names which is used for writing the outout
#                 frames = data['frame']
#                 output_batch = find_segments_frame(data, classifier, device, num_part=num_part)
#                 for output in output_batch:
#                     outputs.append(output)
                
#                 if save:
#                     for output, frame in zip(output_batch, frames):
#                         with open(output_dir + f'{frame[:-4]}.json', 'w') as f:
#                             json.dump(output, f)
#         print("Done!")
#         # Option 1: Visulaize the segments
#         # Option 2: Visualize the landmark
#         # Option 3: Visualize the traajectories
#         n = len(outputs)
#         visualize_sequence_flow([outputs[i]['markers'] for i in range(n)], [outputs[i]['centers'] for i in range(n)], np.array(outputs[0]['points'])[:, :3], np.array(outputs[-1]['points'])[:, :3])
#         # visualize_point_cloud([[results[i]['markers'] for i in range(n)], [results[i]['centers'] for i in range(n)], np.array(results[0]['points'])[:, :3], np.array(results[-1]['points'])[:, :3]], colors=[[0, 1, 0], [1, 0, 0], [0.8, 0.8, 1], [0.8, 1, 0.8]], landmarks=True, offset=50, save_path=res_dir+'vis/')
#         # Option 4: Show the graphs


if __name__ == '__main__':
    main(sys.argv)
