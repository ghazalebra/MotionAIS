import torch
import numpy as np
import importlib
import json
import os
from PointNet2.Pointnet_Pointnet2_pytorch.data_utils.ScoliosisDataLoader import ScoliosisDataset
import random
from tqdm import tqdm
import sys
import open3d as o3d
# from gui.seg_vis import visualize_point_cloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/PointNet2/Pointnet_Pointnet2_pytorch/'
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
    markers = data['markers_original']
    label = data['class']


    points, target, markers, label = points.float().to(device), target.long().to(device), markers.to(device), label.long().to(device)
    points = points.transpose(2, 1)

    vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).to(device)

    for _ in range(num_votes):
        seg_pred, l3_points, l3_xyz, l2_points, l2_xyz, l1_points, l1_xyz = net(points, to_categorical(label, num_classes))
        # if num_part == 7:
        #     seg_pred = seg_pred[:, :, [0, 1, 2, 5, 6, 7, 8]]
        vote_pool += seg_pred
    
    print(l3_points.size(), l3_xyz.size(), l2_points.size(), l2_xyz.size(), l1_points.size(), l1_xyz.size())


    points = points.transpose(2, 1).cpu().data.numpy()
    l3_points = l3_points.transpose(2, 1).cpu().data.numpy()
    l2_points = l2_points.transpose(2, 1).cpu().data.numpy()
    l1_points = l1_points.transpose(2, 1).cpu().data.numpy()

    # print(l1_points.size())
    # print(points_original.size())
    
    seg_pred = vote_pool / num_votes
    cur_pred_val = seg_pred.cpu().data.numpy()
    # target = target.cpu().data.numpy()
    labels = np.argmax(cur_pred_val, 2)
    # print('labels ' + str(labels.shape))
    # print('points ' + str(points.size()))
    centers = []
    if landmarks:
        centers = find_centers(points_original, labels, n=num_part)
    # print(l1_points[0])
    # return labels
    return [{'labels': labels[i].tolist(), 'points': points_original[i].tolist(), 'colors': colors[i].tolist(), 'centers': centers[i], 'markers': markers[i].tolist(), 'targets': target[i].tolist(), 'l3': l3_points[i].tolist(), 'points_norm': points[i].tolist(), 'l2': l2_points[i].tolist(), 'l1': l1_points[i].tolist()} for i in range(len(points))]


def main(args):

    # reproducibility
    manual_seed = 0
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(manual_seed)

    pretrained_dir = '/home/travail/ghebr/project/PointNet2/Pointnet_Pointnet2_pytorch/log/back_segmentation/pointnet2_part_seg_msg/10000/checkpoints'
    root = '/home/travail/ghebr/Data/'
    method = 'pointnet2_part_seg_msg_circles/10000/middle_points/'
    sequence_paths = ['Participant23/autocorrection/Prise02/data/']
    # sequence_paths = ['Participant23/autocorrection/Prise02/data/',
    #                 #    'Participant24/autocorrection/Prise02/data/', 'Participant24/autocorrection/Prise03/data/', 'Participant24/autocorrection/Prise04/data/'
    #                 ]
    device = 'cuda'
    save = True
    npoint = 10000
    normal = True
    batch_size = 1
    landmarks = 8

    for sequence_path in sequence_paths:
        data_path = root + sequence_path
        # loading the data
        Sequence = ScoliosisDataset(root=data_path, npoints=npoint, augmentation=False, normal_channel=normal, task='seg', mode='inference', num_landmarks=landmarks)
        sequenceDataLoader = torch.utils.data.DataLoader(Sequence, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)


        # the model
        num_classes = 1
        num_part = 9
        model_name = 'pointnet2_part_seg_msg'
        MODEL = importlib.import_module(model_name)
        classifier = MODEL.get_model(9, normal_channel=normal, num_points=npoint).to(device)

        # loading the checkpoints
        checkpoint = torch.load(str(pretrained_dir) + '/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])

        # frames = os.listdir(data_path)
        # segments = []
        # for frame in frames:
        #     segments.append(find_segments_frame(frame, classifier, device))

        with torch.no_grad():
            # print('hereeeee')
            results = []
            for batch_id, data in tqdm(enumerate(sequenceDataLoader), total=len(sequenceDataLoader),
                                                        smoothing=0.9):
                
                frames = data['frame']
                # print(len(frames))
                # print(data['points'])
                res_batch = find_segments_frame(data, classifier, device, num_part=num_part)
                for res in res_batch:
                    # print(res.keys())
                    results.append(res)
                # print(len(res))
                # results.append(res)
                # break
                
            
                if save:
                    res_dir = './seg_results/' + method + sequence_path[:-6].replace('/', '_') + '/'
                    # print(res_dir)
                    os.makedirs(res_dir, exist_ok=True)
                    for res, frame in zip(res_batch, frames):
                        # print(data_i)
                        # frame = data_i['frame']
                        with open(res_dir + f'{frame[:-4]}.json', 'w') as f:
                            json.dump(res, f)
            # print(len(results))
            # print([res['markers'] for res in results])
            # print(results['points'][0][:][:])
            n = len(results)
            # print(n)
            # visualize_sequence_flow([results[i]['markers'] for i in range(n)], [results[i]['centers'] for i in range(n)], np.array(results[0]['points'])[:, :3], np.array(results[-1]['points'])[:, :3])
            # visualize_point_cloud([[results[i]['markers'] for i in range(n)], [results[i]['centers'] for i in range(n)], np.array(results[0]['points'])[:, :3], np.array(results[-1]['points'])[:, :3]], colors=[[0, 1, 0], [1, 0, 0], [0.8, 0.8, 1], [0.8, 1, 0.8]], landmarks=True, offset=50, save_path=res_dir+'vis/')


if __name__ == '__main__':
    args = [] 
    main(args)
