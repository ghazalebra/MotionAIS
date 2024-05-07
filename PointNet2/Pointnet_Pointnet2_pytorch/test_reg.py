"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
# from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.MyDataLoader import BackLandmarksDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import cv2
import json
# from display_utils import automatic_crop

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def save_landmarks(landmarks, path):
    
    with open(path+'landmarks.json', 'w') as output_file:
        json.dump(landmarks, output_file)
    

def save_landmarks_dict(landmarks, path):
    landmarks_names = ['C', 'T1', 'T2', 'L2', 'G', 'D', 'IG', 'ID']
    # number of frames
    n = len(landmarks)
    landmarks_dict = {}
    for i in range(n):
        image_name = 'image' + str(i+1)
        landmarks_frame = {}
        for j, l in enumerate(landmarks_names):
            landmarks_frame[l] = landmarks[i][j]
        # print(landmarks_frame)
        landmarks_dict[image_name] = landmarks_frame

    
    with open(path+'landmarks.json', 'w') as f:
        json.dump(landmarks_dict, f)   

def draw_landmarks(markers, landmarks, frames, path):
    os.makedirs(path + 'frames_with_landmarks/', exist_ok=True)
    print(path + 'frames_with_landmarks/')
    # number of frames
    n = len(frames)
    for i in range(n):
        frame_i = np.array(frames[i])
        landmarks_i = landmarks[i]
        markers_i = markers[i]
        # print(landmarks_i)
        for landmark, marker in zip(landmarks_i, markers_i):
            # print(landmark)
            # break
            frame_i = cv2.circle(frame_i, tuple([int(landmark[0]), int(landmark[1])]),5,(0,0,255))
            frame_i = cv2.circle(frame_i, tuple([int(marker[0]), int(marker[1])]),5,(0,255,0))
        cv2.imwrite(path + 'frames_with_landmarks/frame%d.jpg' % i, frame_i)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=False, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        # logger.info(str)
        print(str)
    torch.cuda.empty_cache()

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/regression/' + args.log_dir + '/' + str(args.num_point)

    '''LOG'''
    args = parse_args()
    # logger = logging.getLogger("Model")
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(args)

    root = '/home/travail/ghebr/project/Data/test.txt'

    dataset = BackLandmarksDataset(root=root, npoints=args.num_point)
    torch.manual_seed(0)
    test_dataset = dataset
    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(test_dataset))
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # MODEL = importlib.import_module(model_name)
    # classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])

    model_name = 'pointnet2_reg'
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    #### changing the code to not load the checkpoins for the new regression layers ####
    regression_layers = []
    state_dict = classifier.state_dict()
    # print(checkpoint.keys())
    for name, param in checkpoint['model_state_dict'].items():
        if name not in regression_layers:
            # print(name)
            state_dict[name].copy_(param)
    ####################
    classifier.load_state_dict(state_dict)

    with torch.no_grad():
        test_metrics = {}
        # total_correct = 0
        # total_seen = 0
        # total_seen_class = [0 for _ in range(num_part)]
        # total_correct_class = [0 for _ in range(num_part)]
        # shape_ious = {cat: [] for cat in seg_classes.keys()}
        # seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        # for cat in seg_classes.keys():
        #     for label in seg_classes[cat]:
        #         seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        predicted_landmarks = []
        accuracy = []
        # frames = []
        # target_all = []
        predictions_path = '/home/travail/ghebr/project/Data' + '/predicted/' + str(args.num_point) + '/'
        os.makedirs(predictions_path, exist_ok=True)
        for batch_id, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):

            torch.cuda.empty_cache()
            # batchsize, num_point, _ = points.size()
            # cur_batch_size, NUM_POINT, _ = points.size()
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            # print(target.shape)


            # expanding the channel dimension to match the architecture
            zeros = torch.zeros(points.shape).cuda()

            # merging vector of ones with 'c' along dimension 1 (columns)
            points = torch.cat((points, zeros), 1)
            # print(points.shape)

            ### Landmark Detection ###

            mse_loss = torch.nn.MSELoss()
            # print('batch ' + str(batch_id))

            cls_label = torch.eye(16)[0]
            # Set Abstraction layers
            cls_label = cls_label.repeat(len(points), 1, 1).cuda()

            predicted_landmarks_batch, _ = classifier(points, cls_label)
            # print(predicted_landmarks_batch)
            predicted_landmarks_batch_list = predicted_landmarks_batch.cpu().tolist()
            targets_batch_list = target.cpu().tolist()
            # print(predicted_landmarks_batch_list)
            # print(predicted_landmarks_batch_np.shape)
            for landmarks_frame in predicted_landmarks_batch_list:
                predicted_landmarks.append(landmarks_frame)
            # for target_frame in targets_batch_list:
            #     target_all.append(target_frame)
            # predicted_landmarks += predicted_landmarks_batch_list
            # predicted_landmarks.append(predicted_landmarks_batch_np)
            # frame_list = frame.cpu().tolist()
            # for frame_i in frame_list:
            #     frames.append(frame_i)

            # frames += frame_list
            
            # print(predicted_landmarks.shape)
            # break
            # predicted_landmarks_2D = predicted_landmarks[:][:][:2]
            # target_landmarks = target.cpu().data.numpy()
            # print(target_landmarks.shape)
            # break
            
            loss = mse_loss(predicted_landmarks_batch, target)
            
            accuracy.append(loss.detach().cpu())
        
       
            # print(loss)

            # print(type(frame))
            # frame = np.asarray(frame.cpu()).astype(float)
            # print(frame.shape)
            # predicted_landmarks = np.asarray(target.cpu()).astype(float)
            # print(type(predicted_landmarks[0][0]))

            torch.cuda.empty_cache()
            # for j in range(len(frame)):
            #     # print(batch_id, j)
            #     im_with_predicted = frame[j]
            #     for point in predicted_landmarks[j]:
            #         # for point in Points:
            #         # print(tuple([int(point[0]), int(point[1])]))
            #         im_with_predicted = cv2.circle(im_with_predicted, tuple([int(point[0]), int(point[1])]),5,(0,0, 255))
            #     # im_with_predicted = cv2.drawKeypoints(im_with_predicted, predicted_landmarks[j], np.array([]), (255, 0, 0),
            #                                 # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #     # print('saving the image to ' + predictions_path)
            #     idx = batch_id*args.batch_size + j
            #     # print(idx)
            #     cv2.imwrite(predictions_path + 'annotated_frame%d.jpg' % idx, im_with_predicted)
                # cv2.imshow('predictions', frame[j]) 
                # cv2.waitKey(0) # waits until a key is pressed
                # cv2.destroyAllWindows() 

            ##########################

            # vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            # for _ in range(args.num_votes):
            #     seg_pred, _ = classifier(points, cls_label)
            #     vote_pool += seg_pred

        # saving the landmarks
        
        # print(np.array(predicted_landmarks).shape)
        

    #         seg_pred = vote_pool / args.num_votes
    #         cur_pred_val = seg_pred.cpu().data.numpy()
    #         cur_pred_val_logits = cur_pred_val
    #         cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
    #         target = target.cpu().data.numpy()

    #         for i in range(cur_batch_size):
    #             cat = seg_label_to_cat[target[i, 0]]
    #             logits = cur_pred_val_logits[i, :, :]
    #             cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

    #         correct = np.sum(cur_pred_val == target)
    #         total_correct += correct
    #         total_seen += (cur_batch_size * NUM_POINT)

    #         for l in range(num_part):
    #             total_seen_class[l] += np.sum(target == l)
    #             total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

    #         for i in range(cur_batch_size):
    #             segp = cur_pred_val[i, :]
    #             segl = target[i, :]
    #             cat = seg_label_to_cat[segl[0]]
    #             part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
    #             for l in seg_classes[cat]:
    #                 if (np.sum(segl == l) == 0) and (
    #                         np.sum(segp == l) == 0):  # part is not present, no prediction as well
    #                     part_ious[l - seg_classes[cat][0]] = 1.0
    #                 else:
    #                     part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
    #                         np.sum((segl == l) | (segp == l)))
    #             shape_ious[cat].append(np.mean(part_ious))

    #     all_shape_ious = []
    #     for cat in shape_ious.keys():
    #         for iou in shape_ious[cat]:
    #             all_shape_ious.append(iou)
    #         shape_ious[cat] = np.mean(shape_ious[cat])
    #     mean_shape_ious = np.mean(list(shape_ious.values()))
    #     test_metrics['accuracy'] = total_correct / float(total_seen)
    #     test_metrics['class_avg_accuracy'] = np.mean(
    #         np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    #     for cat in sorted(shape_ious.keys()):
    #         log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
    #     test_metrics['class_avg_iou'] = mean_shape_ious
    #     test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
    accuracy_epoch = np.mean(accuracy)
    save_landmarks(predicted_landmarks, predictions_path)
    # draw_landmarks(target_all, predicted_landmarks, frames, predictions_path)

    log_string('Accuracy is: %.5f' % loss)


if __name__ == '__main__':
    args = parse_args()
    main(args)