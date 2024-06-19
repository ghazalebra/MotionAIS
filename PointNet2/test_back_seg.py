# python test_back_seg.py --log_dir pointnet2_part_seg_msg --normal --save_pred --reg --landmarks 6
import argparse
import os
from PointNet2.data_utils.ScoliosisDataLoader import ScoliosisDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


# reproducability
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

# for part segmentation

seg_classes = {'Back': [0, 1, 2, 3, 4, 5, 6, 7, 8]}
# for semantic segmentation I have to change it but I don't know how
# classes = []
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

def euclidean_dist(points1, points2):
    return (points1 - points2).pow(2).sum(-1).sqrt()



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=10000, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--save_pred', action='store_true', default=False, help='save the predicted segment labels to json files')
    parser.add_argument('--reg', action='store_true', default=False, help='add regression')
    parser.add_argument('--landmarks', type=int, default=8, help='number of landmarks')
    parser.add_argument('--data', type=str, default='test', help='data split')
    parser.add_argument('--sequence', type=str, default='', help='sequence path')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    

    '''HYPER PARAMETER'''
    pretrained_dir = 'log/back_segmentation/pretrained/' + args.log_dir
    Path(pretrained_dir).mkdir(exist_ok=True)
    print(pretrained_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/back_segmentation/' + args.log_dir
    Path(experiment_dir).mkdir(exist_ok=True)
    print(experiment_dir)
    
    if args.npoint != 2048:
        experiment_dir = 'log/back_segmentation/' + args.log_dir + '/' + str(args.npoint)
    print(experiment_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    
    root = '/home/travail/ghebr/Data'
    if args.sequence != '':
        root = root + args.sequence + 'data/'

    TEST_DATASET = ScoliosisDataset(root=root, npoints=args.npoint, augmentation=False, normal_channel=args.normal, task='seg', num_landmarks=args.landmarks, split=args.data, mode='inference')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1
    num_part = args.landmarks + 1

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print(model_name)
    MODEL = importlib.import_module(model_name)
    if model_name=='pointnet2_part_seg_msg':
        classifier = MODEL.get_model(num_part, normal_channel=args.normal, num_points=args.npoint).cuda()
    elif model_name=='pointnet2_sem_seg':
        classifier = MODEL.get_model(num_part).cuda()
    else:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
        print(str(experiment_dir) + '/checkpoints/best_model.pth')
        load_checkpoint(classifier, pretrained_path=str(experiment_dir) + '/checkpoints/best_model.pth')
    try:
        # fine_tuned pretrained model
        print('loading the fine_tuned model')
        print(str(experiment_dir) + '/checkpoints/best_model.pth')
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        state_dict = classifier.state_dict()
        for name, param in checkpoint['model_state_dict'].items():
            try:
                state_dict[name].copy_(param)
            except:
                print(name, state_dict[name].shape, param.shape)
        classifier.load_state_dict(state_dict)
        
        # classifier.load_state_dict(checkpoint['model_state_dict'])
    except:
        # pretrained model
        # classifier = classifier.apply(weights_init)
        checkpoint_pretrained = torch.load(str(pretrained_dir) + '/best_model.pth')
        modified_layers = ['conv2.weight', 'conv2.bias']
        pretrained_state_dict = classifier.state_dict()
        for name, param in checkpoint_pretrained['model_state_dict'].items():
            if name not in modified_layers:
                try:
                    pretrained_state_dict[name].copy_(param)
                except:
                    if name in pretrained_state_dict.keys():
                        print(name + ' is in the dict')
        log_string('Loading the pretrained segmentation model')
        classifier.load_state_dict(pretrained_state_dict)
        

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat][:num_part]:
                seg_label_to_cat[label] = cat
        error_reg = []
        error_reg_mse = []
        classifier = classifier.eval()
        for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):

            points = data['points']
            target = data['segments']
            weight = data['weights']
            markers = data['markers']
            label = data['class']
            frame_name = data['frame']
            scale_norm = data['scale']
            # print(scale_norm)

            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target, markers, scale_norm = points.float().cuda(), label.long().cuda(), target.long().cuda(), markers.cuda(), scale_norm.cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            # expanding the channel dimension to match the architecture for normal channel
            # print(args.normal)
            if args.normal:
                zeros = torch.zeros(points.shape).cuda()
                # the number of input channels is 9 for semantic segmentation. For now, I'm just appending 0s to the end
                if model_name == 'pointnet2_sem_seg':
                    points = torch.cat((points, zeros), 1)
            else:
                features = points[:, 3:, :]
                points = points[:, :3, :]
                
            # print(points.size())
            
            # break
            for _ in range(args.num_votes):
                if model_name=='pointnet2_part_seg_msg':
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                elif model_name=='pointnet2_sem_seg':
                    seg_pred, _ = classifier(points)
                # if args.landmarks < 8:
                #     landmarks_to_test = [0, 1, 2, 5, 6, 7, 8]
                #     seg_pred = seg_pred[:, :, landmarks_to_test]
                vote_pool += seg_pred
            # print('ran the model!')

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            # print(seg_label_to_cat)
            for i in range(cur_batch_size):
                # cat = seg_label_to_cat[target[i, 0]]
                # print(cat)
                # cat = cat[:num_part]
                logits = cur_pred_val_logits[i, :, :]
                # print(seg_classes[cat])
                cur_pred_val[i, :] = np.argmax(logits, 1)
            # print(cur_pred_val)
            # cur_pred_ca torch.Tensor
            # print(cur_pred_val.size)
            # coordinates = coordinates.cpu().data.numpy()
            points = points.transpose(2, 1)
            if args.reg:
                # print(num_part)
                pred_seg_centers_batch = []
                pred_choice_reshaped = torch.Tensor(cur_pred_val).view(cur_batch_size, NUM_POINT, 1).cuda()
                # points = points[:, :, :3]
                # features = points[:, :, 3:]
                for k in range(cur_batch_size):
                    points_k = points[k][:, :3].cuda()
                    pred_seg_centers = torch.zeros((num_part-1, 3)).cuda()
                    pred_seg_num = torch.zeros(num_part-1).cuda()
                    for j, point in enumerate(points_k):
                        point = point.cuda()
                        if pred_choice_reshaped[k][j] != num_part-1:
                            pred_seg_centers[int(pred_choice_reshaped[k][j])] += point
                            pred_seg_num[int(pred_choice_reshaped[k][j])] += 1
                        
                    pred_seg_centers_batch.append(pred_seg_centers / pred_seg_num.unsqueeze(-1))
                seg_centers_tensor = torch.Tensor((len(pred_seg_centers_batch), num_part-1, 3)).cuda()
                torch.cat(pred_seg_centers_batch, out=seg_centers_tensor)
                seg_centers_tensor = seg_centers_tensor.view(len(pred_seg_centers_batch), num_part-1, 3)
                # print(seg_centers_tensor)
                scale_norm = scale_norm.unsqueeze(-1).unsqueeze(-1)
                # break
                # print(scale_norm)
                
                # seg_centers_tensor = seg_centers_tensor

            # MSE loss of the centers
            if args.reg:
                mse_loss = torch.nn.MSELoss(reduction='none').cuda()
                landmarks = [i for i in range(num_part)]
                # if args.landmarks < num_part - 1:
                #     landmarks = [0, 1, 2, 5, 6, 7]
                # print(seg_centers_tensor[:, landmarks, :].size(), markers.size(), scale_norm.size())
                mse_error_batch = mse_loss(seg_centers_tensor, markers).mean(dim=-1).view(cur_batch_size, -1)
                euc_dist_error_batch = euclidean_dist(markers*scale_norm, seg_centers_tensor*scale_norm)
                # print(markers*scale_norm)
                # print(seg_centers_tensor*scale_norm)
                # print(mse_error_batch)
                # break
                # print(mse_error_batch)
                error_reg_mse += list(np.array(mse_error_batch.detach().cpu()))
                # print(error_reg_mse)
                error_reg += list(np.array(euc_dist_error_batch.detach().cpu()))

            # print('here')

            # saving the predictions
            if args.save_pred:
                # coordinates = coordinates.transpose(2, 1)

                for i in range(len(cur_pred_val)):
                    prediction_dir = Path(experiment_dir + '/predictions')
                    # print(prediction_dir)
                    prediction_dir.mkdir(exist_ok=True)
                    result = {'pred': cur_pred_val[i].tolist(), 'target': target[i].tolist(), 'points': points[i].tolist(), 
                    'markers': markers[i].tolist(), 'coordinates': [], 'centers': []}
                    if args.reg:
                        result['centers'] = seg_centers_tensor[i].tolist()
                    with open(str(prediction_dir) + f'/{frame_name[i]}.json', 'w') as f:
                        json.dump(result, f) 
                
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in landmarks:
                # if l == 5:
                #     print((np.sum((cur_pred_val == l) & (target == l))))
                # print(l)
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

            # if batch_id == 2:
            #     # print([list(np.array(error_reg_mse_i)) for error_reg_mse_i in error_reg_mse])
                
            #     print(error_reg)
            # break

        if args.reg:
            # print(np.array(error_reg_mse).shape)
            landmarks_error_reg_mse = np.mean(error_reg_mse, axis=0)
            mean_error_reg_mse = np.mean(landmarks_error_reg_mse)
            landmarks_error_reg = np.mean(error_reg, axis=0)
            mean_error_reg = np.mean(landmarks_error_reg)
            # print(np.array(error_reg).shape)
            # test_metrics['regression_error'] = error_reg 
            test_metrics['landmarks_error'] = landmarks_error_reg
            test_metrics['landmarks_error_mean'] = mean_error_reg
            test_metrics['mse_centers_error'] = mean_error_reg_mse
        
            

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    if args.reg:
        log_string('MSE error of the centers: %.5f' % test_metrics['mse_centers_error'])
        log_string('Average displacement of centers over all segments: %.5f' % test_metrics['landmarks_error_mean'])
        log_string('Average displacement of centers: ' + ', '.join([str(x) for x in test_metrics['landmarks_error']]))
    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
