import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PointNet2.data_utils.ScoliosisDataLoader import ScoliosisDataset 
from torch.utils.tensorboard import SummaryWriter
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# to ensure reproducibility
manual_seed = 0
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

# pass this to the dataloaders
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # print(classname)
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        # print(classname)
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_cls_reg', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch Size during training')
    parser.add_argument('--epoch', default=50, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW or or Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--no_scheduler', action='store_true', default=False, help='don\'t use lr scheduler')
    parser.add_argument('--class_weights', action='store_true', default=False, help='use class weights for training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use CPU')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)


    torch.cuda.empty_cache()

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    '''CREATE DIR'''
    log_dir = args.log_dir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('regression')
    exp_dir.mkdir(exist_ok=True)
    if log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(log_dir)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(str(args.npoint))
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

        
    writer = SummaryWriter()

    # loading the dataset
    root = '/home/travail/ghebr/Data/'


    TRAIN_DATASET = ScoliosisDataset(root=root, npoints=args.npoint, augmentation=False, normal_channel=args.normal, class_weights=args.class_weights, split='train')
    VALID_DATASET = ScoliosisDataset(root=root, npoints=args.npoint, augmentation=False, normal_channel=args.normal, class_weights=args.class_weights, split='valid')

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, 
    drop_last=True, worker_init_fn=seed_worker, generator=g,)
    validDataLoader = torch.utils.data.DataLoader(VALID_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10,
    worker_init_fn=seed_worker,
    generator=g)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(VALID_DATASET))

    # num_classes = 1
    # num_out = 8

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(normal_channel=True).cuda()
    
    criterion = torch.nn.MSELoss().cuda()
    # I'm not sure about this one cause i don't know what it does
    classifier.apply(inplace_relu)

    # loading the pretrained weights
    pretrained_model_path = './log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'
    checkpoint_pretrained = torch.load(pretrained_model_path)



    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        global_epoch = checkpoint['epoch']
        best_loss = checkpoint['test_loss']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Loading fine-tuned pretrained model')
    except:
        log_string('No existing model, starting training from scratch (pre-pretrained model)...')
        start_epoch = 0
        global_epoch = 0
        best_loss = 10000000 
        regression_layers = ['fc_reg1'
        , 'bn_reg1', 'drop_reg1', 'fc_reg2'
        , 'bn_reg2', 'drop_reg2', 'fc_reg3'
        ]
        pretrained_state_dict = classifier.state_dict()
        for name, param in checkpoint_pretrained['model_state_dict'].items():
            if name not in regression_layers:
                try:
                    pretrained_state_dict[name].copy_(param)
                except:
                    if name in pretrained_state_dict.keys():
                        print(name + ' is in the dict')
        classifier = classifier.apply(weights_init)
        classifier.load_state_dict(pretrained_state_dict)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.1)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size



    train_losses = []
    test_losses = []

    # print(best_loss)
    # print(checkpoints_dir)

    for epoch in range(start_epoch, args.epoch):
        
        torch.cuda.empty_cache()

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        if args.no_scheduler:
            lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        log_string('Learning rate:%f' % optimizer.param_groups[0]['lr'])
        
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        accuracy = []
        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            
            # if target is None:
            #     # print('here')
            #     continue
            optimizer.zero_grad()

            points = data['points']
            target = data['markers']

            points = points.data.numpy()
            markers = target.data.numpy()

            points = provider.random_point_dropout(points)
            features = points[:, :, 3:]
            # cause the markers should remain the same in relation to the points
            points_and_markers = np.concatenate((points[:, :, 0:3], markers), axis=1)
            points_and_markers = provider.random_scale_point_cloud(points_and_markers)
            points_and_markers = provider.shift_point_cloud(points_and_markers)
            
            points = torch.Tensor(np.concatenate((points_and_markers[:, :-8, :], features), axis=2))
            target = torch.Tensor(points_and_markers[:, -8:, :])
            points = points.transpose(2, 1)
            # should I transpose the markers too? I don't think so! 

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            

            # cls_label = torch.eye(16)[0]
            # cls_label = cls_label.repeat(len(points), 1, 1).cuda()
            
            # adding fake input channels
            if args.normal:
                zeros = torch.zeros(points[:, :3, :].shape).cuda()
                points = torch.cat((points[:, :3, :], zeros), 1)
            else:
                points = points[:, :3, :]

            landmarks_pred, trans_feat = classifier(points)
            # landmark_pred = landmarks_pred.contiguous().view(-1, num_part)
            # target = target.view(-1, 1)[:, 0]
            # pred_choice = landmarks_pred.data.max(1)[1]

            # correct = pred_choice.eq(target.data).cpu().sum()
            # mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            # print(landmarks_pred.shape, target.shape)
            loss = criterion(landmarks_pred, target)
            loss.backward()
            optimizer.step()

            accuracy.append(loss.detach().cpu())
        
        if not args.no_scheduler:
            scheduler.step()
        
        loss_epoch = np.mean(accuracy)
        writer.add_scalar(f"{exp_dir}/Loss/train", loss_epoch, epoch)
        # train_losses.append(accuracy_epoch)
        log_string('Train Loss (MSE) is: %.5f' % loss_epoch)

        test_accuracy = []
        with torch.no_grad():

            test_metrics = {}

            classifier = classifier.eval()
            for batch_id, data in tqdm(enumerate(validDataLoader), total=len(validDataLoader),
                                                        smoothing=0.9):
                # if target is None: 
                #     continue
                # batchsize, num_point, _ = points.size()
                # cur_batch_size, NUM_POINT, _ = points.size()
                points = data['points']
                target = data['markers']
                # points, target = points.float().cuda(), target.float().cuda()
                points = points.transpose(2, 1)

                if not args.use_cpu:
                    points, target = points.cuda(), target.cuda()

                # print(target)
                # break
                # uncomment for testing with normal channels
                # expanding the channel dimension to match the architecture
                
                # merging vector of ones with 'c' along dimension 1 (columns)
                if args.normal:
                    zeros = torch.zeros(points[:, :3, :].shape).cuda()
                    points = torch.cat((points[:, :3, :], zeros), 1)
                else:
                    points = points[:, :3, :]
                # print(points.shape)

                ### Landmark Detection ###
                mse_loss = torch.nn.MSELoss()

                # cls_label = torch.eye(16)[0]
                # cls_label = cls_label.repeat(len(points), 1, 1).cuda()

                predicted_landmarks, _ = classifier(points)
                # print(predicted_landmarks.shape)
                # break
                # predicted_landmarks_2D = predicted_landmarks[:][:][:2]
                # target_landmarks = target.cpu().data.numpy()
                # print(target_landmarks.shape)
                # break
                loss = criterion(predicted_landmarks, target)
                test_accuracy.append(loss.detach().cpu())
                # print(loss)
            
            loss_epoch_valid = np.mean(test_accuracy)
            writer.add_scalar(f"{exp_dir}/Loss/validation", loss_epoch_valid, epoch)

            writer.flush()
            
            test_metrics['mse-loss'] = loss_epoch_valid
            
            log_string('Validation Loss (MSE) is: %.5f' % loss_epoch_valid)
            # print(test_accuracy_epoch, best_loss)
            # print(checkpoints_dir)

            if (test_metrics['mse-loss'] <= best_loss):
                best_loss = test_metrics['mse-loss']
                log_string('Best loss is: %.5f' % best_loss)
                global_epoch += 1
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'train_loss': accuracy,
                    'test_loss': test_metrics['mse-loss'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
    writer.close()
if __name__ == '__main__':
    args = parse_args()
    main(args)
