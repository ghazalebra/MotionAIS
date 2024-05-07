import argparse
import os
import torch
import datetime
import logging
import sys
import shutil
# import provider
import numpy as np
import json 

from pathlib import Path
from tqdm import tqdm
# from data_utils.MyDataLoader import BackLandmarksDataset
# import read_raw_file as RRF
# import marker_detection
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter
import copy

def find_xyz_coordinates(file_path, markers):
    # markers = []
    # for landmark_name in markers_dict.keys():
    #     markers.append([markers_dict[landmark_name][0], markers_dict[landmark_name][1]])
    # the image dimensions
    w = 1936
    h = 1176
    header_size = 512
    # if it's a raw file
    try:
        with open(file_path, 'r') as f:
            f.seek(header_size)
            data_array = np.fromfile(f, np.float32).reshape((h,w,3))
    except:
        data_array = np.load(file_path)
    #retrieve the depth as an image (and flip upside down)
    x_array = data_array[:, :,0]
    # x_array = x_array[-1:0:-1, :]
    y_array = data_array[:, :,1]
    # y_array = y_array[-1:0:-1, :]
    z_array = data_array[:, :,2]
    # z_array = z_array[-1:0:-1, :]

    coordos = {}
    for landmark_name in markers.keys():
        el = markers[landmark_name]
        # print(el[1], round(el[1]), el[0], round(el[0]))
        x = (x_array[round(el[1]),round(el[0])])
        y = (y_array[round(el[1]),round(el[0])])
        z = (z_array[round(el[1]),round(el[0])])
        # z = (z_array[round(el[1]),round(el[0])])
        # print(z)


        i = -2
        while [x,y,z] == [0.0, 0.0, 0.0]:
        # if z == 0.0:
            # print(i)
            x = (x_array[round(el[1]+i),round(el[0]+i)])
            # print(x)
            y = (y_array[round(el[1]+i),round(el[0]+i)])
            # print(y)
            z = (z_array[round(el[1]+i),round(el[0]+i)])
            # print(z)
            i += 1

        # coordos.append([x, y, z]) #liste de 5 listes contenant les coordos (x,y,z) pour chaque marqueur
        coordos[landmark_name] = [float(x), float(y), float(z)]

    return coordos

def find_xyz_coordinates_sequence(sequence_path):
    xyz_path = sequence_path + '/xyz_images'
    markers_path = sequence_path + '/Positions/original_positions.json'
    markers3d_path = sequence_path + '/Positions/positions3d.json'
    with open(markers_path) as markers_file:
        markers = json.load(markers_file)
    markers_3d = {}
    # print(xyz_path)
    # print(os.listdir(xyz_path))
    for frame in os.listdir(xyz_path):
        frame_path = xyz_path + '/' + frame
        frame_name = frame[:-4]
        # print(frame_name)
        markers_frame = markers[frame_name]
        markers_3d_frame = find_xyz_coordinates(frame_path, markers_frame)
        markers_3d[frame_name] = markers_3d_frame
    # print(markers_3d)
    with open(markers3d_path, 'w') as markers3d_file:
        json.dump(markers_3d, markers3d_file)

def save_landmarks(landmarks, path):
    landmarks_names = ['C', 'T1', 'T2', 'L2', 'G', 'D', 'IG', 'ID']
    # number of frames
    n = len(landmarks)
    landmarks_dict = {}
    for i in range(n):
        image_name = 'image' + str(i+1)
        landmarks_frame = {}
        for j, l in enumerate(landmarks_names):
        
            landmarks_frame[l] = landmarks[i]
        landmarks_dict[image_name] = landmarks_frame
    
    with open(path, 'w') as f:
        json.dump(landmarks_dict, f)


def draw_landmarks_single_frame(frame_path, landmarks):

    frame_name_index = frame_path.rindex('/')
    # print(frame_path)
    frame_name = frame_path[frame_name_index+1:-4] + '.jpg'

    prediction_path = '/home/travail/ghebr/project/Data/predicted/8843/both/frames_landmarks/' + frame_name
    os.makedirs('/home/travail/ghebr/project/Data/predicted/8843/both/frames_landmarks/', exist_ok=True)
    # print(prediction_path)

    frame = cv2.imread(frame_path)
    frame = np.asarray(frame).astype(np.float64)

    for landmark in landmarks:
        # frame_i = cv2.circle(frame_i, tuple([int(landmarks_i[landmark_name][0]), int(landmarks_i[landmark_name][1])]),5,(0,0,255))
        frame = cv2.circle(frame, tuple([int(landmark[0]), int(landmark[1])]),5,(0,0,255))
        # frame_i = cv2.putText(frame_i, landmark_name, tuple([int(markers_i[landmark_name][0])+10, int(markers_i[landmark_name][1])+10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(prediction_path, frame)


def draw_landmarks(path, indices=None, draw_lines=False):
    annotated_path = 'visualizations/annotatations/'
    markers_path = path + '/Positions/original_positions.json'

    # markers = {}
    # print(markers_path)
    with open(markers_path) as markers_file:
        markers = json.load(markers_file)

    # print(markers)
    
    os.makedirs(annotated_path, exist_ok=True)

    # number of frames
    frame_path = path + '/intensity/'
    xyz_path = path + '/xyz_images/'
    frames = os.listdir(frame_path)
    frames_keys = os.listdir(xyz_path)
    if indices is None:
        indices = [i for i in range(len(frames))]
    for i in indices:
        frame_i = cv2.imread(frame_path + frames[i])
        frame_i = np.asarray(frame_i).astype(np.float64)
        # landmarks_i = landmarks['image'+str(i+1)]
        markers_i = markers[frames_keys[i][:-4]]
        for landmark_name in markers_i.keys():
            # frame_i = cv2.circle(frame_i, tuple([int(landmarks_i[landmark_name][0]), int(landmarks_i[landmark_name][1])]),5,(0,0,255))
            try:
                frame_i = cv2.circle(frame_i, tuple([int(markers_i[landmark_name][0]), int(markers_i[landmark_name][1])]),5,(0,255,0))
                landmark_name_ = landmark_name.replace('sup', 'up').replace('inf', 'down').replace('D', 'R').replace('G', 'L').replace('ap', 'appex')
                frame_i = cv2.putText(frame_i, landmark_name_, tuple([int(markers_i[landmark_name][0])+10, int(markers_i[landmark_name][1])+10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            except:
                print('error')
        if draw_lines:
            frame_i = cv2.line(frame_i, tuple([int(markers_i['Tsup'][0]), int(markers_i['Tsup'][1])]), tuple([int(markers_i['Tap'][0]), int(markers_i['Tap'][1])]), (0, 255, 0), thickness=1, lineType=8)
            frame_i = cv2.line(frame_i, tuple([int(markers_i['Tap'][0]), int(markers_i['Tap'][1])]), tuple([int(markers_i['Tinf'][0]), int(markers_i['Tinf'][1])]), (0, 255, 0), thickness=1, lineType=8)

        cv2.imwrite(annotated_path + 'annotated_' + frames[i], frame_i + 30)


# Prend une image xyz et retourne z en binaire 
def remove_bg(xyz, save_path_xyz):
    z = xyz[:,:,2]

    zz = z[np.where(z>0)]
    zz = zz[np.where(zz<2500)]
    z_nobg = copy.deepcopy(z)
    body_z = np.quantile(zz, 0.3)
    if 'Contraint' in os.listdir(save_path_xyz)[0]:
        z_nobg[np.where(z > body_z + 150)] = False
    else:
        z_nobg[np.where(z > body_z + 250)] = False
    z_nobg = median_filter(z_nobg, 3)

    return z_nobg

def automatic_crop(save_path_xyz):
    # timer_debut = time.process_time_ns()
    xyz = np.load(os.path.join(save_path_xyz, os.listdir(save_path_xyz)[0]))
    z_nobg = remove_bg(xyz, save_path_xyz)
    body_LR = np.argwhere(z_nobg[1250,:]) #identifie points n'appartenant pas au bg, donc au corps du patient
    body_HL = np.argwhere(z_nobg[:,600])

    left = int(body_LR[0])
    right = int(body_LR[-1])

    w1 = 0
    w2 = 0
    h1 = 0
    h2 = 0

    if 'BG' in os.listdir(save_path_xyz)[0]:
        # print('BG')
        w1 = np.max(left-100, 0)
        w2 = right+50
        h1 = int(body_HL[0])+100
    elif 'BD' in os.listdir(save_path_xyz)[0]:
        # print('BD')
        w1 = left-50
        w2 = right+100
        h1 = int(body_HL[0])+100
    else:
        # print('other')
        w1 = np.max(left-80, 0)
        w2 = right+80
        h1 = int(body_HL[0])+50

    h2 = h1+int(6/5*(w2-w1))
    # h1 -= 100
    return w1, w2, h1, h2


def reverse_crop(path, crop=None):
    xyz_path = path + '/xyz_images/'
    if crop is None:
        w1, w2, h1, h2 = automatic_crop(xyz_path)
    else:
        (w1, w2, h1, h2) = crop
    
    frame_names = os.listdir(xyz_path)
    markers_path = path + '/Positions/' + os.listdir(path + '/Positions/')[0]
    # print(markers_path)
    original_markers_path = path + '/Positions/original_positions.json'
    original_positions = {}
    with open(markers_path) as f:
        markers = json.load(f)
    for i, image_name in enumerate(markers.keys()):
        markers_image = markers[image_name]
        original_positions_image = {}
        frame_name_i = frame_names[i][:-4]
        for landmark_name in markers_image.keys():
            original_positions_image[landmark_name] = [markers_image[landmark_name][0] + w1, markers_image[landmark_name][1] + h1]
        original_positions[frame_name_i] = original_positions_image
    with open(original_markers_path, 'w') as f:
        json.dump(original_positions, f)

# to change the keys in the markers dict to the actual file names
def change_names(sequence_path):
    intensity_path = sequence_path + 'xyz/'
    markers_path = sequence_path + 'markers_position/original_positions.json'
    old_path = sequence_path + 'markers_position/original_positions_Lea.json'
    frame_names = os.listdir(intensity_path)
    with open(old_path) as f:
        markers = json.load(f)
    
    new_markers = {}

    for i, frame_name in enumerate(frame_names):
        key_frame_i = 'image' + str(i+1)
        # index_I = frame_name.index('_I_')
        new_key = frame_name[:-4]
        new_markers[new_key] = markers[key_frame_i]

    with open(markers_path, 'w') as f_new:
        json.dump(new_markers, f_new)
    # with open(old_path, 'w') as f_old:
    #     json.dump(markers, f_old)

# adding the participant's numbers to the name of the files
# path_variants = ['BG/Contraint/Prise01', 'BG/Contraint/Prise02', 'BG/Libre/Prise01', 'BG/Libre/Prise02', 
# 'BD/Contraint/Prise01', 'BD/Contraint/Prise02', 'BD/Libre/Prise01', 'BD/Libre/Prise02', 'autocorrection/Prise01', 'autocorrection/Prise02']
# path_base = '/home/travail/ghebr/Data/'
# if len(sys.argv) > 1:
#         pt_n = str(sys.argv[1])
#         data_path = path_base + 'Participant' + pt_n + '/'
# else:
#     data_path = path_base + 'Participant3/'
# for path in tqdm(path_variants):
#     sequence_path = data_path + path + '/'
#     for file_name in os.listdir(sequence_path):
#         if os.path.isfile(sequence_path + file_name):
        
#             file_name_new = 'pt' + pt_n + '_' + file_name
#             os.rename(sequence_path + file_name, sequence_path + file_name_new)

# run for all the new annotations
# writing the names of the files in a text file
root = '/home/travail/ghebr/Data/'
validation = [
'Participant2/BD/Libre/Prise01', 'Participant2/BD/Libre/Prise02', 
# 'Participant2/BG/Libre/Prise01', 'Participant2/BG/Libre/Prise02',
'Participant6/autocorrection/Prise01', 'Participant6/autocorrection/Prise02'
]
training = [
# 'Participant15/BG/Libre/Prise01', 'Participant15/BG/Libre/Prise02',
# 'Participant15/BD/Libre/Prise01', 'Participant15/BD/Libre/Prise02',
# 'Participant16/BD/Libre/Prise02', 
# 'Participant19/BG/Libre/Prise01', 
# 'Participant19/BG/Libre/Prise02',
# 'Participant19/BD/Libre/Prise01', 'Participant19/BD/Libre/Prise02',
'Participant23/BD/Libre/Prise01', 'Participant23/BD/Libre/Prise02',
'Participant23/BG/Libre/Prise01', 'Participant23/BG/Libre/Prise02',
# 'Participant23/autocorrection/Prise02',
# 'Participant25/BG/Libre/Prise01'
]
# testing = ['Participant3/BG/Libre/Prise01', 'Participant2/autocorrection/Prise01']
# train_file = '/home/travail/ghebr/project/Data/train.txt'
# test_file = '/home/travail/ghebr/project/Data/test.txt'
# def create_data_file(data_file, sequences):
#     with open(data_file, 'w') as f:
#         for sequence in sequences:
#             path = root + sequence
#             path_pc = path + '/xyz_removed_bg/'
#             for i, file_name in enumerate(os.listdir(path_pc)):
#                 if i % 3 == 0:
#                     file_path = path_pc + file_name
#                     # print(file_path)
#                     f.write(file_path + '\n')
# create_data_file(train_file, training)


# pc_path = '/home/travail/ghebr/project/Data/Participant6/autocorrection/Prise02/xyz_removed_bg/pt6_auto_02_008170_XYZ_78.ply'
# frame_name_index = pc_path.rindex('/')
# frame_name = pc_path[frame_name_index+1:-4]
# sequence_path = pc_path[:frame_name_index-14]
# annotations_path = sequence_path + 'markers_position/' + 'original_positions.json'
# with open(annotations_path, 'r') as annotations_file:
#     annotations_sequence = json.load(annotations_file)
# annotations_sequence[frame_name]

# test_path = '/home/travail/ghebr/project/Data/test.txt'
# train_path = '/home/travail/ghebr/project/Data/train.txt'
# with open('/home/travail/ghebr/project/Data/predicted/8843/cls/landmarks.json', 'r') as landmarks_file:
#     landmarks = json.load(landmarks_file)
#     print(len(landmarks))
# # prediction_path = '/home/travail/ghebr/project/Data/predicted/8843/cls/frames_landmarks/'
# os.makedirs(prediction_path, exist_ok=True)
# with open(test_path, 'r') as pc_paths:
#     i = 0
#     for pc_path in tqdm(pc_paths):
#         # Remove the newline character at the end of the line
#         pc_path = pc_path.strip()
#         # self.pointsets.append(pc_path)
#         frame_name_index = pc_path.rindex('/')
#         frame_name = pc_path[frame_name_index+1:-4]
#         # print(frame_name)
#         sequence_path = pc_path[:frame_name_index-14]
#         # annotations_path = sequence_path + 'markers_position/' + 'original_positions.json'
#         intensity_path = sequence_path + 'annotated_frames/'
#         frame_idx = int(frame_name[frame_name.rindex('_')+1:])
#         # print(frame_idx)
#         intensity_frame = intensity_path + os.listdir(intensity_path)[frame_idx]
#         landmarks_frame = landmarks[i]
#         draw_landmarks_single_frame(intensity_frame, landmarks_frame)
#         i += 1
crops = {'pt15_BD_Libre_Prise01' : (263, 1053, 539, 1487),
'pt15_BD_Libre_Prise02' : (260, 1046, 553, 1496),
'pt15_BG_Libre_Prise01' : (34, 828, 597, 1549),
'pt15_BG_Libre_Prise02' : (88, 846, 554, 1463),

'pt16_BD_Libre_Prise02' : (265, 841, 512, 1203),

'pt19_BD_Libre_Prise01' : (292, 902, 561, 1293),
'pt19_BD_Libre_Prise02' : (289, 894, 572, 1298),
'pt19_BG_Libre_Prise01' : (251, 842, 530, 1239),
'pt19_BG_Libre_Prise02' : (249, 848, 550, 1268),

'pt20_BD_Libre_Prise01' : (244, 961, 697, 1557),
'pt20_BD_Libre_Prise02' : (226, 942, 547, 1406)
}
# for sequence in training:
#     sequence_path = root + sequence
#     sequence_key = sequence.replace('Participant', 'pt').replace('/', '_')
#     print(sequence_key)
#     # crop = crops[sequence_key]
#     reverse_crop(sequence_path)
#     find_xyz_coordinates_sequence(sequence_path)
#     draw_landmarks(sequence_path)