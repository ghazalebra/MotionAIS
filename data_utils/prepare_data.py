# writing the path to each trainiing frame in train.txt file and the same for testing frames
# add the new sequences to the new_test_sequencs and new_train_sequences. uncomment the last two lines and run.
import os
import torch
import sys
import numpy as np
import json 
from tqdm import tqdm
import read_raw_file as Reader

import cv2
from scipy.ndimage import median_filter
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
         # print('here')
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
        try:
            markers_frame = markers[frame_name]
        except:
            markers_frame = markers[frame_name.replace('pt0', 'pt')]
        # print('--------> ' + frame_path)
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


def draw_landmarks(path, preprocessed=False):
    if preprocessed:
        annotated_path = path + '/preprocessed_annotated_frames_corrected/'
        markers_path = path + '/Positions/positions_corrigees.json'
        frame_path = path + '/Preprocessed/'
        r = 8
        t = 2
        s = 0.75
    else:
        annotated_path = path + '/annotated_frames_corrected/'
        markers_path = path + '/Positions/original_positions.json'
        frame_path = path + '/intensity/'
        r = 5
        t = 1
        s = 0.5

    # markers = {}
    # print(markers_path)
    with open(markers_path) as markers_file:
        markers = json.load(markers_file)
    
    os.makedirs(annotated_path, exist_ok=True)

    # number of frames
    
    
    xyz_path = path + '/xyz_images/'
    frames = os.listdir(frame_path)
    frames_keys = os.listdir(xyz_path)
    n = len(frames)
    for i in range(n):
        frame_i = cv2.imread(frame_path + frames[i])
        frame_i = np.asarray(frame_i).astype(np.float64)
        # landmarks_i = landmarks['image'+str(i+1)]
        try:
            markers_i = markers[frames_keys[i][:-4].replace('pt0', 'pt')]
        except:
            try: 
                markers_i = markers[frames_keys[i][:-4]]
            except:
                markers_i = markers[f'image{i+1}']
        for landmark_name in markers_i.keys():
            # frame_i = cv2.circle(frame_i, tuple([int(landmarks_i[landmark_name][0]), int(landmarks_i[landmark_name][1])]),5,(0,0,255))
            landmark_name_ = landmark_name.replace('sup', 'up').replace('inf', 'down').replace('D', 'R').replace('G', 'L').replace('ap', 'apex')
            try:
                frame_i = cv2.circle(frame_i, tuple([int(markers_i[landmark_name][0]), int(markers_i[landmark_name][1])]), r, (0,255,0), t)
                frame_i = cv2.putText(frame_i, landmark_name_, tuple([int(markers_i[landmark_name][0])+10*t, int(markers_i[landmark_name][1])+10*t]), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 255, 0), t, cv2.LINE_AA)
            except:
                print('here')
                continue
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
    print('in automatic crop')
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
    # h1 -= 50

    return w1, w2, h1, h2

# for participants 2, 3, 5, 6, 8, 13
def automatic_crop_old(path):
    # timer_debut = time.process_time_ns()

    # path_nobg = path + '/xyz_images/'
    # if not 'xyz_images' in os.listdir(path):
    #     os.mkdir(path)
    #     Reader.read_raw_xyz_frames(path)

    path_nobg = path+'/xyz_nobg/'
    if not 'xyz_nobg' in os.listdir(path) or len(os.listdir(path_nobg)) == 0:
        os.makedirs(path_nobg, exist_ok=True)
        Reader.read_raw_xyz_frames(path, path_nobg)
    z_nobg = cv2.imread(os.path.join(path_nobg, os.listdir(path_nobg)[0]))
    z_nobg = median_filter(z_nobg, 3)
    print(z_nobg.shape)
    body = np.argwhere(z_nobg[1000,:,0] > 100) #identifie points n'appartenant pas au bg, donc au corps du patient



    # xyz = np.load(os.path.join(path, os.listdir(path)[0]))
    # z_nobg = remove_bg(xyz, path)
    # z_nobg = median_filter(xyz, 3)
    # print(z_nobg.shape)
    # body = np.argwhere(z_nobg[1000,:,0] > 100) #identifie points n'appartenant pas au bg, donc au corps du patient


    left = int(body[0])
    right = int(body[-1])

    if 'BG' in os.listdir(path_nobg)[0]:
        w1 = left-100
        w1 = left-50
        w2 = w1+600
    elif 'BD' in os.listdir(path_nobg)[0]:
        w2 = right+100
        w2 = right+50
        w1 = w2-600
    else:
        w1 = left-50
        w2 = w1+600
    h1 = 560
    h2 = 1280
    print(w1, w2, h1, h2)


    # global crop
    crop = (w1, w2, h1, h2)
    return crop

    # self.ids.width.text = f'({w2-w1}, 0)'
    # self.ids.height.text = f'(0, {h2-h1})'

    # timer_fin = time.process_time_ns()
    # print(timer_debut, timer_fin)
    # print(f'Temps automatic crop : {timer_fin - timer_debut} ns')



def reverse_crop(path, crop=None, version=None):
    xyz_path = path + '/xyz_images/'
    if crop is None:
        if version == 'old':
            w1, w2, h1, h2 = automatic_crop_old(path)
        else:
            w1, w2, h1, h2 = automatic_crop(xyz_path)
    else:
        (w1, w2, h1, h2) = crop
    
    frame_names = os.listdir(xyz_path)
    
    # print(markers_path)
    original_markers_path = path + '/Positions/original_positions.json'
    original_positions = {}
    # print(markers_path)
    markers_path = [path + '/Positions/' + p for p in os.listdir(path + '/Positions/')]
    for i in range(len(markers_path)):
        try:
            with open(markers_path[i]) as f:
                markers = json.load(f)
            print(markers_path[i])
            break
        except:
            continue
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



def create_path(pt_number, sequence_name):
    sequence_path = sequence_name.replace('_', '/')
    for i in range(1, 5):
        sequence_path = sequence_path.replace(f'0{i}', f'Prise0{i}')
    # xyz_path = 'Participant' + str(pt_number) + '/' + sequence_path + '/xyz_removed_bg/'
    # intensity_path = 'Participant' + str(pt_number) + '/' + sequence_path + '/intensity_removed_bg/'
    data_path = 'Participant' + str(pt_number) + '/' + sequence_path
    return data_path

def write_paths_to_file(sequences, file_path):
    
    with open(file_path, 'a') as f:
        for pt in sequences.keys():
            for sequence in sequences[pt]:
                xyz_path, intensity_path, _ = create_path(pt, sequence)
                xyz_path = data_root + xyz_path
                intensity_path = data_root + intensity_path
                xyz_files = os.listdir(xyz_path)
                instensity_files = os.listdir(intensity_path)
                for file_name1, file_name2 in zip(xyz_files, instensity_files):
                    file_path1 = xyz_path + file_name1
                    file_path2 = intensity_path + file_name2
                    f.write(file_path1 + '\n')
                    f.write(file_path2 + '\n')

def rename_files(path):
    path += '/'
    files = os.listdir(path)
    pt = path[path.index('Participant') + 11:path.index('Participant') + 11 + 2]
    for file in files:
        if os.path.isfile(path + file) and 'pt' not in file:
            os.rename(path + file, path + f'pt{pt}_' + file)
        elif os.path.isdir(path + file) and file != 'Positions':
            rename_files(path + file)

# gets the data for each frame of each sequence and writes the name of the frames to train.txt/test.txt
def write_data(sequences, root, root_old='', split='train', new=False, crops=None, version=None):

    data = {}
    save_path = root + f'{split}.txt'
    # os.makedirs(save_path, exist_ok=True)
    with open(save_path, 'a') as f:
        for pt in sequences.keys():
            crop = None
            if crops is not None and pt in crops.keys():
                crop = crops[pt]
            sequences_pt = sequences[pt]
            for sequence in sequences_pt:
                data_path = create_path(pt, sequence)
                rename_files(root + data_path)
                markers_path = root + data_path + '/Positions/positions3d.json'
                read_path = root + data_path
                print(pt + ' ' + sequence)
                os.makedirs(read_path + '/data', exist_ok=True)
                frames = os.listdir(read_path + '/data')
                if len(frames) == 0 or new:
                    # checking the positions
                    sequence_path = read_path
                    # positions3D_path = sequence_path + '/Positions/positions_xyz.json'
                    # sequence_key = sequence.replace('Participant', 'pt').replace('/', '_')

                    if os.path.exists(sequence_path + '/Positions/positions3d.json') and not new:
                        print('positions are ready!')
                        
                        try:
                            draw_landmarks(sequence_path)
                            draw_landmarks(sequence_path, preprocessed=True)
                        except:
                            print('could not draw the landmarks!')


                    else:
                        # crop = crops[sequence_key]
                        if not os.path.exists(sequence_path + '/Positions/original_positions.json') or new:
                            print('reverse cropping')
                            reverse_crop(sequence_path, crop=crop, version=version)
                        find_xyz_coordinates_sequence(sequence_path)
                        draw_landmarks(sequence_path)
                        draw_landmarks(sequence_path, preprocessed=True)
                        print('positions are ready but not validated!')
                    Reader.read_data(read_path, markers_path)
                    frames = os.listdir(read_path + '/data')
                for frame in frames:
                    f.write(frame[:-4] + '\n')

root = '/home/travail/ghebr/Data/'

new_test_sequences = {
            # '02': [
            #        'autocorrection_01', 
                #    'autocorrection_02'
                # ],
            # '03': ['autocorrection_01', 'autocorrection_02', 'autocorrection_03'],
            # '05': ['autocorrection_01', 'autocorrection_02'],
            # '06': ['autocorrection_01', 'autocorrection_02'],
            # '08': ['autocorrection_01', 'autocorrection_02'],
            # '13': ['autocorrection_01', 'autocorrection_02'],                 
            #  '26': ['autocorrection_02'], 
            #  '25': ['autocorrection_01', 'autocorrection_02'],
            #  '24': ['autocorrection_01', 'autocorrection_02', 'autocorrection_03', 'autocorrection_04'],
            #  '23': ['autocorrection_01', 'autocorrection_02', 'autocorrection_03'],
            #  '22': ['autocorrection_01', 'autocorrection_02', 'autocorrection_03'],
            #  '20': ['autocorrection_01', 'autocorrection_02'],
            #  '19': [
                #  'autocorrection_01', 
                #  'autocorrection_02'],
            #  '15': ['autocorrection_01', 'autocorrection_02']
             }

# 4 and 17 acquisition problem.
# 11 no marker
# 7 crop problem
# 1, 5 markers
# 14, 4 markers 
# Summer 2022
test_sequences = {
    '08': ['autocorrection_01', 'BD_Libre_01', 'BD_Libre_02', 'BG_Libre_01', 'BG_Libre_02',],
    '06': ['autocorrection_01'],
    '09': ['autocorrection_01'],
    '10': ['autocorrection_01'],
    '12': ['autocorrection_01'],
    '23': ['autocorrection_01'],
}

# test_sequences_8_auto = {'06': ['autocorrection_01', 'autocorrection_02'], '08': ['autocorrection_01', 'autocorrection_02'], '03': ['autocorrection_01', 'autocorrection_03']}
# test_sequence_auto_8_seen = {'24': ['autocorrection_01', 'autocorrection_02', 'autocorrection_03', 'autocorrection_04']}
# crops = {'02': (285, 898, 507, 1242)}


# write_data(test_sequences, root, split='test')


sequence_path = '/home/travail/ghebr/Data/Participant23/autocorrection/Prise01'

draw_landmarks(sequence_path, preprocessed=True)


# # getting the xyz and intensity files
# for i in tqdm(range(22, 26)):
#     src_path = root_src + 'Participant' + str(i)
#     dst_path = root_dst + 'Participant' + str(i)
#     # print(os.path.isdir(src_path + '/' + os.listdir(src_path)[0]))
#     sequence_names = [name for name in os.listdir(src_path) if os.path.isdir(src_path + '/' + name)]
#     # print(sequence_names)
#     for sequence_name in sequence_names: 
#         bending_types = ['']
#         if sequence_name == 'BD' or sequence_name == 'BG':
#             bending_types = ['/Libre/', '/Contraint/']
#         # print(bending_types)
#         for bending_type in bending_types:
#             takes = [name for name in os.listdir(src_path + '/' + sequence_name + bending_type) if os.path.isdir(src_path + '/' + sequence_name + '/' + bending_type + name)]

#             for take in takes:
#                 sequence_path_src = src_path + '/' + sequence_name + bending_type + '/' + take
#                 sequence_path_dst = dst_path + '/' + sequence_name + bending_type + '/' + take
#                 os.makedirs(sequence_path_dst+'/xyz/', exist_ok=True)
#                 os.makedirs(sequence_path_dst+'/intensity/', exist_ok=True)
#                 print(sequence_path_src, sequence_path_dst+'/xyz/')
#                 if len(os.listdir(sequence_path_dst+'/xyz/')) = 0:
#                     Reader.copy_xyz_frames(sequence_path_src, sequence_path_dst+'/xyz/')
#                 if len(os.listdir(sequence_path_dst+'/intensity/')) = 0:
#                     Reader.read_raw_intensity_frames(sequence_path_src, sequence_path_dst+'/intensity/')


# finding and validating the 3D coordinates of landmarks. Check annotated_frames for validation!
# for sequence in training:
#     sequence_path = root + sequence
#     positions3D_path = sequence_path + '/Positions/positions_xyz.json'
#     sequence_key = sequence.replace('Participant', 'pt').replace('/', '_')
#     print(sequence_key)
#     if os.path.exists(sequence_path + '/Positions/positions_xyz.json') or os.path.exists(sequence_path + '/Positions/positions3d.json'):
#         try:
#             os.rename(sequence_path + '/Positions/positions_xyz.json', sequence_path + '/Positions/positions3d.json')
#         except:
#             try:
#                 shutil.copyfile(root_old + sequence + '/markers_position/positions3d.json', sequence_path + '/Positions/positions3d.json')
#             except:
#                 print('positions3d.json found!')
#         draw_landmarks(sequence_path)


#     else:
#         # crop = crops[sequence_key]
#         reverse_crop(sequence_path)
#         find_xyz_coordinates_sequence(sequence_path)
#         draw_landmarks(sequence_path)

# data_root = '/home/travail/ghebr/project/Data/'

# train_sequences = {'2': ['BD_Libre_01', 'BD_Libre_02'], '5': ['BG_Libre_01', 'BG_Libre_02', 'BD_Libre_01', 'BD_Libre_02'], \
# '8': ['BG_Libre_01', 'BG_Libre_02', 'BD_Libre_01', 'BD_Libre_02'], '13': ['BG_Libre_01', 'BG_Libre_02', 'BD_Libre_02'],
# '20': ['BD_Libre_01', 'BD_Libre_02'], '15': ['BG_Libre_01', 'BG_Libre_02', 'BD_Libre_01', 'BD_Libre_02'],
# '19': ['BG_Libre_01', 'BD_Libre_01', 'BD_Libre_02']}


# train_path = save_root + 'train/'
# test_path = save_root + 'test/'
# for frame_key in data.keys():
#     with open(test_path + frame_key + '.npy', 'wb') as f:
#         np.save(f, data[frame_key])


# uncomment these for adding new sequences
# write_paths_to_file(new_train_sequences, train_file_path)
# write_paths_to_file(test_sequences, test_file_path)

# ignore this!
# test_root = '/home/travail/ghebr/project/data_utils/test/'
# # read_frames(read_path, save_path)
# with open('/home/travail/ghebr/project/Data/test/pt3_BG_libre_01_001695_66.npy', 'rb') as f:
#     a = np.load(f, allow_pickle=True).item()
# print(a.keys())
