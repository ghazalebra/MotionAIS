import numpy as np
from scipy.ndimage import median_filter
import cv2
import os
import csv
from tqdm import tqdm
import shutil
import json as json
import open3d as o3d

# the image dimensions
w = 1936
h = 1176
header_size = 512

# folder_path = '/home/travail/ghebr/Data/Participant01/autocorrection/Prise01'
# path_variants = ['']
# save_path = '/home/travail/ghebr/project/Data/Participant1/autocorrection/take1/'
# os.makedirs(save_path, exist_ok=True)


# reads the intensity raw file and converts it to an image_like array
def read_single_intensity_raw_file(file_path):
    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h, w)).T
    
    # reversing the image vertically
    data_array = np.flip(data_array, 0)
    # normalizing the values to be in the range (0, 255)
    data_array = 255 * (data_array - np.min(data_array)) / np.max(data_array)
    return data_array

# reads all the intensity raw files in a folder and save them as images in save_path folder (all the frames of an acquisition)
def read_raw_intensity_frames(folder_path, save_path):
    for filename in os.listdir(folder_path):
        # print('he')
        # check if it's an intensity file
        if '_I_' in filename:
            frame = read_single_intensity_raw_file(os.path.join(folder_path, filename))
            # removes the '.raw' extension from the end of the filename and replaces it with '.jpg'
            cv2.imwrite(save_path + filename[:-4] + '.jpg', frame)

def read_single_xyz_raw_file(file_path):
    with open(file_path, 'r') as f:
        f.seek(header_size)
        # data = f.read()
        frame = np.fromfile(f, np.float32).reshape((h, w, 3))
        #retrieve the depth as an image (and flip upside down)
        z = frame[:,:,2].T
        z = z[-1:0:-1, :]
        zz = z[np.where(z>0)]
        body_z = np.median(zz)
        z_nobg = z
        z_nobg[np.where(z > body_z + 400)] = 0
        z_nobg = median_filter(z_nobg, 5) #correction pour trous dans l'image
    # reversing the image vertically
    # data_array = data_array[0:1:1, :]
    # normalizing the values to be in the range (0, 255)
    # data_array = 255 * (data_array - np.min(data_array)) / np.max(data_array)
    return frame, z_nobg

def read_raw_xyz_frames(folder_path, save_path):
    for filename in os.listdir(folder_path):
        # print('found xyz file!')
        # check if it's an xyz file
        if '_XYZ_' in filename:
            frame, z_nobg = read_single_xyz_raw_file(os.path.join(folder_path, filename))
            
            # removes the '.raw' extension from the end of the filename and replaces it with '.txt'
            # print ('writing to ' + save_path + filename[:-4])
            cv2.imwrite(save_path + filename[:-4] + '.png', z_nobg)

#path = r'D:\StageE23\Data\Ete_2022\Participant07\autocorrection\Prise02'
#read_raw_xyz_frames(path, path+'\\xyz_nobg\\')


def copy_xyz_frames(src_path, des_path):
    for filename in os.listdir(src_path):
        # print('found xyz file!')
        # check if it's an xyz file
        if '_XYZ_' in filename:
            file_path = os.path.join(src_path, filename)
            save_file_path = os.path.join(des_path, filename)
            # print(file_path, save_file_path)
            shutil.copy(file_path, save_file_path)

def write_xyz_coordinates(folder_path, dict_coordo, crop):
    xyz_path = folder_path + '/xyz/'
    for filename in os.listdir(xyz_path):
        i = filename.index('XYZ')+4
        marqueurs = dict_coordo[f'image{int(filename[i:-4])+1}']
        # trouve les coordonnées associées aux marqueurs détectés
        coordos = find_xyz_coordinates(os.path.join(xyz_path, filename), marqueurs, crop)
        # removes the '.raw' extension from the end of the filename and replaces it with '.png'
        csv_filename = folder_path + '/XYZ_converted/' + filename[:-4] + '.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            # writer = csv.writer(csvfile, delimiter=';')
            for c in coordos:
                writer.writerow(c)

def find_xyz_coordinates(file_path, marqueurs, crop):
    # the image dimensions
    w = 1936
    h = 1176
    header_size = 512
    with open(file_path, 'r') as f:
        f.seek(header_size)
        data_array = np.fromfile(f, np.float32).reshape((h,w,3))
    #retrieve the depth as an image (and flip upside down)
    (w1, w2, h1, h2) = crop
    x_array = data_array[:, :,0].T
    x_array = x_array[-1:0:-1, :][h1:h2, w1:w2]
    y_array = data_array[:, :,1].T
    y_array = y_array[-1:0:-1, :][h1:h2, w1:w2]
    z_array = data_array[:, :,2].T
    z_array = z_array[-1:0:-1, :][h1:h2, w1:w2]

    coordos = []
    for el in marqueurs:
        x = (x_array[round(el[1]),round(el[0])])
        y = (y_array[round(el[1]),round(el[0])])
        z = (z_array[round(el[1]),round(el[0])])

        i = -2
        while [x,y,z] == [0.0, 0.0, 0.0]:
            print(i)
            x = (x_array[round(el[1]+i),round(el[0]+i)])
            print(x)
            y = (y_array[round(el[1]+i),round(el[0]+i)])
            print(y)
            z = (z_array[round(el[1]+i),round(el[0]+i)])
            print(z)
            i += 1

        coordos.append([x, y, z]) #liste de 5 listes contenant les coordos (x,y,z) pour chaque marqueur

    return coordos


def remove_bg(points):
    non_zero =  np.where(points[:, 2]>0)
    points_removed_0 = points[non_zero[0]]
    
    z_bg = np.max(points_removed_0[:, 2])
    non_max = np.where((points_removed_0[:, 2] < z_bg - 500))
    z_body = np.median(points_removed_0[non_max, 2])

    
    # print(z_body)
    # z_bg = np.max(points_removed_0[:, 2])
    # print(z_bg - z_body)
    # z_front = np.min(points_removed_0[:, 2])
    non_bg = np.where((points_removed_0[:, 2] < ((z_body + z_bg) / 2.)))
    points_non_bg = points_removed_0[non_bg]
    # y_middle = np.mean(points_non_bg[:, 0])
    # y_bottom = np.where(points_non_bg[:, 0] < y_middle)
    # print(y_middle)
    # print(y_bottom)
    # z_kun = np.min(points_non_bg[y_bottom, 2])
    # kun = np.where(points_non_bg[:, 2] == z_kun)
    # print(kun)
    # y_kun = points_non_bg[kun, 0][0][0]
   
    # y_kun = np.where(points_non_bg[:, 2] == )
    # y_down = np.min(points_non_bg[:, 0])
    # print('y_kun is ', y_kun, 'y_down is ', y_down)
    # non_feet = np.where((points_non_bg[:, 0] > (y_down + 700)))
    # non_feet = np.where((points_non_bg[:, 0] > (y_kun - 200)))
    # print(len(points))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_non_bg)
    cl, ind = pc.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=2.0)
    inlier = pc.select_by_index(ind)
    # print(ind)
    # return ind
    return non_bg[0], non_zero[0], ind


# reads the xyz, intensity, and markers position info of all the frames in a file and returns a dicts
def read_data(read_path, markers_path):

    # os.makedirs(save_path, exist_ok=True)
    pt = int(read_path[read_path.index('Participant') + 11: read_path.index('Participant') + 13])
    with open(markers_path) as markers_file:
        markers = json.load(markers_file)
    
    files = [file for file in os.listdir(read_path) 
         if os.path.isfile(os.path.join(read_path, file))]
    ready = 0
    data = {}
    data_frame = {}
    for file_name in tqdm(files):
        # print(file_name)
        file_path = read_path + '/' + file_name
        try:
            with open(file_path, 'r') as f:
                f.seek(header_size)
                if '_XYZ_' in file_name:
                    data_frame['xyz'] = np.fromfile(f, np.float32).reshape((w*h, 3))
                    frame_key = file_name.replace('_XYZ_', '_')[:-4]
                    ready += 1
                elif '_I_' in file_name:
                    data_frame['intensity'] = np.fromfile(f, np.float32)
                    frame_key = file_name.replace('_I_', '_')[:-4]
                    ready += 1
        except:
            print('Error in reading from ' + file_path + '!')
            continue
        if ready == 2:
            # with open(file_save_path, 'wb') as f:
            #     np.save(f, data)
            # print(data)
            # print('removing the bg')
            # print(data_frame['xyz'])
            non_bg, non_zero, inlier = remove_bg(data_frame['xyz'])
            data_frame['xyz'] = data_frame['xyz'][non_zero]
            data_frame['xyz'] = data_frame['xyz'][non_bg]
            # data_frame['xyz'] = data_frame['xyz'][non_feet]
            data_frame['xyz'] = data_frame['xyz'][inlier]
            # print('done removing the bg')
            data_frame['intensity'] = data_frame['intensity'][non_zero]
            data_frame['intensity'] = data_frame['intensity'][non_bg]
            # data_frame['intensity'] = data_frame['intensity'][non_feet]
            data_frame['intensity'] = data_frame['intensity'][inlier]
            intensities = data_frame['intensity']
            data_frame['intensity'] = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            try:
                data_frame.update({'markers': markers['pt' + str(pt) + '_' + file_name[:-4].replace('_I_', '_XYZ_')]})
            except:
                try:
                    data_frame.update({'markers': markers[file_name[:-4].replace('_I_', '_XYZ_').replace('pt0', 'pt')]})
                except:
                    # print(file_name[:-4])
                    data_frame.update({'markers': markers[file_name[:-4].replace('_I_', '_XYZ_')]})

            data.update({frame_key: data_frame})
            data_frame = {}
            ready = 0
    data_path = read_path + '/data/'
    os.makedirs(data_path, exist_ok=True)
    frames_names = data.keys()
    for frame_name in frames_names:
        with open(data_path + frame_name + '.npy', 'wb') as data_file:
            np.save(data_file, data[frame_name])

    return data


            

# testing
# file_path = '/home/travail/ghebr/Data/Participant26/BD/Libre/Prise01/pt26_BD_Libre_01_078493_XYZ_0.raw'
# try:
#     with open(file_path, 'r') as f:
#         f.seek(header_size)
#         if '_XYZ_' in file_path:
#             frame = np.fromfile(f, np.float32).reshape((w*h, 3))
#         else:
#             print('not an xyz file!')
#             # frame_key = file_name.replace('_XYZ_', '_')[:-4]
#             # ready += 1
# except:
#     print('Error in reading from ' + file_path + '!')

# non_bg, non_zero, inlier = remove_bg(frame)
# print(len(frame))
# frame = frame[non_zero]
# frame = frame[non_bg]
# frame = frame[inlier]
# print(len(frame))