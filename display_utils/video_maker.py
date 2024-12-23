import cv2
import numpy as np
import os
from os.path import isfile, join



pathIn = '/home/travail/ghebr/Data/Participant23/autocorrection/Prise03/preprocessed_annotated_frames_corrected/'
pathOut = 'pt23_auto_03_markers.mp4'
fps = 10
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]  # for sorting the file names properly
files.sort(key=lambda x: x[5:-4])
files.sort()
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]  # for sorting the file names properly
files.sort(key=lambda x: x[5:-4])

for i in range(len(files)):
    filename = pathIn + files[i]
    # reading each files
    img = cv2.imread(filename)
    # cv2.imshow('hi', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # break
    height, width, layers = img.shape
    size = (width, height)
    
    # inserting the frames into an image array
    frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
out.release()
