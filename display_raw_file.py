import numpy as np
import cv2

path = '../Data/Participant01/autocorrection/Prise01/auto_01_014830_I_49.raw'


fd = open(path, 'rb')
rows = 480
cols = 640
f = np.fromfile(fd, dtype=np.uint8, count=rows * cols)
im = f.reshape((rows, cols))  # notice row, column format
fd.close()

cv2.imshow('', im)
cv2.waitKey()
cv2.destroyAllWindows()
