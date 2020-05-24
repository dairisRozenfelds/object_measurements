import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('test.jpg')
grayImageCV2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grayImage = []

for row in image:
    grayRow = []

    for val in row:
        grayVal = 0

        for color in val:
            grayVal += color
        
        grayVal /= 3
        grayVal = round(grayVal)
        grayVal = int(grayVal)

        grayRow.append(grayVal)
    grayImage.append(grayRow)

grayImage = np.array(grayImage)

w = plt.figure()

w.add_subplot(2, 1, 1)
plt.imshow(cv2.convertScaleAbs(grayImage), 'gray')
plt.axis('off')

w.add_subplot(2, 1, 2)
plt.imshow(cv2.convertScaleAbs(grayImageCV2), 'gray')
plt.axis('off')

plt.show(block=True)
