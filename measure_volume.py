import cv2
import numpy as np
import urllib.request
import imutils
from imutils import contours
from imutils import perspective
import math
from scipy.spatial import distance

image = cv2.imread('test_images/2d/2pr/7_prieksa.jpg')
volImage = cv2.imread('test_images/3d/7_sanskats.jpg')

refArea = 400 # Reference area
refAreaUnit = 'cm^2' # Reference area measurement units

refVolHeight = 20 # Volume reference object's height
refVolWidth = 20 # Volume reference object's width
refVolArea = refVolHeight * refVolWidth
refVolUnit = 'cm' # Volume reference object's units

scale_percent = 100 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(image.copy(), dim)
volImg = cv2.resize(volImage.copy(), dim)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
volGray = cv2.cvtColor(volImg, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
volGray = cv2.GaussianBlur(volGray, (5, 5), 0)

edges = cv2.Canny(gray, 50, 100)
edges = cv2.dilate(edges, None, iterations=1)
edges = cv2.erode(edges, None, iterations=1)
volEdges = cv2.Canny(volGray, 50, 100)
volEdges = cv2.dilate(volEdges, None, iterations=1)
volEdges = cv2.erode(volEdges, None, iterations=1)

_, rawCntrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# Sort the contours as they are in the picture
(cntrs, _) = contours.sort_contours(rawCntrs)

refCnt = cntrs[0]
refCntArea = cv2.contourArea(refCnt)
refUnitPerPix = refArea / refCntArea

refCntBox = cv2.minAreaRect(refCnt)
refCntBox = cv2.cv.BoxPoints(refCntBox) if imutils.is_cv2() else cv2.boxPoints(refCntBox)
refCntBox = np.array(refCntBox, dtype="int")
refCntBox = perspective.order_points(refCntBox)

cv2.drawContours(img, refCnt, -1, (255, 0, 0), 2)
cv2.putText(img, str(refArea) + ' ' + refAreaUnit, (int(refCntBox[0][0]) - 20, int(refCntBox[0][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

objCntrs = cntrs[1:]

for cnt in objCntrs:
    cntrArea = cv2.contourArea(cnt)

    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    areaObjHeightPx = box[0][1] - box[1][1]

    # Get the frontal view object's area
    cntrArea = cv2.contourArea(cnt)
    cntrArea = cntrArea * refUnitPerPix # Convert to units

    # cv2.drawContours(img, cnt, -1, (255, 0, 0), 2)
    # cv2.putText(img, str(cntrArea)  + ' ' + refAreaUnit, (int(box[0][0]) - 20, int(box[0][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# Calculate object's volume
_, rawVolCntrs, _ = cv2.findContours(volEdges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
(volCntrs, _) = contours.sort_contours(rawVolCntrs)

volRefCnt = volCntrs[0]

volRefCntrBox = cv2.minAreaRect(volRefCnt)
volRefCntrBox = cv2.cv.BoxPoints(volRefCntrBox) if imutils.is_cv2() else cv2.boxPoints(volRefCntrBox)
volRefCntrBox = np.array(volRefCntrBox, dtype="int")
volRefCntrBox = perspective.order_points(volRefCntrBox)

# Find the unit per pixel coeficient
cv2.drawContours(volImg, [volRefCntrBox.astype("int")], -1, (255, 0, 0), 2)
volRefPxWidth = distance.euclidean((volRefCntrBox[0][0], volRefCntrBox[0][1]), (volRefCntrBox[1][0], volRefCntrBox[1][1]))

volRefUnitPerPix = refVolWidth / volRefPxWidth
refVolAreaUnitPerPix = refVolArea * volRefUnitPerPix**2

areaObjHeightUnits = areaObjHeightPx * math.sqrt(refUnitPerPix) # Calculate the area object's height in units to avoid not matching pixel to unit ratios in frontal and side images
areaObjHeightVolPx = areaObjHeightUnits / volRefUnitPerPix

volObjCntrs = volCntrs[1:]
optimizedVolume = roughVolume = 0
volMidYPoint = 0

for cnt in volObjCntrs:
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    volRefPxWidth = distance.euclidean((box[0][0], box[0][1]), (box[3][0], box[3][1]))
    
    projArea = areaObjHeightVolPx * volRefPxWidth
    volObjArea = cv2.contourArea(cnt)

    projAreaToRealAreaCoef = volObjArea / projArea

    # Calculate the projected (rough) volume and with removed empty spaces (optimized)
    roughVolumePx = volRefPxWidth * cntrArea
    roughVolume = roughVolumePx * volRefUnitPerPix
    optimizedVolume = roughVolume * projAreaToRealAreaCoef
    
    print(str(optimizedVolume).replace('.', ','))
    
    # cv2.drawContours(volImg, cnt, -1, (255, 0, 0), 2)
    # cv2.putText(volImg, str(cntrArea)  + ' ' + refAreaUnit, (int(box[0][0]) - 20, int(box[0][1]) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# cv2.imshow('area', img)
# cv2.imshow('volume', volImg)
# cv2.waitKey(0)
