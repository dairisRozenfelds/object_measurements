import cv2
import numpy as np
import urllib.request
import imutils
import pandas as pd
from imutils import contours
from imutils import perspective

imagesToMeasure = [
    {
        'folder': 'test_images/2d/10pr',
        'filename': '1_prieksa.jpg',
        'ref_area': 400,
    },
    {
        'folder': 'test_images/2d/10pr',
        'filename': '2_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/10pr',
        'filename': '3_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/10pr',
        'filename': '4_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/10pr',
        'filename': '5_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/10pr',
        'filename': '6_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/10pr',
        'filename': '7_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/10pr',
        'filename': '8_prieksa.jpg',
        'ref_area': 5625
    },
        {
        'folder': 'test_images/2d/5pr',
        'filename': '1_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/5pr',
        'filename': '2_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/5pr',
        'filename': '3_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/5pr',
        'filename': '4_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/5pr',
        'filename': '5_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/5pr',
        'filename': '6_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/5pr',
        'filename': '7_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/5pr',
        'filename': '8_prieksa.jpg',
        'ref_area': 5625
    },
        {
        'folder': 'test_images/2d/2pr',
        'filename': '1_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/2pr',
        'filename': '2_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/2pr',
        'filename': '3_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/2pr',
        'filename': '4_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/2pr',
        'filename': '5_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/2pr',
        'filename': '6_prieksa.jpg',
        'ref_area': 5625
    },
    {
        'folder': 'test_images/2d/2pr',
        'filename': '7_prieksa.jpg',
        'ref_area': 400
    },
    {
        'folder': 'test_images/2d/2pr',
        'filename': '8_prieksa.jpg',
        'ref_area': 5625
    },
]

areaInfo = {}

for imageInfo in imagesToMeasure:
    imageFolder = imageInfo['folder']
    imageFilename = imageInfo['filename']
    imagePath = imageFolder + '/' + imageFilename

    image = cv2.imread(imagePath)

    scale_percent = 60 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    refArea = imageInfo['ref_area'] # Reference area
    refAreaUnit = 'cm^2' # Reference area measurement units

    img = cv2.resize(image.copy(), dim)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 100)

    _, cntrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort the contours as they are in the picture
    (cntrs, _) = contours.sort_contours(cntrs)

    refCnt = cntrs[0]
    refCntArea = cv2.contourArea(refCnt)

    if (refCntArea == 0):
        print('Could not distinguish reference object from picture ' + imageFolder + '/' + imageFilename)
        continue

    refUnitPerPix = refArea / refCntArea

    refCntBox = cv2.minAreaRect(refCnt)
    refCntBox = cv2.cv.BoxPoints(refCntBox) if imutils.is_cv2() else cv2.boxPoints(refCntBox)
    refCntBox = np.array(refCntBox, dtype="int")
    refCntBox = perspective.order_points(refCntBox)

    cv2.drawContours(img, refCnt, -1, (255, 0, 0), 2)
    cv2.putText(img, str(refArea) + ' ' + refAreaUnit, (int(refCntBox[3][0]) - 20, int(refCntBox[3][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    objCntrs = cntrs[1:]

    for cnt in objCntrs:
        cntrArea = cv2.contourArea(cnt)

        if cntrArea < 100:
            continue

        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        cntrArea = cv2.contourArea(cnt)
        cntrArea = cntrArea * refUnitPerPix

        cv2.drawContours(img, cnt, -1, (255, 0, 0), 2)
        cv2.putText(img, str(cntrArea)  + ' ' + refAreaUnit, (int(box[0][0]) - 20, int(box[0][1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    areaInfo[imagePath] = cntrArea

    cv2.imwrite('results/' + imagePath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print('Found area for ' + imagePath)

df = pd.DataFrame(areaInfo.items(), columns=['Filename', 'Area (cm2)'])
df.to_excel('results/area_info.xlsx')
