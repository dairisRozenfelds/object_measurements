import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('images/2.jpg')

def sobel():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, -1, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, -1, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)    
        
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    w = plt.figure()

    w.add_subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    w.add_subplot(2, 2, 2)
    plt.imshow(abs_grad_x, 'gray')
    plt.axis('off')

    w.add_subplot(2, 2, 3)
    plt.imshow(abs_grad_y, 'gray')
    plt.axis('off')

    w.add_subplot(2, 2, 4)
    plt.imshow(grad, 'gray')
    plt.axis('off')

    plt.show(block=True)

def prewitt():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    grad_x = cv2.filter2D(gray, -1, kernel_x)
    grad_y = cv2.filter2D(gray, -1, kernel_y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
        
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    w = plt.figure()

    w.add_subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    w.add_subplot(2, 2, 2)
    plt.imshow(abs_grad_x, 'gray')
    plt.axis('off')

    w.add_subplot(2, 2, 3)
    plt.imshow(abs_grad_y, 'gray')
    plt.axis('off')

    w.add_subplot(2, 2, 4)
    plt.imshow(grad, 'gray')
    plt.axis('off')

    plt.show(block=True)

def canny():
    cv2.namedWindow('canny')
    cv2.createTrackbar('T1', 'canny' , 0, 900, on_canny_update)
    cv2.createTrackbar('T2', 'canny' , 0, 900, on_canny_update)
    cv2.namedWindow('original')
    cv2.imshow('original', img)
    cv2.waitKey()

def on_canny_update(val):
    t1 = cv2.getTrackbarPos('T1', 'canny')
    if t1 < 1:
        cv2.setTrackbarPos('T1', 'canny', 1)

    t2 = cv2.getTrackbarPos('T2', 'canny')
    if t2 < 1:
        cv2.setTrackbarPos('T1', 'canny', 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    res = cv2.Canny(blurred, t1, t2)

    cv2.imshow('canny', res)

# sobel()
# prewitt()
canny()
