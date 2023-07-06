import json

import os, cv2

from PIL import Image

import numpy as np

import json

import os, cv2

import math

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def image_hist(image_path: str):
    # def image_hist(img):

    """
An image histogram is a statistical table that reflects the distribution of pixels in an image.
Its horizontal axis represents the type of pixels in the image, which can be grayscale or color.
The vertical axis represents the total number of pixels or percentage of all pixels in the image for each color value.
An image is composed of pixels, so a histogram that reflects the distribution of pixels can
often be an important feature of the image.,The display method of the histogram
is dark and bright on the left, with the left side used to describe the darkness
of the image and the right side used to describe the brightness of the image.

param image_ Path: Pass in the image file to search for pixels
return:  No return value


    """

    # One dimensional histogram (single channel histogram)

    img = cv2.imread(image_path, 0)

    color = ('blue', 'green', 'red')

    # Use the plt Intrinsic function to directly draw

    global hist

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    plt.plot(hist, color="r")


def cal_res(img_path, image_name):
    tmp = []

    for i in range(256):
        tmp.append(0)

    val = 0

    k = 0

    res = 0


    I = Image.open(img_path)


    greyIm = I.convert('L')


    img = np.array(greyIm)

    x, y = img.shape

    dst = np.zeros([x, y])

    for i in range(x):

        for j in range(y):

            if img[i, j] > 150:

                dst[i, j] = 255

            else:

                dst[i, j] = 0

    m = 0

    for i in range(dst.shape[0]):

        for j in range(dst.shape[1]):

            # print(dst[i,j])

            if dst[i, j] == 255:
                m += 1

    for i in range(len(img)):

        for j in range(len(img[i])):
            val = img[i][j]

            tmp[val] = float(tmp[val] + 1)

            k = float(k + 1)

    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)

    tmp = np.array(tmp)

    for i in range(len(tmp)):

        if (tmp[i] == 0):

            res = res

        else:

            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))


    return res, dst


def getFileList(dir, Filelist, ext=None):
    """
    Obtain a list of files in the folder and its subfolders.
    Input dir: Folder root directory.
    Input ext: Extension.
    Return: File path list
    """
    
    newDir = dir

    if os.path.isfile(dir):

        if ext is None:

            Filelist.append(dir)

        else:

            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)

            getFileList(newDir, Filelist, ext)

    return Filelist


def visualization_bbox1(num_image, json_path, img_path):
    with open(json_path) as annos:
        annotation_json = json.load(annos)

        image_name = annotation_json['images'][num_image - 1]['file_name']  

        id = annotation_json['images'][num_image - 1]['id']  # Read Image ID

        image_path = os.path.join(img_path, str(image_name).zfill(5))  

        res, dst = cal_res((str(image_path)), str(image_name))

        image = cv2.imread(image_path, 1)  

        img = cv2.imread(image_path, 0)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        return hist, res


train_json = '/train1/train.json'
test_json = '/test1/test.json'
val_json = '/val1/val.json'

train_path = '/train1/train_image/'
test_path = '/test1/test_image/'
val_path = '/val1/val_image/'

imglist = getFileList(train_path, [], 'jpg')

# cal_res()

res1 = []

for imgpath in imglist:

    d = d + 1

    hist, res = visualization_bbox1(d, train_json, train_path)

    res1.append(res)

    print(imgpath)

    print("res:", res)
    if res >= 2 and res < 3:

        plt.plot(hist, color="y", alpha=1, linestyle='--', linewidth=0.5, marker='o', markersize=1)

    elif res >= 3 and res < 4:

        plt.plot(hist, color="g", alpha=1, linestyle='-.', linewidth=0.5, marker='>', markersize=1)

    elif res >= 4 and res < 5:

            plt.plot(hist, color="r", alpha=1, linestyle=':', linewidth=0.5, marker='*', markersize=1)
    elif res >= 5 and res < 6:

        plt.plot(hist, color="b", alpha=1, linestyle=':', linewidth=0.5, marker='x', markersize=1)

    elif res >= 6 and res < 7 :

        plt.plot(hist, color="c", alpha=1, linestyle=':', linewidth=0.5, marker='D', markersize=1)

    elif res >= 7:

        plt.plot(hist, color="m", alpha=1, linestyle=':', linewidth=0.5, marker='1', markersize=1)

    plt.legend()

print(max(res1), min(res1))

plt.legend(loc=1, fontsize='10')
plt.xlabel('Pixel')
plt.ylabel('Number')
plt.savefig('/media/ExtDisk/yxt/IMG/pic.jpg', dpi=500)
