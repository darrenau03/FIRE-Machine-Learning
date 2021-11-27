import os
from glob import glob

import cv2
import numpy as np
import xlsxwriter
from tqdm import tqdm

# printToExcel = False

OUTPUT_DIRECTORY = "output"
# DIRECTORY = "validation_images"
DIRECTORY = "big_train_dataset_images"
FILE_TYPE = "*.jpg"


def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

# convert image path to output image binary
def convertImage2_0_2(path):
    # path of image
    img = cv2.imread(path)

    # convert to grey scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur to reduce noise
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # sobel filtering matrices
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    # get dimensions
    [rows, columns] = np.shape(img_gray)

    # matrix/image where magnitude is based on gradient of initial image
    gradient_img = np.zeros(shape=(rows, columns))

    # sweep the image in both x and y directions and compute the output
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, img_gray[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, img_gray[i:i + 3, j:j + 3]))  # y direction
            gradient_img[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

    if printToExcel:
        wb = xlsxwriter.Workbook('../FIRE/demo.xlsx')
        sheet1 = wb.add_worksheet('derivative')
        sheet2 = wb.add_worksheet('value')
        for i in range(len(gradient_img)):
            for j in range(len(gradient_img[i])):
                sheet1.write(i, j, gradient_img[i][j])
                sheet2.write(i, j, int(img_gray[i][j]))

    # convert magnitude based gradient image to binary
    binary_img = np.zeros(shape=(rows, columns), dtype=np.uint8)

    # mean
    average = 0
    for i in range(len(gradient_img)):
        for j in range(len(gradient_img[i])):
            average += gradient_img[i][j]
    average = average / (rows * columns)

    # median
    arr = []
    for i in range(len(gradient_img)):
        for j in range(len(gradient_img[i])):
            arr.append(gradient_img[i][j])
    arr.sort()
    middle = arr[len(arr) // 2]

    # threshold = middle * 9 / 10
    threshold = 20

    for i in range(len(gradient_img)):
        for j in range(len(gradient_img[i])):
            if gradient_img[i][j] > int(threshold):
                binary_img[i][j] = 255
            else:
                binary_img[i][j] = 0

    # blurring to reduce noise again
    binary_img = cv2.GaussianBlur(binary_img, (5, 5), 0)

    '''
    Note: i'm not entirely sure why, but without this line above, the contour process doesn't work
    I guess, rigid edges has a really high impact on the computers ability to find contours
    '''

    _, binary = cv2.threshold(binary_img, 100, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    '''
    I still need to do more work on understanding the threshold and hierarchy inputs that affect the ouput
    Currently, "THRESH_BINARY_INV" is the only one that doesn't ignore all the contours in the middle
    and am not fully sure why
    '''
    # creates an output image of only green pixels
    output = np.ones(shape=(rows, columns, 3), dtype=np.uint8)
    for i in range(len(gradient_img)):
        for j in range(len(gradient_img[i])):
            output[i][j] = np.array([0, 0, 0])

    # fills out the red areas (by gradient)
    # for i in range(len(gradient_img)):
    #     for j in range(len(gradient_img[i])):
    #         if gradient_img[i][j] > int(average * 15 / 10):
    #             output[i][j] = np.array([0, 0, 255])

    # draws contours onto output

    if printToExcel:
        sheet3 = wb.add_worksheet('contours')
    for i in contours:
        big = 0
        small = 255
        for j in i:
            big = max(
                big, int(
                    img_gray[int(j[0][1])][int(j[0][0])]
                )
            )
            small = min(
                small, int(
                    img_gray[int(j[0][1])][int(j[0][0])])
            )
        if printToExcel:
            try:
                sheet3.write(int(j[0][1]), int(j[0][0]), "1")
            except:
                pass

        if cv2.contourArea(i) > 40:
            # cv2.fillPoly(output, pts=[i], color=(0, 0, 255))
            if big - small > 10:
                # print(str(big) + " " + str(small))
                # cv2.drawContours(output, i, -1, (0, 0, 255), thickness=-1)
                cv2.fillPoly(output, pts=[i], color=(255, 255, 255))

    try:
        wb.close()
    except:
        pass

    return output

def convertImage1_0_1(path):
    img = cv2.imread(path)  # path of image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # blur to reduce noise


    # sobel filtering
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(img_gray)  # we need to know the shape of the input grayscale image
    sobel_filtered_image = np.zeros(
        shape=(rows, columns))  # initialization of the output image array (all elements are 0)

    # sweep" the image in both x and y directions and compute the output
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, img_blur[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, img_blur[i:i + 3, j:j + 3]))  # y direction
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

    # create output image
    sobel_colored_image = np.zeros(
        shape=(rows, columns, 3))
    for i in range(len(sobel_filtered_image)):
        for j in range(len(sobel_filtered_image[i])):
            if sobel_filtered_image[i][j] > 60:
                sobel_colored_image[i][j] = [255]  # green
            else:
                sobel_colored_image[i][j] = [0]  # red
    return sobel_colored_image

def main():
    dir1 = os.path.join(OUTPUT_DIRECTORY, "image")
    dir2 = os.path.join(OUTPUT_DIRECTORY, "mask")

    create_dir(dir1)
    create_dir(dir2)
    path = os.path.join(DIRECTORY, FILE_TYPE)
    images = sorted(glob(path))
    c = 0

    for i in tqdm(images):
        path1 = os.path.join(OUTPUT_DIRECTORY, "image", ("image" + str(c) + ".png"))
        path2 = os.path.join(OUTPUT_DIRECTORY, "mask", ("image" + str(c) + ".png"))
        cv2.imwrite(path1, cv2.imread(i))
        cv2.imwrite(path2, convertImage1_0_1(i))
        c += 1


if __name__ == "__main__":


    main()
