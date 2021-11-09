# %%
import imghdr
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compare_images(img_dir, show_imgs=False, similarity="high", compression=50, ranked=True):
    """
    :param ranked: whether or not the images have been ranked with NIMA
    :param img_dir: folder to search for duplicate/similar images.
    :param show_imgs: True = shows the duplicate/similar images found in output.
                      False = doesn't show found images.
    :param similarity: "high" = searches for duplicate images, more precise.
                       "low" = finds similar images.
    :param compression: compression in px (height x width) of the images before being compared;
                        the higher the compression, the more computation time required
    :return: lower_quality: set containing names of lower quality images
    """

    # list where the found duplicate/similar images are stored
    duplicates_list = []
    lower_quality = []

    imgs_matrix = create_imgs_matrix(img_dir, compression)

    # search for similar images
    if similarity == "high":
        ref = 12000
    # search for 1:1 duplicate images
    else:
        ref = 200

    main_img = 0
    compared_img = 1
    nrows, ncols = compression, compression
    srow_A = 0
    erow_A = nrows
    srow_B = erow_A
    erow_B = srow_B + nrows

    while erow_B <= imgs_matrix.shape[0]:
        while compared_img < (len(image_files)):
            # select two images from imgs_matrix
            imgA = imgs_matrix[srow_A: erow_A, 0: ncols]  # columns
            imgB = imgs_matrix[srow_B: erow_B, 0: ncols]  # columns
            # compare the images
            rotations = 0
            while image_files[main_img] not in duplicates_list and rotations <= 3:
                if rotations != 0:
                    imgB = rotate_img(imgB)
                err = mse(imgA, imgB)
                if err < ref:
                    if show_imgs:
                        show_img_figs(imgA, imgB, err)
                        show_file_info(compared_img, main_img)
                    add_to_list(image_files[main_img], duplicates_list)
                    check_img_quality(image_files[main_img], image_files[compared_img], lower_quality)
                rotations += 1
            srow_B += nrows
            erow_B += nrows
            compared_img += 1

        srow_A += nrows
        erow_A += nrows
        srow_B = erow_A
        erow_B = srow_B + nrows
        main_img += 1
        compared_img = main_img + 1

    print(f'Found {len(duplicates_list)} duplicate images pairs in {len(image_files)} total images.\n')
    print("The following files have lower quality:")
    print(set(lower_quality))
    return list(set(lower_quality))


# Function that searches the folder for image files, converts them to a matrix
def create_imgs_matrix(directory, compression):
    global image_files, imgs_matrix
    image_files = []
    # create list of all files in directory
    folder_files = [filename for filename in os.listdir(directory)]

    # create images matrix
    counter = 0
    for filename in folder_files:
        directory_and_filename = os.path.join(directory, filename)
        if not os.path.isdir(directory_and_filename) and imghdr.what(directory_and_filename):
            img = cv2.imdecode(np.fromfile(directory_and_filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if type(img) == np.ndarray:
                img = img[..., 0:3]
                img = cv2.resize(img, dsize=(compression, compression), interpolation=cv2.INTER_CUBIC)
                if counter == 0:
                    imgs_matrix = img
                    image_files.append(filename)
                    counter += 1
                else:
                    try:
                        imgs_matrix = np.concatenate((imgs_matrix, img))
                        image_files.append(filename)
                    except:
                        continue
    return imgs_matrix


# Function that calulates the mean squared error (mse) between two image matrices
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Function that plots two compared image files and their mse
def show_img_figs(imageA, imageB, err):
    fig = plt.figure()
    plt.suptitle("MSE: %.2f" % err)
    # plot first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # plot second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()


# Function for rotating an image matrix by a 90 degree angle
def rotate_img(image):
    image = np.rot90(image, k=1, axes=(0, 1))
    return image


# Function for printing filename info of plotted image files
def show_file_info(compared_img, main_img):
    print("Duplicate file: " + image_files[main_img] + " and " + image_files[compared_img])


# Function for appending items to a list
def add_to_list(filename, list_):
    list_.append(filename)


# Function for checking the quality of compared images, appends the lower quality image to the list
def check_img_quality(imageA, imageB, list_):
    rankA = int(imageA.split('-')[0].split('(')[0])
    rankB = int(imageB.split('-')[0].split('(')[0])
    if rankA < rankB:
        add_to_list(imageB, list_)
    else:
        add_to_list(imageA, list_)


# Check the image size if there are no rankings
def check_img_size(imageA, imageB, list_):
    size_imgA = os.stat(directory + imageA).st_size
    size_imgB = os.stat(directory + imageB).st_size
    if size_imgA > size_imgB:
        add_to_list(imageB, list_)
    else:
        add_to_list(imageA, list_)
