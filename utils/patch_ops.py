'''
Samuel Remedios
NIH CC CNRM
Patch operations
'''

import os
import random
from tqdm import *
import numpy as np
import nibabel as nib
from .display import show_image
from keras.utils import to_categorical
from sklearn.utils import shuffle

def load_patch_data(data_dir, patch_size, classes=None, num_patches=100, verbose=0):
    '''
    Loads in datasets and returns the labeled preprocessed patches for use in the model.

    Determines the number of classes for the problem and assigns labels to each class,
    sorted alphabetically.

    Params:
        - data_dir: string, path to all training class directories
        - preprocess_dir: string, path to destination for robustfov files
        - patch_size: 3-element tuple of integers, size of patches to use for training
        - labels_known: boolean, True if we know the labels, such as for training or
                                 validation.  False if we do not know the labels, such
                                 as loading in data to classify in production
    Returns:
        - data: list of 3D ndarrays, the patches of images to use for training
        - labels: list of 1D ndarrays, one-hot encoding corresponding to classes
        - all_filenames: list of strings, corresponding filenames for use in validation/test
    '''

    labels = []

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    if classes is None:
        all_filenames = []
        data = []
        filenames = [x for x in os.listdir(data_dir) 
                if not os.path.isdir(os.path.join(data_dir,x))]
        filenames.sort()

        for f in tqdm(filenames):
            img = nib.load(os.path.join(robustfov_dir, f)).get_data()
            patches = get_patches(img, patch_size, num_patches)

            for patch in tqdm(patches):
                data.append(patch)
                all_filenames.append(f)

        print("A total of {} patches collected.".format(len(data)))

        data = np.array(data)

        return data, all_filenames

    #################### TRAINING OR VALIDATION ####################

    # determine number of classes
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(class_directories)

    # set up all_filenames and class_labels to speed up shuffling
    all_filenames = []
    class_labels = {}
    i = 0
    for class_directory in class_directories:
        if not os.path.basename(class_directory) in classes:
            print("{} not in {}; omitting.".format(
                os.path.basename(class_directory),
                classes))
            continue

        class_labels[os.path.basename(class_directory)] = i
        i += 1
        for filename in os.listdir(class_directory):
            filepath = os.path.join(class_directory, filename)
            all_filenames.append(filepath)


    img_shape = patch_size
    num_items = len(all_filenames) * num_patches 
    data = np.zeros(shape=((num_items,) + img_shape + (1,)), dtype=np.uint8)
    labels = np.zeros((num_items,) + (num_classes,), dtype=np.uint8)
    filenames = [None] * num_items

    print(data.shape)
    print(labels.shape)

    all_filenames = shuffle(all_filenames, random_state=0)
    indices = np.arange(num_items)
    indices = shuffle(indices, random_state=0)
    cur = 0

    for f in tqdm(all_filenames):
        img = nib.load(f).get_data()
        patches = get_patches(img, patch_size, num_patches)

        cur_label = f.split(os.sep)[-2]

        for patch in patches:
            # graph patches to ensure proper collection
            if verbose:
                middle_slice_idx = patch.shape[2]//2
                show_image(patch[:,:,middle_slice_idx,0])

            data[indices[cur]] = patch
            labels[indices[cur]] = to_categorical(
                class_labels[cur_label], num_classes=num_classes)

            filenames[indices[cur]] = f
            cur += 1



    print("A total of {} patches collected.".format(len(data)))

    labels = np.array(labels, dtype=np.uint8)
    print(data.shape)
    print(labels.shape)

    return data, labels, filenames, num_classes, data[0].shape


def get_patches(img, patch_size, num_patches=100, num_channels=1):
    '''
    Gets num_patches 3D patches of the input image for classification.

    Patches may overlap.

    The center of each patch is some random distance from the center of
    the entire image, where the random distance is drawn from a Gaussian dist.

    Params:
        - img: 3D ndarray, the image data from which to get patches
        - patch_size: 3-element tuple of integers, size of the 3D patch to get
        - num_patches: integer (default=100), number of patches to retrieve
        - num_channels: integer (default=1), number of channels in each image
    Returns:
        - patches: ndarray of 4D ndarrays, the resultant 3D patches by their channels
    '''
    # set random seed and variable params
    random.seed()
    mu = 0
    sigma = 10

    # find center of the given image
    # bias center towards top quarter of brain
    center_coords = [x//2 for x in img.shape]

    # find num_patches random numbers as distances from the center
    patches = np.empty((num_patches, *patch_size, num_channels))
    for i in range(num_patches):
        horizontal_displacement = int(random.gauss(mu, sigma))
        depth_displacement = int(random.gauss(mu, sigma))
        # deviate half as much vertically
        vertical_displacement = int(random.gauss(mu, sigma//2))

        # current center coords
        c = [
            center_coords[0] + horizontal_displacement,
            center_coords[1] + depth_displacement,
            center_coords[2] + vertical_displacement
        ]

        if c[0]+patch_size[0]//2 > img.shape[0] or c[0]-patch_size[0]//2 < 0 or\
            c[1]+patch_size[1]//2 > img.shape[1] or c[1]-patch_size[1]//2 < 0 or\
                c[2]+patch_size[2]//2 > img.shape[2] or c[2]-patch_size[2]//2 < 0:
            continue

        # get patch
        patch = img[
            c[0]-patch_size[0]//2:c[0]+patch_size[0]//2+1,
            c[1]-patch_size[1]//2:c[1]+patch_size[1]//2+1,
            c[2]-patch_size[2]//2:c[2]+patch_size[2]//2+1,
        ]

        if patch.shape != patch_size:
            continue

        #TODO: currently only works for one channel
        patches[i,:,:,:,0] = patch

    return patches