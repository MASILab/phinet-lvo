'''
Samuel Remedios
NIH CC CNRM
Data processing script

==> adapted by Sneha Lingam
Train PhiNet with data that was previously preprocessed
'''

import os
import csv
import random
from tqdm import *
import argparse
import glob
import shutil
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
import sys
from datetime import datetime
from keras.utils import to_categorical, Sequence
from sklearn.utils import shuffle

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


def parse_args(session):
    '''
    Parse command line arguments.

    Params:
        - session: string, one of "train", "validate", or "test"
    Returns:
        - parse_args: object, accessible representation of args
    '''
    parser = argparse.ArgumentParser(
        description="Arguments for Training and Testing")

    if session == "train":
        # parser.add_argument('--datadir', required=True, action='store', dest='TRAIN_DIR',
        #                     help='Where the initial data is')
        parser.add_argument('--traindir', required=True, action='store', dest='TRAIN_DIR',
                            help='Where the preprocessed training data is')
        parser.add_argument('--valdir', required=True, action='store', dest='VAL_DIR',
                            help='Where the preprocessed validation data is')
        parser.add_argument('--outdir', required=True, action='store', dest='OUT_DIR',
                            help='Output directory where the trained models and logs are written')
        # parser.add_argument('--weightdir', required=True, action='store', dest='OUT_DIR',
        #                     help='Output directory where the trained models and logs are written')
        # parser.add_argument('--numcores', required=True, action='store', dest='numcores',
        #                     default='1', type=int,
        #                     help='Number of cores to preprocess in parallel with')
    elif session == "test":
        parser.add_argument('--infile', required=True, action='store', dest='INFILE',
                            help='Image to classify')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--result_dst', required=True, action='store', dest='OUTFILE',
                            help='Output filename (e.g. result.csv) to where the results are written')
    elif session == "validate":
        parser.add_argument('--datadir', required=True, action='store', dest='VAL_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        # parser.add_argument('--result_dst', required=True, action='store', dest='OUT_DIR',
        #                     help='Output directory where the results are written')
        parser.add_argument('--result_file', required=True, action='store', dest='OUTFILE',
                            help='Output filename (e.g. result.csv) where the results are written')
        # parser.add_argument('--numcores', required=True, action='store', dest='numcores',
        #                     default='1', type=int,
        #                     help='Number of cores to preprocess in parallel with')
    else:
        print("Invalid session. Must be one of \"train\", \"validate\", or \"test\"")
        sys.exit()

    parser.add_argument('--classes', required=True, action='store', dest='classes',
                        help='Comma separated list of all classes, CASE-SENSITIVE')
    # parser.add_argument('--gpuid', required=False, action='store', type=int, dest='GPUID',
    #                     help='For a multi-GPU system, the trainng can be run on different GPUs.'
    #                     'Use a GPU id (single number), eg: 1 or 2 to run on that particular GPU.'
    #                     '0 indicates first GPU.  Optional argument. Default is the first GPU.')
    # parser.add_argument('--delete_preprocessed_dir', required=False, action='store', dest='clear',
    #                     default='n', help='delete all temporary directories. Enter either y or n. Default is n.')

    return parser.parse_args()



def load_image(filename):
    img = nib.load(filename).get_data()
    img = np.reshape(img, (1,)+img.shape+(1,))
    #MAX_VAL = 255  # consistent maximum intensity in preprocessing

    # linear scaling so all intensities are in [0,1]
    #return np.divide(img, MAX_VAL)
    return  img

def get_classes(classes):
    '''
    Params:
        - classes: list of strings
    Returns:
        - class_encodings: dictionary mapping an integer to a class_string
    '''
    class_list = classes
    class_list.sort()

    class_encodings = {x: class_list[x] for x in range(len(class_list))}

    return class_encodings

def load_data(data_dir, classes=None):
    '''
    Loads in datasets and returns the labeled preprocessed patches for use in the model.

    Determines the number of classes for the problem and assigns labels to each class,
    sorted alphabetically.

    Params:
        - data_dir: string, path to all training class directories
        - task: string, one of modality, T1-contrast, FL-contrast'
        - labels_known: boolean, True if we know the labels, such as for training or
                                 validation.  False if we do not know the labels, such
                                 as loading in data to classify in production
    Returns:
        - data: list of 3D ndarrays, the patches of images to use for training
        - labels: list of 1D ndarrays, one-hot encoding corresponding to classes
        - all_filenames: list of strings, corresponding filenames for use in validation/test
        - num_classes: integer, number of classes
        - img_shape: ndarray, shape of an individual image
    '''

    labels = []

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    if classes is None:
        all_filenames = []
        filenames = [x for x in os.listdir(data_dir)
                     if not os.path.isdir(os.path.join(data_dir, x))]
        filenames.sort()

        for f in tqdm(filenames):
            img = nib.load(os.path.join(data_dir, f)).get_data()
            img = np.reshape(img, img.shape+(1,))
            data.append(img)
            all_filenames.append(f)

        data = np.array(data)

        return data, all_filenames

    #################### TRAINING OR VALIDATION ####################

    # determine number of classes
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(classes)

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

    img_shape = nib.load(all_filenames[0]).get_data().shape
    data = np.empty(shape=((len(all_filenames),) +
                           img_shape + (1,)), dtype=np.float32)

    # shuffle data
    all_filenames = shuffle(all_filenames, random_state=0)

    data_idx = 0  # pointer to index in data

    for f in tqdm(all_filenames):
        img = nib.load(f).get_data()
        img = np.asarray(img, dtype=np.float32)

        # place this image in its spot in the data array
        data[data_idx] = np.reshape(img, (1,)+img.shape+(1,))
        data_idx += 1

        cur_label = f.split(os.sep)[-2]
        labels.append(to_categorical(
            class_labels[cur_label], num_classes=num_classes))

    labels = np.array(labels, dtype=np.uint8)
    print(data.shape)
    print(labels.shape)
    return data, labels, all_filenames, num_classes, data[0].shape


def prep_datagenerator(dir_files, classes):
    '''
    For a given directory stored in the below format,
    prepares and returns parameters for DataGenerator class.
    +-- data/
    |   +-- train/
    |   |   +-- /my_class_1
    |   |   |   +-- my_class_1_file_1.nii.gz
    |   |   |   +-- my_class_1_file_2.nii.gz
    |   |   |   +-- my_class_1_file_n.nii.gz
    |   |   +-- /my_class_2
    |   |   |   +-- my_class_2_file_1.nii.gz
    |   |   |   +-- my_class_2_file_2.nii.gz
    |   |   |   +-- my_class_2_file_n.nii.gz
    |   +-- validation/
    |   |   +-- /my_class_1
    |   |   |   +-- my_class_1_file_1.nii.gz
    |   |   |   +-- my_class_1_file_2.nii.gz
    |   |   |   +-- my_class_1_file_n.nii.gz
    |   |   +-- /my_class_2
    |   |   |   +-- my_class_2_file_1.nii.gz
    |   |   |   +-- my_class_2_file_2.nii.gz
    |   |   |   +-- my_class_2_file_n.nii.gz
    '''
    labels=dict()
    for label in classes:
        for x in os.walk(os.path.join(dir_files,label)):
            labels.update(dict.fromkeys(x[2], label))
    list_IDs = list(labels.keys())

    img = nib.load(os.path.join(dir_files,labels[list_IDs[0]],list_IDs[0])).get_data()
    dim = img.shape

    return labels, list_IDs, dim

class DataGenerator(Sequence):
    '''
    Generates batches of train or val data for Keras
    '''

    def __init__(self, dir_files, labels, list_IDs, n_classes=2, dim=(512,512,340),
                 batch_size=8, n_channels=1, shuffle=True):
        'Initialization'
        self.dir_files = dir_files
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.on_epoch_end()
        while len(dim)<3:
            dim += (1,)
        self.dim = dim

    def __len__(self):
        'Denotes # of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of files
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        # print('\non_epoch_end called, indexes shuffled\n')

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            #Load and store image    callbacks_list.append(es)
            fpath = os.path.join(self.dir_files,self.labels[ID],ID)
            img = nib.load(fpath).get_data()
            while len(img.shape)<4:
                img = np.reshape(img, img.shape+(1,))
            #img = np.reshape(img, img.shape+(1,))
            img = np.asarray(img, dtype=np.float32)
            X[i,] = img

            #Store class
            y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)


def record_results(csv_filename, args):

    filename, ground_truth, prediction, confidences = args

    if ground_truth is not None:
        fieldnames = [
            "filename",
            "ground_truth",
            "prediction",
            "confidences",
        ]


        # write to file the two sums
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w') as csvfile:
                fieldnames = fieldnames

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidences": confidences,
            })
    else:
        fieldnames = [# use_multiprocessing=True,
            "filename",
            "prediction",
            "confidences",
        ]


        # write to file the two sums
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w') as csvfile:
                fieldnames = fieldnames

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({# use_multiprocessing=True,
                "filename": filename,
                "prediction": prediction,
                "confidences": confidences,
            })

def now():
    '''
    Returns a string format of current time, for use in checkpoint filenaming
    '''
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
