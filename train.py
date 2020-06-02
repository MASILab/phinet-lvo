'''
Samuel Remedios
NIH CC CNRM
Train PhiNet to classify MRI as T1, T2, FLAIR

==> adapted by Sneha Lingam to train on data that was 
previously preprocessed and load using a data generator
'''
## Note: use bash cmd 'nvidia-smi' or 'watch "nvidia-smi"'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

## set seeds for reproducibility
np.random.seed(13)
from tensorflow import set_random_seed
set_random_seed(13)

import shutil
import sys
import json
from sklearn.utils import shuffle
from datetime import datetime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import model_from_json
from models.phinet import phinet
from utils.utils import parse_args, now, prep_datagenerator, DataGenerator
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    ############### DIRECTORIES ###############

    results = parse_args("train")
    # if results.GPUID == None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    TRAIN_DIR = os.path.abspath(os.path.expanduser(results.TRAIN_DIR))
    VAL_DIR = os.path.abspath(os.path.expanduser(results.VAL_DIR))
    # CUR_DIR = os.path.abspath(
    #     os.path.expanduser(
    #         os.path.dirname(__file__)
    #     )
    # )
    # REORIENT_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "reorient.sh")
    # ROBUSTFOV_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "robustfov.sh")

    WEIGHT_DIR = os.path.abspath(os.path.expanduser(results.OUT_DIR))
    CSVLOG_DIR = os.path.abspath(os.path.expanduser(results.OUT_DIR))

    # PREPROCESSED_DIR = os.path.join(TRAIN_DIR, "preprocess")
    # if not os.path.exists(PREPROCESSED_DIR):
    #     os.makedirs(PREPROCESSED_DIR)

    ############### PREPROCESSING ###############

    # classes = results.classes.replace(" ","").split(',')

    # preprocess_dir(TRAIN_DIR, PREPROCESSED_DIR,
    #                REORIENT_SCRIPT_PATH, ROBUSTFOV_SCRIPT_PATH,
    #                classes,
    #                results.numcores,
    #                verbose=0)

    ############### DATA IMPORT ###############

    # X, y, filenames, num_classes, img_shape = load_data(PREPROCESSED_DIR, classes)
    #
    # print("Finished data processing")

    ## load entire dataset into RAM (max 70)
    # X, y, filenames, num_classes, img_shape = load_data(TRAIN_DIR, classes)

    classes = results.classes.replace(" ","").split(',')

    ############### MODEL SELECTION ###############

    LR = 1e-4
    LOAD_WEIGHTS = False
    MODEL_NAME = "phinet_model_" + now()
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME+".json")

    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    if LOAD_WEIGHTS:
        weight_files = os.listdir(WEIGHT_DIR)
        weight_files.sort()
        best_weights = os.path.join(WEIGHT_DIR, weight_files[-1])
        with open(MODEL_PATH) as json_data:
            model = model_from_json(json.load(json_data))
        model.load_weights(best_weights)
    else:
        model = phinet(n_classes=len(classes), learning_rate=LR)

    # save model architecture to file
    json_string = model.to_json()
    with open(MODEL_PATH,'w') as f:
        json.dump(json_string, f)

    ############### CALLBACKS ###############

    callbacks_list = []

    # Checkpoint
    WEIGHT_NAME = MODEL_NAME.replace("model","weights") + "_" + now()+"-epoch-{epoch:04d}-val_accuracy-{val_accuracy:.4f}.hdf5"
    fpath = os.path.join(WEIGHT_DIR, WEIGHT_NAME)
    checkpoint = ModelCheckpoint(fpath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=False,#True,
                                 mode='max',
                                 save_weights_only=True)
    callbacks_list.append(checkpoint)

    ## Dynamic Learning Rate
    #dlr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5,
    #                        mode='max', verbose=1, cooldown=5, min_lr=1e-8)
    #callbacks_list.append(dlr)

    # Early Stopping, used to quantify convergence
    # convergence is defined as no improvement by 1e-4 for 10 consecutive epochs
    #es = EarlyStopping(monitor='loss', min_delta=0, patience=10)
    #es = EarlyStopping(monitor='loss', min_delta=1e-8, patience=10)
    # The code continues even if the validation/training accuracy reaches 1, but loss is not.
    # For a classification task, accuracy is more important. For a regression task, loss is important
    # es = EarlyStopping(monitor='accuracy', min_delta=1e-8, patience=20)
    es = EarlyStopping(monitor='accuracy', min_delta=1e-2, patience=20)
    callbacks_list.append(es)

    # CSVLogger: to save epoch results
    csvlog_file = os.path.join(CSVLOG_DIR, "trainlog.csv")
    csvlog = CSVLogger(csvlog_file)
    callbacks_list.append(csvlog)


    BATCH_SIZE=2

    TB_DIR = os.path.join(os.path.abspath(os.path.expanduser(results.OUT_DIR)), 'tblogs')
    tb = TensorBoard(log_dir=TB_DIR,
                     batch_size=BATCH_SIZE,
                     update_freq='epoch')
    callbacks_list.append(tb)

    ############### TRAINING ###############
    # the number of epochs is set high so that EarlyStopping can be the terminator
    NB_EPOCHS = 1000

    # BATCH_SIZE = 2

    ## if previously loaded entire dataset into RAM (max 70)
    # model.fit(X, y, epochs=NB_EPOCHS, validation_split=valfromtrain_split,
    #           batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

    ## use generators to load each batch at a timeogging
    t_labels, t_list_IDs, t_dim = prep_datagenerator(TRAIN_DIR, classes)
    v_labels, v_list_IDs, v_dim = prep_datagenerator(VAL_DIR, classes)
    if t_dim != v_dim:
        print('dimension mismatch between train & val sets')
    print("\n\nTraining on", len(t_list_IDs), "examples")
    print("Validation on", len(v_list_IDs), "examples")
    print("Classes:", classes)
    print("Dimensions:", t_dim)
    print("Batch size:", BATCH_SIZE)
    print("\n\n")
    params = {'batch_size':BATCH_SIZE, 'n_classes':len(classes), 'dim':t_dim} # ,'n_channels'=1, 'shuffle'=True}
    train_generator = DataGenerator(dir_files=TRAIN_DIR, labels=t_labels, list_IDs=t_list_IDs, **params)
    val_generator = DataGenerator(dir_files=VAL_DIR, labels=v_labels, list_IDs=v_list_IDs, **params)
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        # use_multiprocessing=True,
                        use_multiprocessing=False,
                        # workers=6,
                        epochs=NB_EPOCHS,
                        callbacks=callbacks_list,
                        verbose=1)
                        ## note shuffle=True by default in DataGenerator and fit_generator

    # shutil.rmtree(PREPROCESSED_DIR)
    K.clear_session()
