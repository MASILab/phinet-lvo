'''
Samuel Remedios
NIH CC CNRM
Predict contrast of an image.

==> adapted by Sneha Lingam
Train PhiNet to classify CTA as having LVO or not
using data that was previously preprocessed
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import time
import shutil
import json
from operator import itemgetter
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from models.phinet import phinet
from utils.utils import load_data, now, parse_args, get_classes, load_image, record_results, prep_datagenerator #, preprocess_dir
from keras.models import load_model, model_from_json
from keras import backend as K
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    ############### DIRECTORIES ###############

    results = parse_args("validate")

    # if results.GPUID == None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    VAL_DIR = os.path.abspath(os.path.expanduser(results.VAL_DIR))

    # CUR_DIR = os.path.abspath(
    #     os.path.expanduser(
    #         os.path.dirname(__file__)
    #     )
    # )

    # REORIENT_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "reorient.sh")
    # ROBUSTFOV_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "robustfov.sh")
    #
    # PREPROCESSED_DIR = os.path.join(VAL_DIR, "preprocess")
    # if not os.path.exists(PREPROCESSED_DIR):
    #     os.makedirs(PREPROCESSED_DIR)

    with open(results.model) as json_data:
        model = model_from_json(json.load(json_data))
    model.load_weights(results.weights)

    ############### PREPROCESSING ###############

    classes = results.classes.replace(" ","").split(',')

    # preprocess_dir(VAL_DIR, PREPROCESSED_DIR,
    #                REORIENT_SCRIPT_PATH, ROBUSTFOV_SCRIPT_PATH,
    #                classes,
    #                results.numcores)

    # get class encodings
    class_encodings = get_classes(classes)

    ############### DATA IMPORT ###############

    # X, y, filenames, num_classes, _ = load_data(PREPROCESSED_DIR, classes)
    #
    # print("Test data loaded.")
    #
    # X, y, filenames, num_classes, _ = load_data(VAL_DIR, classes)
    # print("Validation data loaded.")

    y_dict, filenames, dim = prep_datagenerator(VAL_DIR, classes)

    ############### PREDICT ###############

    PRED_DIR = os.path.dirname(results.OUTFILE)
    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)
    BATCH_SIZE = 1

    # # make predictions with best weights and save results
    # preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

    # track overall accuracy
    acc_count = len(set(filenames))
    total = len(set(filenames))

    ############### RECORD RESULTS ###############
    # for filename, pred, ground_truth in zip(filenames, preds, y):
    for filename in tqdm(filenames):
        ground_truth = y_dict[filename]

        # load image, make prediction with best weights and save result
        filepath = os.path.join(VAL_DIR,ground_truth,filename)
        X = load_image(filepath)
        pred = model.predict(X, batch_size=1, verbose=1)

        confidences = ";".join("{:.2f}".format(x*100) for x in pred[0])

        max_idx, max_val = max(enumerate(pred[0]), key=itemgetter(1))
        # max_true, val_true = max(enumerate(ground_truth), key=itemgetter(1))
        pred_class = class_encodings[max_idx]
        # gt_class = class_encodings[max_true]

        # if max_idx != max_true:
        if pred_class != ground_truth:
            acc_count -= 1

        # record_results(results.OUTFILE, (os.path.basename(filename), gt_class, pred_class, confidences))
        record_results(results.OUTFILE, (os.path.basename(filename), ground_truth, pred_class, confidences))

    print("{} of {} images correctly classified.\nAccuracy: {:.2f}\n".format(
        str(acc_count),
        str(total),
        acc_count/total * 100.))

    # with open(os.path.join(PRED_DIR, now()+"_results.txt"), 'w') as f:
    #     with open(os.path.join(PRED_DIR, now()+"_results_errors.txt"), 'w') as e:
    #         for filename, pred, ground_truth in zip(filenames, preds, y):
    #
    #             # find class of prediction via max
    #             max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
    #             max_true, val_true = max(
    #                 enumerate(ground_truth), key=itemgetter(1))
    #             pos = class_encodings[max_idx]
    #
    #             # record confidences
    #             confidences = ", ".join(["{:>5.2f}".format(x*100) for x in pred])
    #
    #             if max_idx == max_true:
    #                 f.write("CORRECT for {:<10} with {:<50}".format(pos, filename))
    #             else:
    #                 f.write("INCRRCT guess with {:<10} {:<50}".format(
    #                     pos, filename))
    #                 e.write("{:<10}\t{:<50}".format(pos, filename))
    #                 e.write("Confidences: {}\n".format(confidences))
    #
    #             f.write("Confidences: {}\n".format(confidences))
    #         f.write("{} of {} images correctly classified.\nAccuracy: {:.2f}\n".format(
    #             str(acc_count),
    #             str(total),
    #             acc_count/total * 100.))

    # prevent small crash from TensorFlow/Keras session close bug
    K.clear_session()
