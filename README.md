# Phi-Net: 3D Convolutional Neural Network Implemented with Keras
## Background
Adapted from github.com/sremedios/phinet by Sneha Lingam for LVO project 
Classifying modalities in brain imaging with deep learning

## Directions:
### Directory Setup
Create data directories and subdirectories as below. Training will be 
executed over the data in the train directory and validated over data in 
the validation directory.

The test directory is for images we don't know the labels for and want to
classify, for example for use in a pipeline or for images fresh from the
scanner.

```
./phinet/
+-- data/
|   +-- train/
|   |   +-- /my_class_1
|   |   |   +-- my_class_1_file_1.nii.gz
|   |   |   +-- my_class_1_file_2.nii.gz
|   |   |   +-- my_class_1_file_3.nii.gz
|   |   +-- /my_class_2
|   |   |   +-- my_class_2_file_1.nii.gz
|   |   |   +-- my_class_2_file_2.nii.gz
|   |   |   +-- my_class_2_file_3.nii.gz
|   |   +-- /my_class_3
|   |   |   +-- my_class_3_file_1.nii.gz
|   |   |   +-- my_class_3_file_2.nii.gz
|   |   |   +-- my_class_3_file_3.nii.gz
|   |   +-- /my_class_4
|   |   |   +-- my_class_4_file_1.nii.gz
|   |   |   +-- my_class_4_file_2.nii.gz
|   |   |   +-- my_class_4_file_3.nii.gz
|   +-- validation/
|   |   +-- /my_class_1
|   |   |   +-- my_class_1_file_1.nii.gz
|   |   |   +-- my_class_1_file_2.nii.gz
|   |   |   +-- my_class_1_file_3.nii.gz
|   |   +-- /my_class_2
|   |   |   +-- my_class_2_file_1.nii.gz
|   |   |   +-- my_class_2_file_2.nii.gz
|   |   |   +-- my_class_2_file_3.nii.gz
|   |   +-- /my_class_3
|   |   |   +-- my_class_3_file_1.nii.gz
|   |   |   +-- my_class_3_file_2.nii.gz
|   |   |   +-- my_class_3_file_3.nii.gz
|   |   +-- /my_class_4
|   |   |   +-- my_class_4_file_1.nii.gz
|   |   |   +-- my_class_4_file_2.nii.gz
|   |   |   +-- my_class_4_file_3.nii.gz
|   +-- test/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
|   |   +-- file_n.nii.gz
```
### Training

Run `train.py` to train a classification model with some desired arguments.

`--traindir`: Path to where the preprocessed training data is
`--valdir`: Path to where the preprocessed validation data is

`--classes`: comma-separated list of classes, case-sensitive. This corresponds to the directory names in your provided `--datadir` directory

`--outdir`: Path to location where the weights and model will be saved

`--numcores`: Number of cores to use in parallel preprocessing:
- 1 refers to 1 core
- 2 refers to 2 cores
- -1 refers to all cores
- -2 refers to all but one core

Example usage:
`python train.py --traindir data/train/ --valdir data/val/ --classes 0,1 --outdir models/mmddyy01/ --numcores -1` 

### Classify

Run `predict.py` to classify a single image with some desired arguments:

`--infile`: path to the file to to classify

`--classes`: comma-separated list of classes, case-sensitive. This corresponds to the directory names in your provided `--datadir` directory when the model was trained

`--weights`: path to the trained model weights (.hdf5) to use

`--model`: path to the neural network model architecture (.JSON) to use

`--results_dst`: path and filename (.csv) where results are written

`--preprocesseddir`: output directory where final preprocessed image will be placed

Example usage:
`python predict.py --infile data/test/my_brain.nii.gz --model phinet_my_class_1-my_class_2.json --weights weights/modality/my_weights.hdf5 --results_dst myresults.csv --preprocesseddir data/test/preprocess --classes my_class_1,my_class_2` 

### Validate

Run `validate.py` to validate the model on some holdout data for which the ground truth is known and record metrics with some desired arguments:

`--datadir`: Path to where the preprocessed data is

`--classes`: comma-separated list of classes, case-sensitive. This corresponds to the directory names in your provided `--datadir` directory when the model was trained

`--model`: path to the neural network model architecture (.JSON) to use

`--weights`: path to the trained model weights (.hdf5) to use

`--results_dst`: path to directory where results are written

`--numcores`: Number of cores to use in parallel preprocessing:
- 1 refers to 1 core
- 2 refers to 2 cores
- -1 refers to all cores
- -2 refers to all but one core

Example usage:
`python validate.py --task modality --datadir data/validation/ --model phinet_my_class_1-my_class_2.json --weights weights/modality/my_weights.hdf5 --results_dst validation_results/ --numcores -1 --classes my_class_1,my_class_2`

### Image Preprocessing
Preprocessing is assumed to be performed beforehand. Steps include:

1) Use fsl's `robustfov` to remove the necks
2) Use fsl's `flirt` to register to a common template
3) Apply window level and width settings to look at soft tissue or contrast as appropriate
4) Linearly scale to [0,1] to ease convergence
- Optional: Generate maximum intensity projection (MIP)
- Optional: Strip skull following steps from https://github.com/muschellij2/CT_BET/blob/master/Skull_Strip_Paper/CT_Skull_Strip_Example.sh

### References
The associated paper is available on ResearchGate: `https://www.researchgate.net/publication/323440662_Classifying_magnetic_resonance_image_modalities_with_convolutional_neural_networks`

If this is used, please cite:
Samuel Remedios, Dzung L. Pham, John A. Butman, Snehashis Roy, "Classifying magnetic resonance image modalities with convolutional neural networks," Proc. SPIE 10575, Medical Imaging 2018: Computer-Aided Diagnosis, 105752I (27 February 2018); doi: 10.1117/12.2293943
