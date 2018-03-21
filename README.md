# Phi-Net: 3D Convolutional Neural Network Implemented with Keras
## Background
Classifying modalities in magnetic resonance brain imaging with deep learning.

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
|   |   +-- /class_1
|   |   |   +-- t_file_1_1.nii.gz
|   |   |   +-- t_file_1_2.nii.gz
|   |   |   +-- t_file_1_3.nii.gz
|   |   +-- /class_2
|   |   |   +-- t_file_2_1.nii.gz
|   |   |   +-- t_file_2_2.nii.gz
|   |   |   +-- t_file_2_3.nii.gz
|   |   +-- [...]
|   |   +-- /class_n
|   |   |   +-- t_file_n_1.nii.gz
|   |   |   +-- t_file_n_2.nii.gz
|   |   |   +-- t_file_n_3.nii.gz
|   +-- validation/
|   |   +-- /class_1
|   |   |   +-- v_file_1_1.nii.gz
|   |   |   +-- v_file_1_2.nii.gz
|   |   |   +-- v_file_1_3.nii.gz
|   |   +-- /class_2
|   |   |   +-- v_file_2_1.nii.gz
|   |   |   +-- v_file_2_2.nii.gz
|   |   |   +-- v_file_2_3.nii.gz
|   |   +-- [...]
|   |   +-- /class_n
|   |   |   +-- v_file_n_1.nii.gz
|   |   |   +-- v_file_n_2.nii.gz
|   |   |   +-- v_file_n_3.nii.gz
|   +-- test/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
```
### Image Preprocessing
First all images are rotated into RAI orientation using AFNI `3dresample`. This ensures
that our 45x45x5 patches capture mostly axial information.

Then each of these images will be run under fsl's `robustfov` during the loading of 
images, and saved into a temporary "robustfov/" directory.  This directory will be 
destroyed at the end of training, validation, or testing.

### Training
Run `train.py`, ensuring that the files are in the correct directories illustrated above.

### Classify
Run `validate.py` to get an accuracy score over data for which the labels are known.
This runs the latest model over the holdout set.

### Example with Sorting 
The file `sort.py` demonstrates an example, using the weights to sort into the appropriate
directories the assorted files of classes on which the model was trained.

Place some unlabeled images in the `data/unsorted` directory, then run `sort.py` after training
your model.

### Results from downsampled data (SPIE conference paper)
{TODO}
accuracy, training time, testing time

### Improved, Current Results from 3D patches
{TODO}
accuracy, training time, testing time

### References
The associated paper is available on ResearchGate: https://www.researchgate.net/publication/323440662_Classifying_magnetic_resonance_image_modalities_with_convolutional_neural_networks

If this is used, please cite our work:
Samuel Remedios, Dzung L. Pham, John A. Butman, Snehashis Roy, "Classifying magnetic resonance image modalities with convolutional neural networks," Proc. SPIE 10575, Medical Imaging 2018: Computer-Aided Diagnosis, 105752I (27 February 2018); doi: 10.1117/12.2293943
