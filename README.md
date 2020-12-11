# Object-Detector
It detects the target objects in a given image and then specifies them in a bounding box. It uses Faster RCNN at it's backend which is forked from ruotianluo's pytorch-faster-rcnn repository.

Here a custom dataset(of PascalVOC2007 format) was used for training instead of a standard dataset. It allows for the flexibility to use a dataset with our own images and classes which may be different than a standard dataset. The code written uses images and .mat labels file(created after image annotations) from MATLAB to create the custom dataset. More than one image folders can also be used at once for conversion to dataset. The code for conversion also takes care of corner cases like -ve annotation values, non-jpeg images(problem while creating xml files) and missing class in a particular folder of images. Complete procedure and requirements have been specified below.

Several changes have been made to pytorch-faster-rcnn repository to make it work for this project,

1. Added VOCdevkit2007 under ./data. This provides a basic structure for thustom dataset.(Has to be added explicitly as explained in prerequisites)

2. Changes made in pascal_voc.py located under ./lib/datasets. The changes made were in accordance with the classes that were used for training and some database parameters.

3. Changes made in demo_all_bboxes.py located under ./tools. Here inference related changes were made according to the classes in dataset. Some extra code was also added so that when inference is taken on images, 'output_info.csv' is exported. This file consists of columns like, target, image_name, class, bbox_x, bbox_y, bbox_w, bbox_h and conf_score.

4. Few changes were also made to files in 'model' and 'nets' directories located under ./lib. This helped to improve the final results.

5. Several files were also added to the root directory which use the VOCdevkit2007 as a structure and create a custom dataset using the images and some other annotation files specified below.

## Results
This model was trained for 1,50,000 iterations on self created custom dataset(~30k images) with variable learning rate and meanAP acheived was 0.72 which is comparable to 0.76 in original paper of Faster RCNN considering former was trained on custom dataset.

Inference Results(with conf. score > 0.8) : https://drive.google.com/open?id=1Yif4OZWjfq1Z4fp3ynQVcnC3PRRjxJ31

## Prerequisites(according to changes made)-
(‘.’ in the path of a file specifies the root directory of faster rcnn)

1. Make sure that you delete the cache(located under ./data/cache) files everytime the dataset is altered(whether images are added, removed or other changes), otherwise the model will throw an error because it will be reading from the old ‘.pkl’ files(in cache folder) and the original data has been changed.

2. If you are using the model for inference only then place the image files (on which inference is to be given) under ./data/demo.

3. Populate the ./data/imagenet_weights with the .pth files of the backbone CNN networks to be used for training process.

4. Place the image folders and their csv files(obtained from exporting the labels and then writing gTruth of that file, in MATLAB) in dataset folder, located in the root directory of the Faster RCNN. Remember that the names of the image folders and their csv files must be the same and start with ‘img’ prefix, followed by a number or a string. Use the following MATLAB code to convert the .mat annotations to csv files,

	1. load(‘path/to/example.mat’)
	2. gTruth
	3. labels=objectDetectorTrainingData(gTruth)
	4. writetable(labels, ‘path/to/save/example.csv’, ‘Delimiter’, ‘ ’)

5. Make sure that name of the object classes are in line with your dataset(A general dataset may use many number of classes, like pascal voc uses 20 classes). Check for the names of classes in pascal_voc.py(located under ./lib/datasets) and demo_all_bboxes.py(located under ./tools). In demo_all_bboxes in line 195, also change the number of classes according to you data.

6. If any two or more of the image folders have a combined csv then it can be separated using ‘sep.ipynb’. Before training make sure that all datasets have their own respective csv files(not combined).

7. Create a folder named 'dataset' at root directory.

8. Download PascalVoc devkit from the link provided below. Extract the zip and place the folder under ./data.
   https://drive.google.com/open?id=1uJ_XjN9ayooCJF7iifrIAfSnzAmXrJbP

9. Place the image folders and their respective csv files(obtained from exporting the labels and then writing gTruth of that file, in MATLAB) in dataset folder, located in the root directory. Remember that the names of the image folders and their respective csv files must be the same and start with ‘img’ prefix, followed by a number or a string.

## Training-
### 1. Run the XMLconvert.ipynb.
1. Input – Image folders and csv files in dataset folder.
2. Output – 
	1. XML files in Annotations folder(located under ./data/VOCdevkit2007/VOC2007).  
	2. Images in JPEGImages folder(located under ./data/VOCdevkit2007/VOC2007).
	3. ‘exp.csv’, located in root directory.

### 2. Run create.ipynb.
1. Input – ‘exp.csv’, generated by XMLconvert.ipynb
2. Output – 
	1. trainval.txt, train.txt, val.txt and test.txt in Layout folder.
	2. Object class specific files of the above listed files, in Main folder. Like person_train.txt, person_val.txt, 	person_trainval.txt and person_test.txt

OR

Run create_random.ipynb to select random images for test and train. Works same as create.ipynb just selects images randomly.

After the 2nd step the custom dataset is ready and is present at .data/VOCdevkit2007. It is basically a format of PascalVOC2007 but containing self specified classes and images.

### 3. Adjust the paramaters 
Adjust the parameters according to yourself in ‘train_faster_rcnn.sh’, located under ./experiments/scripts. Parameters other than these are specified as command line arguments, while giving command for training in terminal.

### 4. Training the model
Open the terminal in root directory of faster rcnn and run the following command,

./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]

GPU_ID – If your PC has multiple GPUs then specify the number of GPU on which the model is to be trained, otherwise set it to 0(for single GPU Pcs).

DATASET – Specify the dataset that you want to use for training. Eg pascal_voc, coco etc.

NET – Specify the name of the backbone CNN model that you want to use for training. Remember to include the .pth file of desired CNN model in ./data/imagenet_weights.

A sample command would look like,
	
	./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101

Parameters like cfg, anchors, ratios and weights are implicitly taken care of. So, no need to worry about that.

## Tensorboard visualisation - 

Tensorboard visualisation can be very useful to check and supervise the process of training in the form of graphs. Tensorboard consists a ton of information related to the training process in form of graphs which are very useful for interpretations related to training. For visualisation using tensorboard, 

Open a terminal window in root directory of faster rcnn and type the following command

	tensorboard --logdir=tensorboard/res101/voc_2007_trainval/ --host localhost --port=7001 &

(Please activate the environment containing tensorboard installation before the above command)


This command will provide a browser link inside the terminal, copy the link and paste it in the browser to open the tensorboard window.

Tip – If the above command gives an error like, 
	‘ERROR: TensorBoard could not bind to port 7001, it was already in use’ 
then try a different port number.


## After training -

1. Trained networks(including snapshots) are saved under, 

	output/[NET]/[DATASET]_trainval/default/

2. Test outputs are saved under,

	output/[NET]/[DATASET]_test/default/

3. Tensorboard information for training and validation is saved under, 

	tensorboard/[NET]/[DATASET]/default/
	tensorboard/[NET]/[DATASET]/default_val/

## Testing and evaluation - 

Although, testing and evaluation are implicitly performed after the training process but if you want to perform testing explicitly, it can be done by using the following command.

	./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]

where, GPU_ID, DATASET and NET have same meanings as before,

GPU_ID – If your PC has multiple GPUs then specify the number of GPU on which the model is to be trained, otherwise set it to 0(for single GPU Pcs).

DATASET – Specify the dataset that you want to use for training. Eg pascal_voc, coco etc.

NET – Specify the name of the backbone CNN model that you want to use for training. Remember to include the .pth file of desired CNN model in ./data/imagenet_weights.

A sample command would look like, 
	
	./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc res101

## Inference(Demo) - 

Inference is the process of taking output(detection in this case) on images using a trained model. Basically, classifying data to ‘infer’ a result.

	GPU_ID=0
	CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo_all_bboxes.py

For systems having a single GPU, GPU_ID can be ignored as it will always be zero.

The demo_all_bboxes.py outputs a csv file named, ‘output_info.csv’. This file consists of columns like, target, image_name, class, bbox_x, bbox_y, bbox_w, bbox_h and conf_score. The inference is given only on the images placed in ‘demo’ folder, located under ./data. And the results inferred on all these images are stored in the output_info.csv.

## Other useful notebook - 

create_lists.ipynb - 
Input – ‘output_info.csv’, generated by demo_all_bboxes.py
Output – List of lists containing info related to each image on which inference is done.

This notebook is used to create python lists using the ‘output_info.csv’. These lists can easily be further used in other programs for processing. At the end of this notebook a list of lists is created containing a separate list for each class in a given image in demo folder. Each list contains values specifying various parameters related to the images on which inference is done. These parameters in order are, target, image_name, class, bbox_x, bbox_y, bbox_w, bbox_h and conf_score.
