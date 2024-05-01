# ML/DL-Project

This is a Github Repository that contains the files for a machine learning project based on AI Biomedical Engineering. The main goal of this project is to create and train a Deep Learning AI Inference Model on calculating complex cardiovascular measurements and diagnosis of cardiomyopathy. 

The model is based on the Echonet-Dynamic algorithm and for the training a compination of MobilenetV2/MobilenetV3 Neural Network architectures and DeepLabV3/DeepLabV3+ encoders is used to achieve high accuracy results.

The [model.py](https://github.com/George-Sakellariou/Machine-Learning-Project/blob/main/model.py) contains the code and comments for the preprocessing of the input data of ECG videos, the training of the model accordingly to the architecture and encoder that is being used and the extraction of the result in CSV document file. For the purpose of this experimental demonstration the files included on the model.py exist, but in order to simplify the explanation of the model they are not being analyzed further.
As an example [video.py](https://github.com/George-Sakellariou/Machine-Learning-Project/blob/main/video.py) is showing the initialization of the DL segmentation model that is referred in model.py.

The [training.sh](https://github.com/George-Sakellariou/Machine-Learning-Project/blob/main/training.sh) file contains the shell script that is used to train the model in various neural network architectures in different size batches to maximize speed efficiency.

Finally [visual.R](https://github.com/George-Sakellariou/Machine-Learning-Project/blob/main/visual.R) file contains the R script that is used to plot the results of the previous batch training of the model. 
[bbxV@DP.jpeg](https://github.com/George-Sakellariou/Machine-Learning-Project/blob/main/bbxV%40DP.jpeg) is an example of the MAE in different batch training with MobileNetV2 architecture of neural network and DeepLabV3 as encoder.
