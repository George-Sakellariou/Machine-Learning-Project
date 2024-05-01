import os
import sys
import cv2
import csv
import numpy as np
import scipy.signal
import torch
import onnx
import onnxruntime as rt
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn.metrics
from tqdm import tqdm

mean = [33.741943, 33.877575, 34.1646] #Mean from echonet paper
std = [51.184673, 51.356464, 51.660316] #Standard deviation from echonet paper
torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)
bigger_than_64 = True

#Function for loading the video
def load_video(input_filepath):
    #Get video frames, width and height to initialize the array that stores video frames
    vid = cv2.VideoCapture(input_filepath)
    video_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = np.zeros((video_frames, video_height, video_width, 3), np.uint8)

    #Read each video frame and convert it from the color space of BGR to RGB
    for count in range(video_frames):
        ret, frame = vid.read()
        if not ret:
            break
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video[count, :, :] = frame
    
    #If the video is not 112x112 pixels width x height resize it accordingly.
    #The resamplig is done using pixel area relation (cv2.INTER_AREA) scaling the image at 112x112 pixels.
    if (video_height != 112) or (video_width != 112):
        resized_video = [cv2.resize(video[i, :, :], (112,112), interpolation = cv2.INTER_AREA) for i in range(video_frames)]
        final_video = torch.from_numpy(np.asarray(resized_video)).float()
    else:
        final_video = video
    
    return final_video, video_frames #final_video stores the input video

'''Function for the computation of the Ejection Fraction
    As arguments we have:
    i) lvef_model_path : The onnx inference model for ejection fraction computation
    ii) input_filepath : The path to the input video fileList
    iii) segmentation_model_path : The onnx inference model for the segmentation module from the echonet'''
def ef_module(lvef_model_path, input_filepath, segmentation_model_path):

    final_video, video_frames = load_video(input_filepath)
    normalized_video = (final_video - mean) / std #Normalize the video with reference to mean and std from the echonet.

    ort_segm_session = rt.InferenceSession(segmentation_model_path) #Load the onnx segmentation model
    video_array = (len(final_video), 112, 112) #Make an array with dimensions equal to the input video
    segmented_video = np.zeros(np.array(np.array(video_array))) #Segmented video will store the segmented frames from the output of the segmentation model

    '''The shape of the input video is (video_frames, 112, 112, 3).
        First we convert the shape to (1, 3, video_frames, 112, 112) to match the input of the segmentation model
        and after the convertion of each frame we store the output to the "segmented_video" variable.'''
    for i in range(len(normalized_video)):
        
        ort_input = {ort_segm_session.get_inputs()[0].name: (np.expand_dims(np.transpose(normalized_video, (3,0,1,2)), axis = 0))[:,:,i,:,:].astype(np.float32)} #Get input for the segmentation model
        segmentation = ort_segm_session.run(None, ort_input)[0][0][0] #Run segmentation model
        segmented_video[i] = segmentation #Store output

    #print(segmented_video.shape)

    #Check for video with less than 64 frames that is not suitable for computing EF
    bigger_than_64 = True
    if video_frames < 64:
        print(input_filepath, 'Video too small to calculate Ejection Fraction', video_frames)
        bigger_than_64 = False
        return 60

    #Find the frames that contain the peaks of each beat cycle of the segmented video
    size_table = np.zeros(shape=(video_frames, 3)) #Initialization of size_table that will store each frame and if it is a peak or not
    logit = np.asarray(segmented_video)
    size = (logit>0).sum((1,2)) #size stores the sum of the pixels' frames that are inside the area of the left ventricle in each segmented frame (not black = 0) 
    
    #Compute the range of the acceptable values of the size of the peaks to avoid noisy signal peaks
    trim_min = sorted(size)[round(len(size) ** 0.5)] #trim_min stores the lower bountary value of the size of the peak
    trim_max = sorted(size)[round(len(size) ** 0.95)] #trim_max stores the upper bountary value of the size of the peak
    trim_range = trim_max - trim_min #The acceptable peak range
    print(trim_min, trim_max, trim_range)
    systole = set(scipy.signal.find_peaks(-size, distance=30, prominence=(0.50 * trim_range))[0]) #Find the local maxima in at least 30 samples distance with a promince of half the peak range

    for (frame, s) in enumerate(size):
        size_table[frame] =[frame, s, 1 if frame in systole else 1] #Store the information of a frame containing a peak(1) or not(0)
        #if frame in systole:
            #print(frame)

    predictions = []
    ort_echo_session = rt.InferenceSession(lvef_model_path) #Load the onnx estimate ejection fraction model

    '''The shape of the input normalized_video is (video_frames, 112, 112, 3).
        First we convert the shape to (1, 3, 32-frames, 112, 112) to match the input of the segmentation model
        and after that we give as input a clip consisting of 32 frames each after each frame of every 64 frames.'''
    for i in range(len(normalized_video) - 64):

        ort_input = {ort_echo_session.get_inputs()[0].name: (np.expand_dims(np.transpose(normalized_video, (3,0,1,2)), axis = 0))[:, :, i:i+64:2, :, :].astype(np.float32)} #Get input for the EF model
        lvef = ort_echo_session.run(None, ort_input)[0][0][0] #Run EF model
        predictions.append(lvef) #store output prediction

    #Calculations for the computation of the BeatToBeat ejection fraction mean and standard deviation values
    #Constract the dataframe numData that contains the predictions of each frame in numbered position
    numbers = np.arange(len(predictions))
    num = pd.DataFrame(numbers, columns = ['specific']).astype(float)
    data = pd.DataFrame(predictions, columns = ['EF'])
    numData = pd.concat([num, data], axis = 1)

    #Constract the dataframe sizeRelevantFrames that is based on the size_table from above with the positions of the frames of the peaks
    sizeData = pd.DataFrame(size_table, columns = ['Frame', 'Size', 'Peak'])
    sizeData = sizeData.loc[sizeData['Peak'] == 1] #Only the values that have Peak = 1 are valid
    sizeRelevantFrames = sizeData[['Frame']].copy()
    sizeRelevantFrames['Frame'] = sizeRelevantFrames['Frame'] - 32 #Then the subtraction of 32 is done to determine the position of the beginning frame of the clip that contains the beat(beginning of the systolic phase)
    num = sizeRelevantFrames._get_numeric_data()
    num[num < 0] = 0 #Negative values of position frames are turned into frame 0(First frame if the peak is in the first clip)

    #Merging the above dataframes matches the frames of the peaks with its corresponding ejection fraction value
    beatByBeat = pd.merge(sizeRelevantFrames, numData, left_on=['Frame'], right_on=['specific'])
    final_beatByBeat_mean = np.asarray(beatByBeat['EF'].mean()) #Compute mean of ejection fraction
   
    return final_beatByBeat_mean

dataset_path = '/home/georgios/dynamic/a4c-video-dir/'
lvef_model_path = '/home/georgios/dynamic/a4c-video-dir/infer_data/EchoNet_pretrained.onnx'
segmentation_model_path = '/home/georgios/dynamic/a4c-video-dir/infer_data/EchoNet_segmentation.onnx'

fileList = pd.read_csv(dataset_path+'FileList.csv') #Read csv file of the videos
fileList_test = fileList[fileList['Split'] == 'TEST'] #Keep only testing dataset
num_videos = len(fileList_test) #Number of testing videos

lvef_preds = []

# Run ejection fraction model and construct csv file that contains the Name, Mean ejection fraction value and STD of the video
fnames = []
header = fileList.columns.tolist()
fnames = fileList["FileName"].tolist()

header = ['Filename', 'Mean']
with open('final.csv', 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(header)

for j in tqdm(range(num_videos)):

    if(bigger_than_64):

        filePath = dataset_path + 'Videos/' + fileList_test['FileName'].iloc[j]
        ef_mean = ef_module(lvef_model_path, filePath, segmentation_model_path)
        lvef_preds.append(ef_mean)
        with open(os.path.join("final.csv"), "a", newline = "") as f:
            f.write("{},{}\n".format(fileList_test['FileName'].iloc[j], ef_mean))
