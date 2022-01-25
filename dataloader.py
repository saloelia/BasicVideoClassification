import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_of_frames = 100
labels_dict = {'Basketball':0,'Biking':1,'Diving':2,'PizzaTossing':3,'RopeClimbing':4}

class VideoDataset(Dataset):
    def __init__(self,videos,transforms,train=True):
        self.videos = videos
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
       sub_video_names = self.videos[index]
       all_frames = []
       label = sub_video_names[0].split("_")[1]
       path_to_folder = os.path.join('/home/linuxgpu/Downloads/VideoClassification/UCF-101-miniset/',label)
       for sub_video in sub_video_names:
           path_to_file = os.path.join(path_to_folder,sub_video) 
           vid_capture  = cv2.VideoCapture(path_to_file)
           ret, frame = vid_capture.read()
           while ret:           
               all_frames.append(frame)
               ret, frame = vid_capture.read()
               

       chosen_indexes = random.sample(range(0, len(all_frames)), num_of_frames)
       frames = []
       for i,frame in enumerate(all_frames):
          if i in chosen_indexes:
            PIL_image = Image.fromarray(np.uint8(frame)).convert('RGB')
            tensor_img = self.transforms(PIL_image)
            frames.append(tensor_img)   

       frames = torch.stack(frames)
       label = labels_dict.get(label)
       return frames,label


def get_videos_and_labels(path_to_txt="/home/linuxgpu/Downloads/VideoClassification/data_split/trainlist01.txt"):
    f = open(path_to_txt, "r")
    temp = f.read()
    frames_in_file = temp.split('\n')
    frames_in_file = frames_in_file[:-1]
    all_videos = []
    all_labels = []
    str_video_before = 'g01'
    frames = []
    for frame in frames_in_file:
        frame = frame.split(" ")[0]
        frame = frame.split("/")[1]
        str_video_now = frame.split("_")[2]
        if str_video_before==str_video_now:
            frames.append(frame)
        else:
           str_video_before=str_video_now
           all_videos.append(frames)
           label = frame.split("_")[1]
           all_labels.append(label)
           frames = []
           frames.append(frame)
    
    return all_videos,all_labels

