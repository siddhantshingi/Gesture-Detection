import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import pickle
import os
import cv2 

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(4 * 12 * 12, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = pickle.load(open("./models/q1.model", 'rb'))

class_name = ["right","left", "stop", "background"]
cap = cv2.VideoCapture(0) 
counter = 0
while(True): 
    ret, frame = cap.read()
    if (not ret):
        break
    cv2.imshow("camera frame",frame)
    if (counter % 50 == 0):
        out_frame = frame

        frame = cv2.resize(frame,(50,50))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = frame*1.0/255.0
        frame = frame.astype('float32')

        test_x = [frame]
        test_x = np.array(test_x)
        test_x = test_x.reshape(1, 1, 50, 50)
        test_x = torch.from_numpy(test_x)

        with torch.no_grad():
            output = model(test_x)

        softmax = torch.exp(output)
        prob = list(softmax.numpy())
        predictions_test = np.argmax(prob, axis=1)
        print (predictions_test)
        print (class_name[predictions_test[0]])
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50) 
        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
        frame = cv2.putText(out_frame, class_name[predictions_test[0]], org, font,  
                           fontScale, color, thickness, cv2.LINE_AA) 
        cv2.imshow("prediction frame",out_frame)

    counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows() 