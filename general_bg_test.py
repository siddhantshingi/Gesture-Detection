import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import os
import cv2
torch.manual_seed(0)
np.random.seed(0)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 12, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(12),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(12),
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
            Linear(12 * 12 * 12, 4)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model1=torch.load('./models/q2newgpu.model', map_location='cpu')
model1.eval()
count=0
totacc=0
correct=0


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

        frame = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
        frame = frame*1.0/255.0
        frame = frame.astype('float32')

        test_x = [frame]
        test_x = np.array(test_x)
        test_x = test_x.reshape(1, 3, 50, 50)
        test_x = torch.from_numpy(test_x)

        with torch.no_grad():
            output = model1(test_x)

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