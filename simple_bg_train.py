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
torch.manual_seed(0)
np.random.seed(0)

n_train = 1080
train_data_x = []
train_data_y = []
val_data_x = []
val_data_y = []
class_names = ["right", "left", "stop", "background"]
class_code = 0
for class_name in tqdm(class_names):
    image_names = os.listdir('./data/simple_bg/' + class_name)
    image_paths = ["./data/simple_bg/" + class_name + "/" + image_name for image_name in image_names]
    n_img = 0
    for image_path in image_paths:
        img = imread(image_path,as_gray = True)
        img = img*1.0/255.0
        img = img.astype('float32')
        if (n_img < n_train):
            train_data_x.append(img)
            train_data_y.append(class_code)
        else:
            val_data_x.append(img)
            val_data_y.append(class_code)
        n_img += 1
    print (class_name,len(image_paths),len(train_data_x))
    class_code += 1
train_x = np.array(train_data_x)
train_y = np.array(train_data_y)
val_x = np.array(val_data_x)
val_y = np.array(val_data_y)
print ("train_x: ",train_x.shape)
print ("train_y: ",train_y.shape)
print ("val_x: ",val_x.shape)
print ("val_y: ",val_y.shape)


i = 0
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(train_x[i], cmap='gray')
plt.subplot(222), plt.imshow(train_x[i+1800], cmap='gray')
plt.subplot(223), plt.imshow(train_x[i+2800], cmap='gray')
plt.subplot(224), plt.imshow(train_x[-1], cmap='gray')

train_x = train_x.reshape(4320, 1, 50, 50)
train_x  = torch.from_numpy(train_x)

train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

train_x.shape, train_y.shape

val_x = val_x.reshape(480, 1, 50, 50)
val_x  = torch.from_numpy(val_x)

val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)

val_x.shape, val_y.shape

class PrintLayer(Module): 
    comment = ""
    def __init__(self,c):
        self.comment = c
        super(PrintLayer, self).__init__() 
    def forward(self, x): 
        # Do your print / debug stuff here 
        print(self.comment,x.shape) 
        return x

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
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
            Linear(4 * 12 * 12, 4)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Net()
optimizer = Adam(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
def train(epoch):
    model.train()
    tr_loss = 0

    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    optimizer.zero_grad()
    
    output_train = model(x_train)
    output_val = model(x_val)

    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

n_epochs = 100

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    train(epoch)

model1 = pickle.load(open("./models/q1.model", 'rb'))

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

with torch.no_grad():
    output = model1(train_x)
    
softmax = torch.exp(output)
prob = list(softmax.numpy())
predictions_train = np.argmax(prob, axis=1)

print(accuracy_score(train_y, predictions_train))

with torch.no_grad():
    output = model1(val_x)

softmax = torch.exp(output)
prob = list(softmax.numpy())
predictions_val = np.argmax(prob, axis=1)

print(accuracy_score(val_y, predictions_val))