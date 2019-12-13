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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import os
import cv2
torch.manual_seed(0)
np.random.seed(0)

n_train = 2430
train_data_x = []
train_data_y = []
val_data_x = []
val_data_y = []
class_names = ["right", "left", "stop", "background"]
class_code = 0
for class_name in tqdm(class_names):
    image_names = os.listdir('./data/general_bg/' + class_name)
    image_paths = ["./data/general_bg/" + class_name + "/" + image_name for image_name in image_names]
    n_img = 0
    for image_path in image_paths:
        img = cv2.imread(image_path)
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

train_x = train_x.reshape(9720, 3, 50, 50)
val_x = val_x.reshape(1080, 3, 50, 50)
train_y = train_y.astype(int)
val_y = val_y.astype(int)
train_x = torch.tensor(train_x)
val_x = torch.tensor(val_x)
train_y = torch.tensor(train_y)
val_y = torch.tensor(val_y)

batch_size = 100

train_data = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

val_data = TensorDataset(val_x, val_y)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

class PrintLayer(Module): 
    comment = ""
    def __init__(self,c):
        self.comment = c
        super(PrintLayer, self).__init__() 
    def forward(self, x): 
        print(self.comment,x.shape) 
        return x

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
        )

        self.linear_layers = Sequential(
            Linear(12 * 12 * 12, 4)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Net()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)

def train(epoch):
    model.train()
    tr_loss = 0
    val_loss = 0
    tr_correct = 0
    val_correct = 0
    for i, (inputs, labels) in enumerate(train_dataloader):

      x_train, y_train = Variable(inputs),Variable(labels)

      if torch.cuda.is_available():
          x_train = x_train.cuda()
          y_train = y_train.cuda()

      optimizer.zero_grad()

      output_train = model(x_train)

      softmax = torch.exp(output_train)

      prob = list(softmax.cpu().detach().numpy())
      predictions_train = np.argmax(prob, axis=1)
      tr_correct += (predictions_train == y_train.cpu().numpy()).sum()
      loss_train = criterion(output_train, y_train)
      tr_loss+=loss_train.item()

      loss_train.backward()
      optimizer.step()

    train_losses.append(tr_loss)
    train_acc.append(tr_correct/3240)
    model.eval()
    for i, (inputs, labels) in enumerate(val_dataloader):
      x_val, y_val = Variable(inputs),Variable(labels)

      if torch.cuda.is_available():
          x_val = x_val.cuda()
          y_val = y_val.cuda()

      with torch.no_grad():
        output_val = model(x_val)
      softmax = torch.exp(output_val)
      
      prob = list(softmax.cpu().numpy())
      predictions_val = np.argmax(prob, axis=1)
      val_correct += (predictions_val == y_val.cpu().numpy()).sum()
      loss_val = criterion(output_val, y_val)
      val_loss+=loss_val.item()

    val_losses.append(val_loss)
    val_acc.append(val_correct/360)
    print('Epoch : ',epoch+1, '\t', 'train loss :', tr_loss, 'validation loss :', val_loss)
    print('Epoch : ',epoch+1, '\t', 'train accuracy :', tr_correct/9720, 'val accuracy :', val_correct/1080)

n_epochs = 20

train_losses = []
val_losses = []
train_acc=[]
val_acc=[]
for epoch in range(n_epochs):
    train(epoch)

torch.save(model, "./models/q2newgpu.model")

# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

# plotting the training and validation loss
plt.plot(train_acc, label='Training accuracy')
plt.plot(val_acc, label='Validation accuracy')
plt.legend()
plt.show()

with torch.no_grad():
    output = model1(train_x.cuda())
    
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