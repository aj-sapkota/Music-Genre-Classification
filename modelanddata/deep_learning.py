#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import torchvision
from torchsummary import summary


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import json


# In[7]:


#for gpu model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[8]:


torch.cuda.is_available()


# In[4]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:





# In[9]:


#loading data from json file 
# path = '/content/drive/MyDrive/Colab Notebooks/data_test.json'
path = 'dataset/data_10.json'

with open(path,"r") as jsonLoad:
    mainData = json.load(jsonLoad)
    labels = mainData['labels']
    data = mainData['mfcc']
    mappings = mainData['mapping']
    


# In[10]:


#inspection of data

# print(labels)
print(len(data))

dataT = torch.tensor(data).float()
labelsT = torch.tensor(labels).long()

#transform to 4d tensors 
mfccCoeff = dataT.view([5992,1,216,13]).float()
print(mfccCoeff.shape)
print(type(mfccCoeff))


# In[11]:


# print(dataT.shape)
print(labelsT.shape)

torch.unique(dataT)


# In[12]:


#normalization of data
plt.hist(mfccCoeff[:10,:,:,:].view(1,-1).detach(),40)
plt.title('Raw values')
plt.show()

mfccCoeff /= torch.max(mfccCoeff)
plt.hist(mfccCoeff[:10,:,:,:].view(1,-1).detach(),40)
plt.title('Raw values')
plt.show()


# In[13]:


train_data,test_data,train_labels,test_labels = train_test_split(mfccCoeff,labelsT,test_size=.1)


train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

batchsize = 32
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])


# In[14]:


#check size
print(train_loader.dataset.tensors[0].shape)
print(train_loader.dataset.tensors[1].shape)

print('\n')
print(test_loader.dataset.tensors[0].shape)
print(test_loader.dataset.tensors[1].shape)

print(test_data.tensors[0].shape[0])


# In[15]:


def makeTheModel(printtoggle=False):
    class musicNet(nn.Module):
        def __init__(self,printtoggle):
            super().__init__()
            
            #printtoggle
            self.print = printtoggle
            
            #firstconvolution layer
            self.conv1 = nn.Conv2d(1,64,3,padding=1)
            self.bnorm1 = nn.BatchNorm2d(64) 
            
            #second convolution layer
            self.conv2 = nn.Conv2d(64,128,3,padding=1)
            self.bnorm2 = nn.BatchNorm2d(128)
            
            #linear layers
            self.fc1 = nn.Linear(54*3*128,256)
            self.fc2 = nn.Linear(256,64)
            self.fc3 = nn.Linear(64,10)
            
        def forward(self,x):
            
            if self.print: print(f'Input: {list(x.shape)}')
                
            #firstblock 
            x = F.leaky_relu( self.bnorm1( F.max_pool2d(self.conv1(x),2) ) )
            if self.print: print(f'First block : {list(x.shape)}')
            
            #secondblock
            x = F.leaky_relu( self.bnorm2( F.max_pool2d(self.conv2(x),2) ) )
            if self.print: print(f'Second block : {list(x.shape)}')
            
            #reshape for linear layer
            nUnits = x.shape.numel()/x.shape[0]
#             print(x.shape.numel()),print(x.shape[0]),print(nUnits)
            x = x.view(-1,int(nUnits))
            if self.print: print(f'Vectorized : {list(x.shape)}')

            #linear layer
            x = F.leaky_relu(self.fc1(x))
            x = F.dropout(x,p=.5,training=self.training)
            x = F.leaky_relu(self.fc2(x))
            x = F.dropout(x,p=.65,training=self.training)
            x = self.fc3(x)
            if self.print: print(f'Final output : {list(x.shape)}')
                
            return x
    
    #model instance
    net = musicNet(printtoggle)
    
    #lossfun
    lossfun = nn.CrossEntropyLoss()
    
    #optimzer
    optimizer = torch.optim.Adam(net.parameters(),lr=.0001,weight_decay=1e-5)
    
    return net,lossfun,optimizer
                       


# In[16]:


# test the model with one batch
net,lossfun,optimizer = makeTheModel(True)

X,y = iter(train_loader).next()
yHat = net(X)

# check size of output
print('\nOutput size:')
print(yHat.shape)

# # now let's compute the loss
loss = lossfun(yHat,torch.squeeze(y))
print(' ')
print('Loss:')
print(loss)


# In[ ]:





# In[17]:


#train model
def function2trainTheModel():

  # number of epochs
  numepochs = 35
  
  # create a new model
  net,lossfun,optimizer = makeTheModel()

  # send the model to the GPU
  net.to(device)

  # initialize losses
  trainLoss = torch.zeros(numepochs)
  testLoss  = torch.zeros(numepochs)
  trainErr  = torch.zeros(numepochs)
  testErr   = torch.zeros(numepochs)


  # loop over epochs
  for epochi in range(numepochs):

    # loop over training data batches
    net.train()
    batchLoss = []
    batchErr  = []
    for X,y in train_loader:

      # push data to GPU
      X = X.to(device)
      y = y.to(device)

      # forward pass and loss
      yHat = net(X)
      loss = lossfun(yHat,y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # loss and error from this batch
      batchLoss.append(loss.item())
      batchErr.append( torch.mean((torch.argmax(yHat,axis=1) != y).float()).item() )
    # end of batch loop...

    # and get average losses and error rates across the batches
    trainLoss[epochi] = np.mean(batchLoss)
    trainErr[epochi]  = 100*np.mean(batchErr)



    ### test performance
    net.eval()
    X,y = next(iter(test_loader)) # extract X,y from test dataloader

    # push data to GPU
    X = X.to(device)
    y = y.to(device)

    with torch.no_grad(): # deactivates autograd
      yHat = net(X)
      loss = lossfun(yHat,y)
      
    # get loss and error rate from the test batch
    testLoss[epochi] = loss.item()
    testErr[epochi]  = 100*torch.mean((torch.argmax(yHat,axis=1) != y).float()).item()

  # end epochs

  # function output
  return trainLoss,testLoss,trainErr,testErr,net,yHat


# In[19]:


trainLoss,testLoss,trainErr,testErr,net, y_predicted_feature = function2trainTheModel()


# In[20]:


def predictUpload(mfccCoeff):
    mfccCoeff /= torch.max(mfccCoeff)
    yUploadPred = net(mfccCoeff)
    return yUploadPred

    


# In[ ]:





# In[21]:


y_predicted = torch.argmax(y_predicted_feature.cpu(),axis=1)


# In[22]:


fig,ax = plt.subplots(1,2,figsize=(16,5))

ax[0].plot(trainLoss,'s-',label='Train')
ax[0].plot(testLoss,'o-',label='Test')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss ')
ax[0].set_title('Model loss')

ax[1].plot(trainErr,'s-',label='Train')
ax[1].plot(testErr,'o-',label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Error rates (%)')
ax[1].set_title(f'Final model test error rate: {testErr[-1]:.2f}%')
ax[1].legend()

plt.show()


# In[23]:



from sklearn.metrics import classification_report
print(classification_report(test_labels.detach(),y_predicted))


# In[ ]:





# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt
plt = plt.figure(figsize=(15,5))
c = skm.confusion_matrix(test_labels.detach(), y_predicted, normalize='pred')
plt_1 =sns.heatmap(c,annot=True,cmap='summer',xticklabels=mappings,yticklabels=mappings)
plt_1.set_title('Confusion matrix\n\n');
plt_1.set_xlabel('\nGenre')
plt_1.set_ylabel('\nGenre')





# In[ ]:


print(y_predicted.shape)


# In[ ]:


print(test_labels.shape)
# print(test_labels)


# In[ ]:


y_predicted_labels = torch.softmax(y_predicted_feature.cpu(),axis=1)


# In[ ]:


y_predicted_labels[1]


# In[ ]:


map=['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop']


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop']

ax.bar(langs,y_predicted_labels[1])
plt.show()


# In[ ]:




