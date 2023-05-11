import torch
import torch.nn as nn
import torch.nn.functional as F

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
            x = F.softmax(x,dim=-1)

            if self.print: print(f'Final output : {list(x.shape)}')
                
            return x
    
    #model instance
    net = musicNet(printtoggle)
    
    return net
                       