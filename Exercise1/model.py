import torch.nn as nn

# Make it simple at the beginning..

class Net(nn.Module):
    
    '''
        input = dimension 69*69
        output = dimension 10
    '''
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.layer1 = nn.Linear(69*69,256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256,128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128,64)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(64,10)

    def forward(self,x):

        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        
        # ... you need to pass the input through all the layers

        return x
    