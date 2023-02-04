import torch 
import torch.nn as nn
import pandas as pd
from numpy import sum
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader
from tqdm import tqdm

#download data from https://www.kaggle.com/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcard.csv')

# distribution of legit  & fraudlen transactions 

print(df['Class'].value_counts())

# THIS DATASET IS HIGHLY UNBALANCED
#  0 -----> Normal Transactions
#  1 -----> Fraudulent transactions

# separating the data for analysis
legit = df[df.Class == 0]
fraud = df[df.Class == 1]

# Number of Fradulent transaction --> 492
legitsample =legit.sample(n=492)

# concatenating two DataFrames
# axis 0 = rows
# axis 1 = columns
newDataset = pd.concat([legitsample,fraud],axis=0)
print('After balance Data')
print(newDataset['Class'].value_counts())

# splitting the data into Features & Targets
X = newDataset.drop(columns='Class',axis=1)
Y = newDataset['Class']

scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)

xTrain,xTest,yTrain,yTest = train_test_split(X,Y,test_size=0.20,stratify=Y)

# dataframe to tensor
xTrain = torch.FloatTensor(xTrain)
xTest = torch.FloatTensor(xTest)
yTrain = torch.FloatTensor(yTrain.values)
yTest = torch.FloatTensor(yTest.values)

trainDataset = TensorDataset(xTrain,yTrain)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Available Device: {device}')

batchSize = 128
trainDataloader = DataLoader(trainDataset,batch_size=batchSize,shuffle=True)


class LogisticReg(torch.nn.Module):
    def __init__(self,inpDim,outDim) -> None:
        super().__init__()
        print(f'model of input: {inpDim}, outPut: {outDim}')
        self.linear = torch.nn.Linear(inpDim,outDim)
    
    def forward(self,x):
        out = torch.sigmoid(self.linear(x))
        return out


def trainModel(model,epochs,lossFunc,opt):
    model.train()
    for e in range(epochs):
        with tqdm(trainDataloader,unit='Batch') as tq:
            for data,target in tq:
                data,target = data.to(device),target.to(device)
                tq.set_description(f'Epoch: {e+1}')

                opt.zero_grad()
                preds = model(data)
                loss = lossFunc(torch.squeeze(preds),target)
                loss.backward()

                opt.step()
                tq.set_postfix(loss=loss.item())

INP_DIM =xTrain.shape[1]
OUT_DIM = 1

model = LogisticReg(inpDim=INP_DIM,outDim=OUT_DIM).to(device=device)
lossFunc = nn.BCELoss()

LEARING_RATE = 0.01
optim = torch.optim.SGD(model.parameters(),lr=LEARING_RATE)

EPOCHS = 1000
trainModel(model=model,epochs=EPOCHS,lossFunc=lossFunc,opt=optim)

