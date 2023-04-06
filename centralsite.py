#!/gpfs/commons/home/aelhussein/anaconda3/envs/pytorch_env/bin/python

import pandas as pd
import numpy as np
import pickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import argparse
from collections import OrderedDict
from sklearn.cluster import KMeans
from joblib import dump, load
from sklearn import metrics



PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/central_task/'
PATH_DATA = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/task_data/'
PATH_FL_SCRIPT = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/code/'
FT_TYPES = ['meds', 'dx', 'physio']
DIMS = {'meds':1056, 'dx':483, 'physio': 7}


#HYPERPARAMETERS
BATCH_SIZE = 30
LEARNING_RATE = 1e-3
BETAS=(0.9, 0.999)
EPOCHS = 200
TRAIN_SIZE = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#LOAD DATA
def load_data():
    mortality = pd.read_csv(f'{PATH_DATA}mortality.csv', usecols = ['patientunitstayid', 'expired'], index_col = 'patientunitstayid')
    drugs = pd.read_csv(f'{PATH_DATA}medications.csv', index_col = 'patientunitstayid')
    dx = pd.read_csv(f'{PATH_DATA}diagnosis.csv',  index_col = 'patientunitstayid')
    physio = pd.read_csv(f'{PATH_DATA}physio.csv', index_col = 'patientunitstayid')
    dem = pd.read_csv(f'{PATH_DATA}demographics.csv', usecols = ['patientunitstayid', 'age', 'admissionheight', 'admissionweight'], index_col = 'patientunitstayid')
    return drugs.apply(minmaxscale, axis = 1), dx.apply(minmaxscale, axis = 1), physio.apply(minmaxscale, axis = 1), dem, mortality

def minmaxscale(column):
 max_, min_ = column.max(), column.min()
 return (column - min_) / (max_ - min_)


def load_dataset(features, outcome):
    tensor_list = [torch.tensor(df.values, dtype=torch.float32) for df in features]
    outcome_tensor = torch.tensor(outcome.values, dtype=torch.float32)

    #dataset
    dataset = torch.utils.data.TensorDataset(*tensor_list, outcome_tensor)

    #split into test and train
    train_size = int(TRAIN_SIZE * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #load onto dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader


#MODELS
class FeedForward(nn.Module):
    def __init__(self, input_dim_drugs, input_dim_dx, input_dim_physio):
        super().__init__()
        
        self.input_dim_drugs = input_dim_drugs
        self.input_dim_dx = input_dim_dx
        self.input_dim_physio = input_dim_physio

        
        self.FF_meds = nn.Sequential(
                        nn.Linear(self.input_dim_drugs, 100),
                        nn.ReLU(),
                        nn.Linear(100, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10),
                        nn.ReLU(),
                        nn.Linear(10, 5)
                        )
        
        self.FF_dx = nn.Sequential(
                        nn.Linear(self.input_dim_dx, 100),
                        nn.ReLU(),
                        nn.Linear(100, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10),
                        nn.ReLU(),
                        nn.Linear(10, 5)
                        )
        
        self.FF_physio = nn.Sequential(
                        nn.Linear(self.input_dim_physio, 40),
                        nn.ReLU(),
                        nn.Linear(40, 20),
                        nn.ReLU(),
                        nn.Linear(20, 10),
                        nn.ReLU(),
                        nn.Linear(10, 5)
                        )
        
        self.FF_multihead = nn.Sequential(
                        nn.Linear(15, 15),
                        nn.ReLU(),
                        nn.Linear(15, 10),
                        nn.ReLU(),
                        nn.Linear(10, 5),
                        nn.ReLU(),
                        nn.Linear(5, 1),
                        nn.Sigmoid()
                        )

    def forward(self, x_drugs, x_dx, x_physio):
        meds = self.FF_meds(x_drugs)
        dx =  self.FF_dx(x_dx)
        physio = self.FF_physio(x_physio)
        ##concatentate
        x_concat = torch.cat((meds, dx, physio), dim = 1)
        #run through final head
        scores = self.FF_multihead(x_concat)
        return scores

#PREDICTION
#TRAIN
def train_model(train_dataloader, dims):
    # load model
    model = FeedForward(dims[0], dims[1], dims[2])
    model.load_state_dict(torch.load(f'{PATH}prediction.pt'))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, betas=BETAS)
    loss_fn = nn.BCELoss()

    # train model
    for epoch in range(EPOCHS):
        model.train()
        running_loss_train = 0.0
        k = 0
        for drugs, dx, physio, label in train_dataloader:
            drugs, dx, physio = drugs.to(device), dx.to(device), physio.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(drugs, dx, physio)
            loss = loss_fn(pred, label)  
            loss.backward()
            optimizer.step()
            k += 1
            running_loss_train += loss.item()
        running_loss_train /= k

        if (epoch) % 10 == 0:
            print(f"EPOCH: {epoch + 1} -> Train Loss: {running_loss_train:.5f}")
    return model

#TEST
def test_model(test_dataloader, dims):
    # load global model
    model = FeedForward(dims[0], dims[1], dims[2])
    model.load_state_dict(torch.load(f'{PATH}prediction.pt'))
    model.to(device)
    model.eval()
    # test global model
    predictions = []
    true_labels = []
    with torch.no_grad():
        for drugs, dx, physio, label in test_dataloader:
            drugs, dx, physio = drugs.to(device), dx.to(device), physio.to(device)
            pred = model(drugs, dx, physio)
            predictions.append(pred.detach().cpu().numpy())
            true_labels.append(label.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
    auc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(true_labels, predictions)
    return auc, auprc


def main(iteration):
    drugs_f, dx_f, physio_f, dem, outcome = load_data()
    features = {'meds':drugs_f, "dx": dx_f, "physio": physio_f}
    #initialize model
    dim0, dim1, dim2 = list(DIMS.values())
    initial_model = FeedForward(dim0, dim1, dim2)
    torch.save(initial_model.state_dict(), f'{PATH}prediction.pt')

    ##data onto dataloaders
    feat_data = [f for f in features.values()]
    dims = [features[f].shape[1] for f in FT_TYPES]
    train_dataloader, test_dataloader = load_dataset(feat_data, outcome)

    #train model
    model= train_model(train_dataloader, dims) 
    
    # Save models
    model.cpu()
    torch.save(model.state_dict(), f'{PATH}prediction.pt')
    
    #test model
    auc, auprc = test_model(test_dataloader, dims)
    
    ## Check if the average AUC is less than or equal to 0.5 i.e. whether model learned
    if auc<= 0.5:
        print("Average AUC is less than or equal to 0.5. Rerunning the function...")
        return main(iteration)
    else:
        ##Save results
        auc = pd.DataFrame([['central', auc]], columns = ['site', 'AUC'])
        auprc = pd.DataFrame([['central', auprc]], columns = ['site', 'AUPRC'])
        auc.to_csv(f'{PATH}results_{iteration}.csv', index = False)
        auprc.to_csv(f'{PATH}results_auprc_{iteration}.csv', index = False)
        return


if __name__ == '__main__':
    for iteration in range(100):
        main(iteration)
    
