#!/gpfs/commons/home/aelhussein/anaconda3/envs/pytorch_env/bin/python

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import argparse
from sklearn import metrics

#HYPERPARAMETERS
BATCH_SIZE = 30
LEARNING_RATE = 1e-3
BETAS=(0.9, 0.999)
EPOCHS = 7
TRAIN_SIZE = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
FT_TYPES = ['meds', 'dx', 'physio']
torch.manual_seed(0)

#LOAD DATA
global PATH_DATA
PATH_DATA = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/task_data/'

def minmaxscale(column):
 max_, min_ = column.max(), column.min()
 return (column - min_) / (max_ - min_)

def load_data(hospid):
    mortality = pd.read_csv(f'{PATH_DATA}{hospid}/mortality.csv', usecols = ['patientunitstayid', 'expired'], index_col = 'patientunitstayid')
    drugs = pd.read_csv(f'{PATH_DATA}{hospid}/medications.csv', index_col = 'patientunitstayid')
    dx = pd.read_csv(f'{PATH_DATA}{hospid}/diagnosis.csv',  index_col = 'patientunitstayid')
    physio = pd.read_csv(f'{PATH_DATA}{hospid}/physio.csv', index_col = 'patientunitstayid')
    dem = pd.read_csv(f'{PATH_DATA}{hospid}/demographics.csv', usecols = ['patientunitstayid', 'age', 'admissionheight', 'admissionweight'], index_col = 'patientunitstayid')
    return drugs.apply(minmaxscale, axis = 1), dx.apply(minmaxscale, axis = 1), physio.apply(minmaxscale, axis = 1), dem, mortality


def load_dataset(features_, outcome):
    tensor_list = [torch.tensor(df.values, dtype=torch.float32) for df in features_]
    outcome_tensor = torch.tensor(outcome.values, dtype=torch.float32)

    #dataset
    dataset = torch.utils.data.TensorDataset(*tensor_list, outcome_tensor)

    #split into test and train
    train_size = int(TRAIN_SIZE * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    #load onto dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader

#MODEL
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


#TRAIN
def train_model(features, outcome):
    #load to tensor
    features_ = [f for f in features.values()]
    train_dataloader, _ = load_dataset(features_, outcome)
    # load model
    dims = [features[f].shape[1] for f in FT_TYPES]
    model = FeedForward(dims[0], dims[1], dims[2])
    model.load_state_dict(torch.load(f'{PATH}{hospid}/prediction.pt'))
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

        if (epoch) % 100 == 0:
            print(f"EPOCH: {epoch + 1} -> Train Loss: {running_loss_train:.5f}")
    return model

#TEST
def test_model(features, outcome):
    # load data
    features_ = [f for f in features.values()]
    _, test_dataloader = load_dataset(features_, outcome)

    # load global model
    dims = [features[f].shape[1] for f in FT_TYPES]
    model = FeedForward(dims[0], dims[1], dims[2])
    model.load_state_dict(torch.load(f'{PATH}{hospid}/global_prediction.pt'))
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
   


def train_model_c(features, outcome, clusters):
    cluster_models  = {}
    for i in SITE_CLUSTER:
        cluster_pids = clusters['patientunitstayid'][clusters['cluster'] == i]
        cluster_features = [f.loc[cluster_pids] for f in features.values()]
        cluster_outcome = outcome.loc[cluster_pids]

        #load to tensor
        train_dataloader, _ = load_dataset(cluster_features, cluster_outcome)

        # load model
        dims = [features[f].shape[1] for f in FT_TYPES]
        model = FeedForward(dims[0], dims[1], dims[2])
        model.load_state_dict(torch.load(f'{PATH}{hospid}/prediction_cluster_{i}.pt'))
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

            if (epoch) % 100 == 0:
                print(f"EPOCH: {epoch + 1} -> Train Loss: {running_loss_train:.5f}")
        cluster_models[i] = model
    return cluster_models

#TEST
def test_model_c(features, outcome, clusters):
    cluster_auc  = {}
    cluster_auprc = {}
    for i in SITE_CLUSTER:
        # load data
        cluster_pids = clusters['patientunitstayid'][clusters['cluster'] == i]
        cluster_features = [f.loc[cluster_pids] for f in features.values()]
        cluster_outcome = outcome.loc[cluster_pids]

        # load data
        _, test_dataloader = load_dataset(cluster_features, cluster_outcome)
    

        # load global model
        dims = [features[f].shape[1] for f in FT_TYPES]
        model = FeedForward(dims[0], dims[1], dims[2])
        model.load_state_dict(torch.load(f'{PATH}{hospid}/global_prediction_cluster_{i}.pt'))
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
        cluster_auc[i] = auc
        cluster_auprc[i] = auprc
    return cluster_auc, cluster_auprc
   
def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-rn', '--run', default = 'train')
    parser.add_argument('-tk', '--task', default = 'mortality')
    parser.add_argument('-mt', '--modeltype', default = 'emb')

    
    args = parser.parse_args()
    global hospid
    hospid = args.client
    global run
    run = args.run
    global task
    task = args.task
    global modeltype
    modeltype = args.modeltype
    
    global PATH
    if modeltype == 'emb':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/emb_task/'
    elif modeltype =='cbfl':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/cbfl_task/'
    elif modeltype =='p_cbfl':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/p_cbfl_task/'
    elif modeltype == 'avg':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/avg_task/'

    # Load data
    drugs_f, dx_f, physio_f, dem, outcome = load_data(hospid)
    features = {'meds':drugs_f, "dx": dx_f, "physio": physio_f}
    
    if modeltype != 'avg':
        global SITE_CLUSTER
        clusters = pd.read_csv(f'{PATH}{hospid}/clusters.csv')
        SITE_CLUSTER = clusters['cluster'].unique()
    
        if run == 'train':
            # Train models
            cluster_models = train_model_c(features, outcome, clusters)
            # Save models
            for i in SITE_CLUSTER:
                model = cluster_models[i]
                model.cpu()
                torch.save(model.state_dict(), f'{PATH}{hospid}/prediction_cluster_{i}.pt')
            print(f'{hospid} completed')
        
        elif run == 'test':
            #Test m:xodels
            cluster_auc, cluster_auprc = test_model_c(features, outcome, clusters)
            #Save results
            auc = pd.DataFrame.from_dict(cluster_auc, orient = 'index', 
                                    columns = ['AUC']).reset_index().rename(
                                    columns={'index': 'cluster'})
            
            auprc = pd.DataFrame.from_dict(cluster_auprc, orient = 'index', 
                                    columns = ['AUPRC']).reset_index().rename(
                                    columns={'index': 'cluster'})
            auc.to_csv(f'{PATH}{hospid}/results.csv', index = False)
            auprc.to_csv(f'{PATH}{hospid}/results_auprc.csv', index = False)
        
    elif modeltype == 'avg':
        if run == 'train':
            # Train models
            model = train_model(features, outcome)
            model.cpu()
            torch.save(model.state_dict(), f'{PATH}{hospid}/prediction.pt')
            print(f'{hospid} completed')
        
        elif run == 'test':
            #Test models
            auc, auprc = test_model(features, outcome)
            #Save results
            auc = pd.DataFrame([[hospid, auc]], columns = ['site', 'AUC'])
            auprc = pd.DataFrame([[hospid, auprc]], columns = ['site', 'AUPRC'])
            auc.to_csv(f'{PATH}{hospid}/results.csv', index = False)
            auprc.to_csv(f'{PATH}{hospid}/results_auprc.csv', index = False)
    
    print(f'{hospid} COMPLETED TRAINING ROUND')
    
if __name__ == '__main__':
    main()
