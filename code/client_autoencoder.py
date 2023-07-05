#!/gpfs/commons/home/aelhussein/anaconda3/envs/pytorch_env/bin/python

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import argparse
from joblib import dump, load
from sklearn.cluster import KMeans
import os

#HYPERPARAMETERS
BATCH_SIZE = 30
LEARNING_RATE = 1e-3
BETAS=(0.9, 0.999)
EPOCHS = 10
TRAIN_SIZE = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
FT_TYPES = ['meds', 'dx', 'physio']

#LOAD DATA
global PATH
PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/cbfl_task/'
global PATH_DATA
PATH_DATA = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/task_data/'

def minmaxscale(column):
   max_, min_ = column.max(), column.min()
   return (column - min_) / (max_ - min_)

def load_data(hospid):
    mortality = pd.read_csv(f'{PATH_DATA}{hospid}/mortality.csv', usecols = ['patientunitstayid', 'expired'])
    drugs = pd.read_csv(f'{PATH_DATA}{hospid}/medications.csv', index_col = 'patientunitstayid')
    dx = pd.read_csv(f'{PATH_DATA}{hospid}/diagnosis.csv',  index_col = 'patientunitstayid')
    physio = pd.read_csv(f'{PATH_DATA}{hospid}/physio.csv', index_col = 'patientunitstayid')
    dem = pd.read_csv(f'{PATH_DATA}{hospid}/demographics.csv', usecols = ['patientunitstayid', 'age', 'admissionheight', 'admissionweight'], index_col = 'patientunitstayid')
    return drugs.apply(minmaxscale, axis = 1), dx.apply(minmaxscale, axis = 1), physio.apply(minmaxscale, axis = 1), dem, mortality


#Corrupt drug features with prob 0.5
class CorruptDataset(Dataset):
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # randomly set 50% of the features to zero
        mask = torch.bernoulli(torch.ones_like(self.features[idx]) * 0.5).bool()
        corrupted_features = self.features[idx] * mask.float()
        return corrupted_features
    
#Load data onto bathcloader
def load_dataset(df):
    #load to tensor
    features_tensor = torch.FloatTensor(np.array(df))
    dataset = CorruptDataset(features_tensor)

    #split into test and train
    train_size = int(TRAIN_SIZE * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #load onto dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader

#AUTOENCODER
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
                        nn.Linear(self.input_dim, self.input_dim//4),
                        nn.ReLU(),
                        nn.Linear(self.input_dim//4, self.input_dim//6),
                        nn.ReLU(),
                        nn.Linear(self.input_dim//6, 12)
                        )
        self.emb = None
        
        self.decoder = nn.Sequential(
                        nn.Linear(12, self.input_dim//6),
                        nn.ReLU(),
                        nn.Linear(self.input_dim//6, self.input_dim//4),
                        nn.ReLU(),
                        nn.Linear(self.input_dim//4, self.input_dim),
                        nn.Sigmoid()
                        )
        
    def forward(self, x):
        self.emb = self.encoder(x)
        reconstruction = self.decoder(self.emb)
        return reconstruction
        
    def get_embedding(self, df):
        embedding = self.encoder(df)
        return embedding
    

#TRAIN
def train_model(df):
    dim = df.shape[1]
    model = Autoencoder(dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, betas=BETAS)
    loss_fn = nn.MSELoss()

    train_dataloader, test_dataloader = load_dataset(df)


    for epoch in range(EPOCHS):
        model.train()
        running_loss_train = 0.0
        i = 0
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, data)  
            loss.backward()
            optimizer.step()
            i += 1
            running_loss_train += loss.item()
            
        running_loss_train /= i
        
        i = 0
        with torch.no_grad():
            running_loss_test = 0.0
            for data in test_dataloader:
                data = data.to(device)
                pred = model(data)
                loss = loss_fn(pred, data)  
                i+=1
                running_loss_test += loss.item()
            running_loss_test /= i

        
        print(f"EPOCH: {epoch + 1} -> Train Loss: {running_loss_train:.4f}; Test Loss: {running_loss_test:.4f}")
    return model

def get_emb(features, feat_type):
    features_tensor = torch.FloatTensor(np.array(features))
    features_tensor = features_tensor.to(device)
    #initialise
    dim = features.shape[1]
    model = Autoencoder(dim)
    # load global model
    model.load_state_dict(torch.load(f'{PATH}{hospid}/global_autoencoder_{feat_type}.pt'))
    model.eval()
    embeddings = model.get_embedding(features_tensor)
    embeddings = embeddings.detach().numpy()
    return embeddings

def cluster_patients(features, embeddings):
    kmeans = load(f'{PATH}{hospid}/kmeans.joblib')
    cluster = kmeans.predict(embeddings)
    clusters = pd.DataFrame((features.index.values),  columns = ['patientunitstayid'])
    clusters['cluster'] = cluster
    return clusters

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

   
def main():
    #set seed
    set_seed()
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-rn', '--run', default = 'train')
    
    args = parser.parse_args()
    global hospid
    hospid = args.client
    global run
    run = args.run
    
    # Load data
    drugs_f, dx_f, physio_f, dem, mortality = load_data(hospid)
    features = {'meds':drugs_f, "dx": dx_f, "physio": physio_f}
    
    if run == 'train':
        for feat_type in FT_TYPES:
            # Train model
            model = train_model(features[feat_type])
            # Save model
            model.cpu()
            torch.save(model.state_dict(), f'{PATH}{hospid}/autoencoder_{feat_type}.pt')
        print(f'{hospid} completed')
    
    elif run == 'embed':
        feat_embs = {}
        for feat_type in FT_TYPES:
            ##get embs
            emb = get_emb(features[feat_type], feat_type)
            feat_embs[feat_type] = emb.mean(axis  = 0)
        avg_emb = np.concatenate(list(feat_embs.values()))
        np.savetxt(f'{PATH}{hospid}/avgembedding', avg_emb, fmt = '%.9e')
            
    else:
        feat_embs = {}
        for feat_type in FT_TYPES:
            ##get embs
            feat_embs[feat_type] = get_emb(features[feat_type], feat_type)
        avg_emb = np.concatenate(list(feat_embs.values()), axis = 1)
        clusters = cluster_patients(list(features.values())[0], avg_emb)
        clusters.to_csv(f'{PATH}{hospid}/clusters.csv', index = False)
        site_cluster = clusters['cluster'].unique()
        np.savetxt(f'{PATH}{hospid}/site_clusters', site_cluster, fmt ='%i')
      
    print(f'{hospid} COMPLETED TRAINING ROUND')
    
if __name__ == '__main__':
    main()
