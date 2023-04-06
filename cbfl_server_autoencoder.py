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
import subprocess
import concurrent.futures
from collections import OrderedDict
from sklearn.cluster import KMeans
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
import os


global PATH
PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/cbfl_task/'
PATH_DATA = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/task_data/'
PATH_FL_SCRIPT = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/code/'
N_CLUSTERS = 3
FT_TYPES = ['meds', 'dx', 'physio']
DIMS = {'meds':1056, 'dx':483, 'physio': 7}
HOSPITALS=[264,142,148,281,154,283,157,420,165,167,176,449,199,458,79,338,227,248,122,252]

#MODELS
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
    

#FEDAVG
def FedAvg(hospitals, global_model, model):
    n = hospitals.shape[0]
    # Load the state dicts for each hospital model and set them to eval mode
    hospital_params_list = []
    # Set the weights for each hospital
    weights = hospitals['weight'].values

    hospital_params_list = []
    for i, hosp in enumerate(hospitals.index):
        hospital_params = torch.load(f'{PATH}{hosp}/{model}.pt')
        hospital_params_list.append(hospital_params)

    # Compute the weighted average of the model parameters
    global_params = OrderedDict()
    for key in hospital_params_list[0]:
        global_params[key] = torch.zeros(hospital_params_list[0][key].shape)
    
    for i, hospital_params in enumerate(hospital_params_list):
        for key in hospital_params:
            global_params[key] += hospital_params[key] * weights[i]
    
    # Set the global model parameters to the averaged parameters
    global_model.load_state_dict(global_params)
    return global_model

def runFedAvg(hospitals, model, feat):
    # run once for each model
    global_model = Autoencoder(DIMS[feat])
    global_model = FedAvg(hospitals, global_model, f'{model}_{feat}')
    return global_model

    
#CLUSTERING
def run_kmeans(hospitals):
    #load avg embedding
    avg_embds = []
    for hosp in hospitals.index:
        avg_embds.append(np.loadtxt(f'{PATH}{hosp}/avgembedding'))
    avg_embds = np.array(avg_embds, dtype = np.float32)
    #run k-means
    kmeans = KMeans(n_clusters = N_CLUSTERS, random_state = 1).fit(avg_embds)
    #save models
    for hosp in hospitals.index:
        dump(kmeans, f'{PATH}{hosp}/kmeans.joblib')
        
#COORDINATION
def clear_clients(hosp, model):
    ##clear models from clients
    command = f'rm {PATH}{hosp}/{model}.pt ' 
    subprocess.call(command, shell = True)
    return

def run_clients(hosp, model, run):
    command = f'python {PATH_FL_SCRIPT}client_{model}.py -cl={hosp} -rn={run}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response

def run_autoencoder(hospitals):
    #########CREATE COMMUNITY CLUSTERS#########
    MODEL = 'autoencoder'

    ##Train encoder        
    RUN = 'train'
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hosp in hospitals.index:
            futures.append(executor.submit(run_clients, hosp, MODEL, RUN))
    concurrent.futures.wait(futures)

    ##Average
    for feat in FT_TYPES:
        global_model = runFedAvg(hospitals, MODEL, feat)
        ##save
        for hosp in hospitals.index:
            torch.save(global_model.state_dict(), f'{PATH}{hosp}/global_autoencoder_{feat}.pt')


    ##Embed patients
    RUN = 'embed'
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hosp in hospitals.index:
            futures.append(executor.submit(run_clients,hosp, MODEL, RUN))
    concurrent.futures.wait(futures)

    ##k-means cluster based on avg embed
    run_kmeans(hospitals)

    ##Cluster patients
    RUN = 'cluster'
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hosp in hospitals.index:
            futures.append(executor.submit(run_clients,hosp, MODEL, RUN))
    concurrent.futures.wait(futures)
    return

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
    set_seed()
    #Load hospitals
    hospitals = pd.read_csv(f'{PATH_DATA}hospitals.csv', index_col = 'hospitalid')
    hospitals['weight'] = hospitals['count'] / hospitals['count'].sum()
    run_autoencoder(hospitals)

if __name__ == '__main__':
    main()
    