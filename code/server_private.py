#!/gpfs/commons/home/aelhussein/anaconda3/envs/pytorch_env/bin/python

import pandas as pd
import numpy as np
import pickle
import sympy as sym
import concurrent.futures
import argparse
import subprocess
from sklearn.cluster import SpectralClustering
import itertools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from collections import OrderedDict

HOSPITALS=[264,142,148,281,154,283,157,420,165,167,176,449,199,458,79,338,227,248,122,252]
NPATIENTS = 250
EMB_DIM = 36
N_CLUSTERS = 3
PATH_DATA = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/task_data/'
PATH_FL_SCRIPT = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/code/'
FT_TYPES = ['meds', 'dx', 'physio']
DIMS = {'meds':1056, 'dx':483, 'physio': 7}
EPOCHS = 20

##Embedding
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
    for epoch in EPOCHS:        
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

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

##########################################################################################################################################################

#get combinations of hospital pairs
HOSPITAL_PAIRS = list(itertools.combinations(HOSPITALS,2))
COMBINATIONS = {}
for pair in HOSPITAL_PAIRS:
    if not str(pair[0]) in COMBINATIONS:
        COMBINATIONS[str(pair[0])] = []
    if not str(pair[1]) in COMBINATIONS:
        COMBINATIONS[str(pair[1])] = []
    
    COMBINATIONS[str(pair[0])].append(f'{pair[0]}_{pair[1]}')
    COMBINATIONS[str(pair[1])].append(f'{pair[0]}_{pair[1]}')

###Private calculation
def generatorMatrix(k):
    p = sym.nextprime(k)
    V = np.vander(np.arange(1,k//2 + 1), p - 1, increasing = True) % p
    G = np.array(sym.Matrix(V).rref(pivots = False), dtype = float)

    remaining_cols = G.shape[1] - (p-1-G.shape[1])
    G_ = G[:,:remaining_cols]
    
    ##Break down matrix G and G^-1 into A,B,C,D
    A = G_.T
    D = np.hstack((G_[:,k//2:].T, G_[:,:k//2] - 2* G_[:,:k//2]))
    ##B1 = I, B2 = -I + A2
    B = np.vstack((np.eye(k//2), np.eye(k//2)- 2*np.eye(k//2) + A[k//2:]))
    ##C1 = I-A2, C2 = I
    C =  np.hstack((np.eye(k//2) - A[k//2:], np.eye(k//2)))
    return A, B, C, D

def generate_client_matrices():
    A,B,C,D = generatorMatrix(EMB_DIM)
    for pair in HOSPITAL_PAIRS:
        hospital_matrices = {}
        hospital_matrices[pair[0]] = [A,B]
        hospital_matrices[pair[1]] = [C,D]
        with open(f'{PATH}private/matrix_{pair[0]}_{pair[1]}.pkl', 'wb') as file:
            pickle.dump(hospital_matrices, file)
    return

def privateDotproduct():
    dotproducts = {}
    for comb in HOSPITAL_PAIRS:
        V1 = np.loadtxt(f'{PATH}private/matrix_{comb[0]}_{comb[1]}_V_0')
        V2 = np.loadtxt(f'{PATH}private/matrix_{comb[0]}_{comb[1]}_V_1')
        dotproduct = V1 + V2
        dotproducts[comb] = dotproduct

    for hosp in HOSPITALS:
        dotproduct = np.loadtxt(f'{PATH}private/matrix_{hosp}_sim')
        dotproducts[(hosp,hosp)] = dotproduct

    return dotproducts

def run_clients(hosp, task):
    command = f'python {PATH_FL_SCRIPT}client_private.py -cl={hosp} -tk={task}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response

def cos_sim_matrix(dotproducts):
    n = len(HOSPITALS)
    cos_sim = np.zeros((NPATIENTS*n, NPATIENTS*n))
    for hosp1 in HOSPITALS:
        for hosp2 in HOSPITALS:
            i = HOSPITALS.index(hosp1)
            j = HOSPITALS.index(hosp2)
            try:
                mat = dotproducts[(hosp1, hosp2)]
            except KeyError:
                mat = dotproducts[(hosp2, hosp1)]
            cos_sim[i*NPATIENTS:(i+1)*NPATIENTS, j*NPATIENTS:(j+1)*NPATIENTS] = mat.T
            cos_sim[j*NPATIENTS:(j+1)*NPATIENTS, i*NPATIENTS:(i+1)*NPATIENTS] = mat
    np.savetxt(f'{PATH}private/cos_sim', cos_sim)
    return cos_sim


def cluster_patients(cos_sim):
    patients_all = pd.DataFrame()
    for hospid in HOSPITALS:
        patients = pd.read_csv(f'{PATH_DATA}{hospid}/mortality.csv', usecols = ['patientunitstayid', 'hospitalid'])
        patients_all = pd.concat([patients_all, patients])

    cos_sim = pd.DataFrame(cos_sim, index = patients_all['patientunitstayid'], columns = patients_all['patientunitstayid'])
    # conduct spectral clustering i.e. k-means on the eigenvectors of the matrix
    sc = SpectralClustering(n_clusters=N_CLUSTERS, affinity='precomputed', eigen_solver='arpack', random_state = 1)
    clusters = sc.fit_predict(cos_sim)
    patients_all['cluster'] = clusters

    for hospid in HOSPITALS:
        cluster_hosp = patients_all[['patientunitstayid', 'cluster']][patients_all['hospitalid'] == hospid]
        cluster_hosp.to_csv(f'{PATH}{hospid}/clusters.csv', index = False)
        np.savetxt(f'{PATH}{hospid}/site_clusters', cluster_hosp['cluster'].unique(), fmt = '%i')
    return

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    set_seed()
    #EMBEDDING
    #Load hospitals
    hospitals = pd.read_csv(f'{PATH_DATA}hospitals.csv', index_col = 'hospitalid')
    hospitals['weight'] = hospitals['count'] / hospitals['count'].sum()
    run_autoencoder(hospitals)

    #PRIVACY
    #generate matrices
    generate_client_matrices()

    #client calculations
    task = 'matrix'
    futures = [] 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hosp in HOSPITALS:
            futures.append(executor.submit(run_clients, hosp, task))
    concurrent.futures.wait(futures)


    #client calculations
    task = 'dot'
    futures = [] 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hosp in HOSPITALS:
            futures.append(executor.submit(run_clients, hosp, task))
    concurrent.futures.wait(futures)


    #calculate dot products
    dotproducts  = privateDotproduct()
    #get matrix
    cos_sim = cos_sim_matrix(dotproducts)

    #cluster patients
    cluster_patients(cos_sim)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--modeltype', default = 'emb')
    
    args = parser.parse_args()
    global MODELTYPE
    MODELTYPE= args.modeltype

    global PATH
    if MODELTYPE == 'emb':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/emb_task/'
    elif MODELTYPE =='p_cbfl':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/p_cbfl_task/'
        
    main()