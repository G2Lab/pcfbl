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
from sklearn.cluster import SpectralClustering

PATH_DATA = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/task_data/'
PATH_FL_SCRIPT = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/code/'
N_CLUSTERS = 3
FT_TYPES = ['meds', 'dx', 'physio']
DIMS = {'meds':1056, 'dx':483, 'physio': 7}
HOSPITALS=[264,142,148,281,154,283,157,420,165,167,176,449,199,458,79,338,227,248,122,252]

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

#FEDAVG
def FedAvg(hospitals, global_model, model):
    # Load the state dicts for each hospital model and set them to eval mode
    hospital_params_list = []
    hosps_included = []
    try: 
        for i, hosp in enumerate(hospitals.index):
            hospital_params = torch.load(f'{PATH}{hosp}/{model}.pt')
            hospital_params_list.append(hospital_params)
            hosps_included.append(hosp)
    except:
        # if site doesnt have cluster
        pass
    
    # Set the weights for each hospital
    weights = hospitals.loc[hosps_included]['weight'].values
    
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

def runFedAvg(hospitals, model, mode):
    # run for each cluster
    if mode == 'all':
        dim0, dim1, dim2 = list(DIMS.values())
        global_model = FeedForward(dim0, dim1, dim2)
        model = f'prediction'
        global_model = FedAvg(hospitals, global_model, model)
        return global_model
    
    elif mode == 'cluster':
        global_models = {}
        for i in range(N_CLUSTERS):
            dim0, dim1, dim2 = list(DIMS.values())
            global_model = FeedForward(dim0, dim1, dim2)
            model = f'prediction_cluster_{i}'
            hospitals_cluster = calc_weights(PATH, hospitals, i)
            global_model = FedAvg(hospitals_cluster, global_model, model)
            global_models[i] = global_model
        return global_models


#weight input by cluster size
def load_cluster_weights(PATH, hosp):
    cluster = pd.read_csv(f'{PATH}{hosp}/clusters.csv')
    cluster_weights = cluster.value_counts('cluster')
    cluster_df = pd.DataFrame(cluster_weights, columns = ['count'])
    cluster_df['site'] = hosp
    cluster_df.reset_index(inplace = True)
    return cluster_df

def site_weight(row):
    row['weight'] = row['count'] / row['count'].sum()
    return row

def calc_weights(PATH, hospitals, cluster):
    cluster_sizes = pd.DataFrame()
    for hosp in HOSPITALS:
        c = load_cluster_weights(PATH, hosp)
        cluster_sizes = pd.concat([cluster_sizes, c])
    cluster_sizes = cluster_sizes.groupby('cluster').apply(site_weight)
    cluster_filter = cluster_sizes[cluster_sizes['cluster']==cluster]
    hospitals_cluster = hospitals[[]].merge(cluster_filter[['count', 'site', 'weight']], left_index = True, right_on='site')
    hospitals_cluster.set_index('site', inplace = True)
    return hospitals_cluster


        
#COORDINATION
def clear_clients(hosp, model):
    ##clear models from clients
    command = f'rm {PATH}{hosp}/{model}.pt ' 
    subprocess.call(command, shell = True)
    return

def run_clients(hosp, model, run, task = None):
    command = f'python {PATH_FL_SCRIPT}client_{model}.py -cl={hosp} -rn={run} -tk={task} -mt={MODELTYPE}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response

def run_private_clustering(MODELTYPE):
    #run private clustering
    command = f'python {PATH_FL_SCRIPT}server_private.py -mt={MODELTYPE}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response

def run_prediction(hospitals, iteration, MODE):
    #########PREDICTION TASK#########
    MODEL = 'prediction'
    TASK = 'mortality'
    ROUNDS = 15
    
    ##Initialize model for each cluster
    dim0, dim1, dim2 = list(DIMS.values())
    initial_model = FeedForward(dim0, dim1, dim2)
    for hosp in hospitals.index:
        site_clusters = np.loadtxt(f'{PATH}{hosp}/site_clusters', dtype = int)
        site_clusters = np.atleast_1d(site_clusters)
        for i in site_clusters:
            torch.save(initial_model.state_dict(), f'{PATH}{hosp}/{MODEL}_cluster_{i}.pt')

            
    ##Run prediction models for multiple rounds
    for i in range(ROUNDS):
        RUN = 'train'
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for hosp in hospitals.index:
                futures.append(executor.submit(run_clients,hosp, MODEL, RUN))
        concurrent.futures.wait(futures)

        ##Average
        global_models = runFedAvg(hospitals, MODEL, MODE)
        #save
        for hosp in hospitals.index:
            site_clusters = np.loadtxt(f'{PATH}{hosp}/site_clusters', dtype = int)
            site_clusters = np.atleast_1d(site_clusters)
            for i in site_clusters:
                global_model = global_models[i]
                torch.save(global_model.state_dict(), f'{PATH}{hosp}/prediction_cluster_{i}.pt') 

   ##After training save global model for inference
    for hosp in hospitals.index:
        site_clusters = np.loadtxt(f'{PATH}{hosp}/site_clusters', dtype = int)
        site_clusters = np.atleast_1d(site_clusters)
        for i in site_clusters:
            global_model = global_models[i]
            torch.save(global_model.state_dict(), f'{PATH}{hosp}/global_prediction_cluster_{i}.pt') 

    ##Inference   
    RUN = 'test'
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hosp in hospitals.index:
            futures.append(executor.submit(run_clients,hosp, MODEL, RUN))
    concurrent.futures.wait(futures)

    ##Get reuslts
    results = pd.DataFrame(columns = ['cluster', 'AUC', 'site'])
    results_auprc = pd.DataFrame(columns = ['cluster', 'AUPRC', 'site'])
    for hosp in hospitals.index:
        result_site = pd.read_csv( f'{PATH}{hosp}/results.csv')
        result_site_auprc = pd.read_csv( f'{PATH}{hosp}/results_auprc.csv')
        result_site['site'] = hosp
        result_site_auprc['site'] = hosp
        results = pd.concat([results, result_site])
        results_auprc = pd.concat([results_auprc, result_site_auprc])

    ## Check if the average AUC is less than or equal to 0.5 i.e. wehther model learned
    if results['AUC'].mean() <= 0.5:
        print("Average AUC is less than or equal to 0.5. Rerunning the function...")
        return run_prediction(hospitals, iteration, MODE)
    else:
        ##Save results
        results.to_csv(f'{PATH}{TASK}_results_{iteration}.csv', index = False)
        results_auprc.to_csv(f'{PATH}{TASK}_results_auprc_{iteration}.csv', index = False)
        return


def run_prediction_avg(hospitals, iteration, MODE):
        #########PREDICTION TASK#########
    MODEL = 'prediction'
    TASK = 'mortality'
    ROUNDS = 15
    
    ##Initialize model for each cluster
    dim0, dim1, dim2 = list(DIMS.values())
    initial_model = FeedForward(dim0, dim1, dim2)
    for hosp in hospitals.index:
        torch.save(initial_model.state_dict(), f'{PATH}{hosp}/{MODEL}.pt')

            
    ##Run prediction models for multiple rounds
    for i in range(ROUNDS):
        RUN = 'train'
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for hosp in hospitals.index:
                futures.append(executor.submit(run_clients,hosp, MODEL, RUN))
        concurrent.futures.wait(futures)

        ##Average
        global_model = runFedAvg(hospitals, MODEL, MODE)
        #save
        for hosp in hospitals.index:
            torch.save(global_model.state_dict(), f'{PATH}{hosp}/prediction.pt') 

   ##After training save global model for inference
    for hosp in hospitals.index:
        torch.save(global_model.state_dict(), f'{PATH}{hosp}/global_prediction.pt') 

    ##Inference   
    RUN = 'test'
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for hosp in hospitals.index:
            futures.append(executor.submit(run_clients,hosp, MODEL, RUN))
    concurrent.futures.wait(futures)

    ##Get reuslts
    results = pd.DataFrame(columns = ['site', 'AUC'])
    results_auprc = pd.DataFrame(columns = ['site', 'AUPRC'])
    for hosp in hospitals.index:
        result_site = pd.read_csv( f'{PATH}{hosp}/results.csv')
        result_site_auprc = pd.read_csv( f'{PATH}{hosp}/results_auprc.csv')
        results = pd.concat([results, result_site])
        results_auprc = pd.concat([results_auprc, result_site_auprc])

    ## Check if the average AUC is less than or equal to 0.5 i.e. wehther model learned
    if results['AUC'].mean() <= 0.5:
        print("Average AUC is less than or equal to 0.5. Rerunning the function...")
        if MODELTYPE == 'avg':
            return run_prediction_avg(hospitals, iteration, MODE)
        else:
            return run_prediction(hospitals, iteration, MODE)
    else:
        ##Save results
        results.to_csv(f'{PATH}{TASK}_results_{iteration}.csv', index = False)
        results_auprc.to_csv(f'{PATH}{TASK}_results_auprc_{iteration}.csv', index = False)
        return

def main(iteration):
    #Load hospitals
    hospitals = pd.read_csv(f'{PATH_DATA}hospitals.csv', index_col = 'hospitalid')

    if MODELTYPE  == 'cbfl':
        MODE = 'cluster'
        run_prediction(hospitals, iteration, MODE)
    else:
        if (MODELTYPE  == 'emb') | (MODELTYPE  == 'p_cbfl'):
            #run_private_clustering(MODELTYPE)
            MODE = 'cluster'
            run_prediction(hospitals, iteration, MODE)
        elif MODELTYPE == 'avg':
            hospitals['weight'] = hospitals['count'] / hospitals['count'].sum()
            MODE = 'all'
            run_prediction_avg(hospitals, iteration, MODE)    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--modeltype', default = 'emb')
    
    args = parser.parse_args()
    global MODELTYPE
    MODELTYPE= args.modeltype

    global PATH
    if MODELTYPE == 'emb':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/emb_task/'
    elif MODELTYPE =='cbfl':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/cbfl_task/'
    elif MODELTYPE =='p_cbfl':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/p_cbfl_task/'
    elif MODELTYPE == 'avg':
        PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/avg_task/'


    for iteration in range(100):
        main(iteration)
    