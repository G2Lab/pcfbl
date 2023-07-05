#!/gpfs/commons/home/aelhussein/anaconda3/envs/pytorch_env/bin/python

import pandas as pd
import numpy as np
import pickle
import pickle
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import itertools
import os


HOSPITALS=[264,142,148,281,154,283,157,420,165,167,176,449,199,458,79,338,227,248,122,252]
EMB_DIM = 36
PATH_DATA = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/task_data/'
PATH_FL_SCRIPT = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/code/'
PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/patient_pfl/20_sites/p_cbfl_task/'
AVAIL_LETTERS = {0: ['A', 'B'], 1: ['C', 'D']}

#get hospital pair combinations
HOSPITAL_PAIRS = list(itertools.combinations(HOSPITALS,2))
COMBINATIONS = {}
for pair in HOSPITAL_PAIRS:
    if not str(pair[0]) in COMBINATIONS:
        COMBINATIONS[str(pair[0])] = []
    if not str(pair[1]) in COMBINATIONS:
        COMBINATIONS[str(pair[1])] = []
    
    COMBINATIONS[str(pair[0])].append(f'{pair[0]}_{pair[1]}')
    COMBINATIONS[str(pair[1])].append(f'{pair[0]}_{pair[1]}')


def multipartyComp(X, M, POSITION):
    if POSITION == 0:
        A, B = M
        X1 = np.dot(X.T, A)
        X2 = np.dot(X.T, B)

    elif POSITION == 1:
        C, D = M
        X1 = np.dot(C, X)
        X2 = np.dot(D, X)
    return X1, X2

def check_position(comb, hospid):
    #determin if A,B or C,D matrices for the site
    sites = [x for x in comb.split('_')]
    return sites.index(hospid)


def run_mat_calculation(hospid):
    for comb in COMBINATIONS[hospid]:
        POSITION = check_position(comb, hospid)
        #load matrix
        with open(f'{PATH}private/matrix_{comb}.pkl', 'rb') as file:
            matrix = pickle.load(file)
        #Load embedding
        embedding = pd.read_csv(f'{PATH}{hospid}/embedding.csv', index_col = 'patientunitstayid')
        embedding = normalize(embedding, norm = 'l2', axis = 1)
        #Calculate half matrices
        X1, X2 = multipartyComp(embedding.T, matrix[int(hospid)], POSITION)
        #save
        letters = AVAIL_LETTERS[POSITION]
        np.savetxt(f'{PATH}private/matrix_{comb}_{letters[0]}', X1)
        np.savetxt(f'{PATH}private/matrix_{comb}_{letters[1]}', X2)
    return


def run_dot_calculation(hospid):
    for comb in COMBINATIONS[hospid]:
        POSITION = check_position(comb, hospid)
        letters = [x[POSITION] for x in list(AVAIL_LETTERS.values())]
        #load matrix
        m1 = np.loadtxt(f'{PATH}private/matrix_{comb}_{letters[0]}')
        m2 = np.loadtxt(f'{PATH}private/matrix_{comb}_{letters[1]}')
        V = np.dot(m1, m2)
        np.savetxt(f'{PATH}private/matrix_{comb}_V_{POSITION}', V)
    return

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-tk','--task')
    args = parser.parse_args()
    global hospid
    hospid = args.client
    task = args.task

    if task == 'matrix':
        run_mat_calculation(hospid)

    elif task == 'dot':
        run_dot_calculation(hospid)
        
        #own patients
        embedding = pd.read_csv(f'{PATH}{hospid}/embedding.csv', index_col = 'patientunitstayid')
        cos_sim = cosine_similarity(embedding)
        np.savetxt(f'{PATH}private/matrix_{hospid}_sim', cos_sim)

    return

if __name__ == '__main__':
    main()
