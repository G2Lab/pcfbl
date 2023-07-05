# Privacy-preserving patient clustering for personalized federated learning
This README accompanies the paper “Privacy-preserving patient clustering for personalized federated learning”. The main contribution of the paper is to provide a method to calculate individual-level patient similarity scores without leaking patient information. The patient similarity scores can be used to cluster patients into clinically meaningful groups for downstream analysis. The README is a step-by-step guide to replicate the main findings of the paper. 
## PCBFL protocol
![schematic_pcbfl](https://github.com/G2Lab/pcfbl/assets/43360672/30fa78e3-984c-490b-be7a-7ed6cf1392d4)

The figure above highlights the main steps of PBCFL:
1.	A federated autoencoder is trained to embed patient clinical records into 30-dimension vectors.
2.	Patient similarity is estimated using Secure Multi-party Computation (SMPC), using Du et al., 2004 protocol.
3.	Patients are clustered into groups using spectral clustering of the similarity matrix calculated.
4.	A separate prediction model is trained on each cluster. 
## Comparators
We compare the performance of our protocol (PCBFL) against:
- CBFL (Huang et al., 2019) (federated)
- Traditional FedAvg (federated)
- Single site training (not federated)
- Centralized training (not federated) 
## Data
All data can be downloaded from the publicly available eICU dataset (https://physionet.org/content/eicu-crd/2.0/, credentialed user access is required). 
To replicate the findings, downloaded data *must* be processed using the  ```code/fl_task_data_processing.ipynb workbook```. 
## Running the pipeline
The scripts directory can be used to run the full pipeline for each protocol. Each protocol has its own script which can be called via the ```bash``` or ```sbatch``` (if SLURM available) commands. Results for each run will be automatically saved and available for downstream analysis. 
For example, to run PCBFL, simply run the command:
```
bash scripts/script_pcbfl
```
Note, the scripts should be updated to reflect the code directory on your machine. 
## Code requirements
All code was written in Python 3.9.7 and Pytorch 1.12.1
