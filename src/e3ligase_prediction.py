#!/usr/bin/env python
# coding: utf-8

# # Fingerprint generator
# 
# This notebook is used to generate the fingerprints from RDKit that are used as training data for the ML models.

# In[2]:


# In[ ]:

#pip install -r requirements.txt

print('Importing necessary packages')
# In[1]:

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, DataStructs

from mhfp.encoder import MHFPEncoder


import pickle
import json
import optuna

import logging
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from treeinterpreter import treeinterpreter as ti
from hpo_trainer import objective_rf, report, SEED, finalize_model, objective_xgboost

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

###### Part 1: Load test dataset and generate FPs

# In[2]:
print('Prediction initiated')

df = pd.read_csv('../data/e3binder_predictor_model/input/20240313_MGL.csv')


# In[3]:


df = pd.DataFrame(df['SMILES'])
print('Initial length of dataframe: ',len(df))

# In[4]:

#remove duplicates after generating inchikey for each smiles

temp = []

for item in df['SMILES']:
    mol = Chem.MolFromSmiles(item)
    ikey = Chem.MolToInchiKey(mol)
    temp.append(ikey)

df['inchikey'] = temp


# In[5]:

df = df.drop_duplicates(subset=['inchikey'], keep='first')

print('Length of dataframe after pre-processing: ',len(df))

# # Generate the different fingerprints

# In[6]:

#Morgan FP
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024) # ECFP4 fingerprint
#rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024) # RDKit fingerprint


# In[7]:


# Disable warnings
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# In[8]:


ecfp_fingerprints = []
#rdkit_fingerprints = []
#maccs_fingerprints = []

for smiles in tqdm(df["SMILES"]):
    # Canonicalize the smiles
    try:
        can_smiles = Chem.CanonSmiles(smiles)
    except:
        can_smiles = smiles

    # Generate the mol object
    mol = Chem.MolFromSmiles(can_smiles)

    if not mol:
        ecfp_fingerprints.append(None)
        #rdkit_fingerprints.append(None)
        maccs_fingerprints.append(None)
        continue

    ecfp_fingerprints.append(mfpgen.GetFingerprint(mol))
    #rdkit_fingerprints.append(rdkgen.GetFingerprint(mol))
    #maccs_fingerprints.append(MACCSkeys.GenMACCSKeys(mol))


# In[9]:


df['ecfp4'] = ecfp_fingerprints
#amr_df['rdkit'] = rdkit_fingerprints
#df['maccs'] = maccs_fingerprints


# In[10]:


#df.head(3)


# In[11]:


#df.to_pickle("../data/processed/fingerprints.pkl")


# # Formatting data for each fingerprint
# 
# Each fingerprint vector is now converted to bit columns i.e. one bit per column.

# In[12]:


#os.makedirs("../data/fingerprints", exist_ok=True)


# ### ECFP4

# In[35]:


ecfp4_df = df[['ecfp4']]
ecfp4_df = ecfp4_df.dropna() # Drop rows with no fingerprint
ecfp4_df.shape


# In[40]:

data = []

for ecfp_fp in tqdm(ecfp4_df['ecfp4']):

    # Convert fingerprint to numpy array
    fp = np.zeros((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(ecfp_fp, fp)

    # Convert to dataframe
    t = pd.DataFrame(fp).T
    t.rename(columns=lambda x: "bit" + str(x), inplace=True)

    data.append(t)

ecfp4_dataframe = pd.concat(data, ignore_index=True)


# In[41]:


ecfp4_dataframe.head(5)


# In[53]:


ecfp4_dataframe.to_csv("../data/fingerprints/ecfp4.csv",index_label=False)

######## Part 2 - Prediction 

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Disabling trial info messages

logger = logging.getLogger(__name__)

#load trained model

xgboost_model = pickle.load(open(f"../data/e3ligase_predictor_model/models/ecfp4_xgboost.pickle.dat", "rb"))

test_data = pd.read_csv(f"../data/fingerprints/ecfp4.csv",index_col=False)

y_pred = xgboost_model.predict(test_data)

y_pred_prob = xgboost_model.predict_proba(test_data)
df_predicted = pd.DataFrame(y_pred_prob)

df_predicted = pd.DataFrame(y_pred_prob)

df_predicted['Predicted'] = list(y_pred)

#need to map original labels as prediction generates integers
label_to_idx = {0: "CRBN", 1: "VHL", 2: "IAP", 3: "XIAP", 4: "Other"}

# Label mapping to integers
df_predicted['Predicted'] = df_predicted['Predicted'].map(label_to_idx)

colnames = 'CRBN VHL IAP XIAP Other Predicted'.split(" ")

df_predicted.columns = colnames

df_predicted['SMILES'] = list(df["SMILES"])

os.makedirs("../data/e3ligase_predictor_model/output", exist_ok=True)

df_predicted.to_csv("../data/e3ligase_predictor_model/output/e3_ligase_prediction.csv",index = False)

print('Process completed. Please look at the results in e3liagse_predictor_model/output folder')