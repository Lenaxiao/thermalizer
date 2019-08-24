
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path


# In[3]:


name_folder = "blastp_best_result/"
thermal_seq_folder = "thermal_fasta/"
meso_seq_folder = "meso_fasta/"

target_folder = "blastp_best_result_final/"


# In[ ]:


if not os.path.exists(target_folder):
    os.mkdir(target_folder)


# In[ ]:


def append_seq(df, file, seq_type):
    if os.path.exists(file):
        lines = open(file, "r").readlines()
        for i, line in enumerate(lines):
            if not df[seq_type+"_seq"].isnull().values.any():
                break
            if i % 2 == 0:
                name_list = list(df[seq_type])
                obj = line[1:-1]
                if obj in name_list:
                    index = [idx for idx, val in enumerate(name_list) if val == obj]
                    df.at[index, seq_type+"_seq"] = lines[i+1][:-1]
    return df


# In[ ]:


def query_name(i, file, thermal_seq_folder, meso_seq_folder, target_folder):
    if i == 0:
        print("Start:")
    elif i % 20 == 0:
        print("{0:.1f}% Completed!".format(i / 214 * 100))
    thermal_meso = os.path.basename(file).split("-")
    meso = meso_seq_folder + thermal_meso[0] + ".fasta"
    thermal = thermal_seq_folder + thermal_meso[1][:-4] + ".fasta"
    
    df = pd.read_csv(file)
    df["thermal_seq"] = np.nan
    df["meso_seq"] = np.nan
    
    df_add_meso = append_seq(df, meso, "meso")
    df_add_thermal = append_seq(df_add_meso, thermal, "thermal")

    df_add_thermal.to_csv(target_folder+os.path.basename(file))


# In[ ]:


files = Path(name_folder).glob("*.csv")
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores-5)(delayed(query_name)(i, file, thermal_seq_folder, meso_seq_folder, target_folder) for i, file in enumerate(files))

