
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import os
from joblib import Parallel, delayed
import multiprocessing


# In[2]:


folder = "blastp_result/"
meso_file = "best_hit_org/hit_meso.csv"
thermal_file = "best_hit_org/query_thermal.csv"
best_hit_folder = "blastp_best_result/"


# In[ ]:


if not os.path.exists(best_hit_folder):
    os.mkdir(best_hit_folder)


# In[6]:


files = Path(folder).glob("*.csv")


# In[ ]:


def blastp_hit(file, best_hit_folder):
    df = pd.read_csv(file, sep=",")
    df_groups = df.groupby("query_seq")
    tofile = best_hit_folder + os.path.basename(file)
    f = open(tofile, "w")
    f.write("thermal,meso\n")
    for key in df_groups.groups:
        temp = df_groups.get_group(key)
        sort = temp.sort_values(by=["hit_len", "identity"], ascending=False)
        f.write(f"{sort.query_seq.iloc[0]},{sort.hit_seq.iloc[0]}\n")

    f.close()


# In[ ]:


num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores-5)(delayed(blastp_hit)(file, best_hit_folder) for file in files)

