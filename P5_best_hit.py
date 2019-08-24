
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file = "16S_blast_org.csv"
df = pd.read_csv(file)
df.head()


# In[3]:


# include temperature
df_temp = pd.read_csv("16S_seq_dic_temp.tsv", sep="\t", index_col=0)
df_temp.head()


# In[4]:


dfs = df.groupby("query_seq")


# In[5]:


# mesophile can have duplicates
thermal=[]
meso=[]
for key in dfs.groups:
    temp = dfs.get_group(key)
    sort = temp.sort_values(by=["hit_len", "identity"], ascending=False)
    thermal.append(sort.query_seq.iloc[0])
    meso.append(sort.hit_seq.iloc[0])
print("thermal unique: ", len(set(thermal)))
print("meso unique: ", len(set(meso)))


# In[6]:


thermal_file = "query_thermal.csv"
meso_file = "hit_meso.csv"
thermal_f = open(thermal_file, "w")
meso_f = open(meso_file, "w")
thermal_f.write("thermophile\n")
meso_f.write("mesophile\n")

for i in range(len(thermal)):
    assert thermal[i] not in meso, "Thermophile cannot be mesophile!"
    thermal_f.write(f"{thermal[i]}\n")
    meso_f.write(f"{meso[i]}\n")
    i += 1
thermal_f.close()
meso_f.close()
print("pass")

