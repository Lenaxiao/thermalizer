
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import re


# In[2]:


file = "total.csv"


# In[5]:


df = pd.read_csv(file, sep='\t')
print(df.shape)
df.head()


# In[6]:


df_unduplicate = df.drop_duplicates("Organism", keep="first")
print(list(df_unduplicate.columns.values))
df_unduplicate = df_unduplicate.drop(["Unnamed: 0", "Txid", "Source", "Features", "OGT"], axis=1)
df_unduplicate.head()


# In[7]:


df_unduplicate.shape


# In[61]:


df_thermal = pd.DataFrame(columns=["organism", "thermal"])
df_meso = pd.DataFrame(columns=["organism", "thermal"])
for index, row in df_unduplicate.iterrows():
    thermal = re.match(r"^thermo.*$", str(row.Temp), flags=re.IGNORECASE)
    if thermal != None:
        df_thermal.loc[index, "organism"] = row["Organism"]
        df_thermal.loc[index, "thermal"] = row["Temp"]
    else:
        df_meso.loc[index, "organism"] = row["Organism"]
        df_meso.loc[index, "thermal"] = row["Temp"]


# In[62]:


print("thermalphile: ", df_thermal.shape)
print("mesophile: ", df_meso.shape)


# In[63]:


df_thermal.to_csv("thermalphile.csv")
df_meso.to_csv("mesophile.csv")

