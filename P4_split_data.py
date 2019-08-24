
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


df = pd.read_csv("16S_seq_dic_temp.tsv", sep="\t", index_col=0)
df.head()


# In[6]:


thermal=df[df.GR=="thermophilic"]
meso=df[df.GR=="mesophilic"]
print(thermal.shape)
print(meso.shape)


# In[8]:


def split_meso_thermal(df, filename):
    f = open(filename, "w")
    for index, rows in df.iterrows():
        f.write(f">{rows.locus}\n{rows.sequence}\n")
    f.close()


# In[9]:


split_meso_thermal(thermal, "16S_thermophile.fasta")
split_meso_thermal(meso, "16S_mesophile.fasta")