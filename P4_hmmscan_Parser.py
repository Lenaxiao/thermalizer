import os
from datetime import datetime

from Bio import SeqIO
from Bio import SearchIO
import numpy as np
import pandas as pd

hmmin = "Pfam_table.txt"

# find the most significant E-value for every protein.
f = open("HitResult.csv", "w")
f.write("product_id\tEvalue\n")
print("Progressing...")
wrong = 0
for qresult in SearchIO.parse(hmmin, 'hmmer3-tab'):
	wong = wrong + 1
    top = 0.01  # Evalue larger than 0.01 is not statistally significant.
    for item in qresult.hits:
        val = item.evalue
        if val < top:
            top = val
        else:
            pass
    f.write(f"{qresult.id}\t{top}\n")

f.close()
stop_time = datetime.now()
total_time = stop_time - start_time
print("run time was:", total_time)

###### Merge Two Table ##########
'''
import pandas as pd
import numpy as np

file1 = "seq_dic.tsv"
file2 = "HitResult.csv"

df1 = pd.read_csv(file1, sep='\t', header=None)
df2 = pd.read_csv(file2, sep='\t', header=None)

df3 = df1.merge(df2, on='protein_id', how='outer').dropna()
df3.to_csv("seq_dic_merged.csv", seq='\t')
'''