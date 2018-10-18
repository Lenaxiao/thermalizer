# group by protein families

import pandas as pd

df = pd.read_csv('Pfam_table2.csv', sep='\t')
grouped = df.groupby('family')
for name, group in grouped:
    if group.shape[0] != 1:
        file_name = name+'.csv'
        group.to_csv('protein_family/'+file_name, index=False)