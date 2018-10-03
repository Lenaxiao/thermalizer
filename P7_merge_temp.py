import os
import pandas as pd

if not os.path.exists('seq_temp.csv'):
	df_temp = pd.read_csv('data_fetching/resistance.csv').drop(['organism', 'resistance_boo'], axis=1)
	df_seq = pd.read_csv('seq_dic.tsv', sep='\t')
	df_temp_seq = pd.merge(df_temp, df_seq, on='locus', how='inner')
	df_temp_seq.to_csv('seq_temp.csv')
df_temp_seq = pd.read_csv('seq_temp.csv')
print('Data Loaded!')
print('Start:')

for subdir, dirs, files in os.walk('protein_family'):
    for file in files:
        df = pd.read_csv('protein_family/'+file)
        df_final = pd.merge(df_temp_seq, df, on='product_id', how='inner')
        new_dir = 'temped_protein_family'
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        if not df_final.empty:
            df_final.to_csv(new_dir+'/temped_'+file)