from Bio import SeqIO
import os
import pandas as pd
import numpy as np

# Specify data location
homedir = "/work/data/refseq/"

file_count = 0
for root, dirs, files in os.walk(homedir):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".gbff"):
            file_count = file_count + 1

current_file = 0
f = open('seq_dic.csv', 'w')
f.write(f"'organism','locus','product_id','product','translation'\n")
print("Mission start:")
for root, dirs, files in os.walk(homedir):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".gbff"):
            current_file += 1
            if current_file % 100 == 0:
                print("{:.2%} complete ({} of {} files)".format(current_file/file_count, current_file, file_count))
            for seq_record in SeqIO.parse(filepath, "genbank"):
                organism = seq_record.annotations["organism"]
                #if os.path.isfile(root+'/'+seq_record.id+'.csv'):
                    #print('file ', seq_record.id+'.csv exists!')
                    #pass
                #else:
                #protein_table = pd.DataFrame(columns=['organism','locus','protein_id', 'product','translation'])
                for feature in seq_record.features:
                    #df_temp = pd.DataFrame(columns=['organism','locus','protein_id', 'product','translation'])
                    if feature.type == 'CDS':
                        #df_temp['locus'] = seq_record.id
                        #df_temp['organism'] = organism
                        #for qual in feature.qualifiers:
                            #print(f"{qual} = {feature.qualifiers[qual]}")
                        #if 'product' in feature.qualifiers:
                            #df_temp['product'] = feature.qualifiers['product']
                        #else:
                            #df_temp['product'] = np.nan
                        #if 'protein_id' in feature.qualifiers:
                            #df_temp['protein_id'] = feature.qualifiers['protein_id']
                        #else:
                            #df_temp['protein_id'] = np.nan
                        #if 'translation' in feature.qualifiers:
                            #df_temp['translation'] = feature.qualifiers['translation']
                        #else:
                            #df_temp['translation'] = np.nan
                        f.write(f"{organism},{seq_record.id},")
                        f.write(f"{feature.qualifiers['product'] if 'product' in feature.qualifiers else np.nan},")
                        f.write(f"{feature.qualifiers['protein_id'] if 'protein_id' in feature.qualifiers else np.nan},")
                        f.write(f"{feature.qualifiers['translation'] if 'translation' in feature.qualifiers else np.nan}\n")
                        f.flush()
                        #protein_table = protein_table.append(df_temp, ignore_index=True)
                    #protein_table.to_csv(root+'/'+seq_record.id+'.csv', index=False)
f.close()

