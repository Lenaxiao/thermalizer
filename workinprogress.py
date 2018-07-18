from Bio import SeqIO
import os
import pandas as pd
import numpy as np

# Specify data location
homedir = "/work/data/refseq"

file_count = 0
for root, dirs, files in os.walk(homedir):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".gbff"):
            file_count = file_count + 1

current_file = 0
for root, dirs, files in os.walk(homedir):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".gbff"):
            current_file = current_file + 1
            if current_file % 10 == 0:
                #print("{0:.2f}% complete ({} of {} files)".format((current_file / file_count) * 100, current_file, file_count))
                print("{}% complete ({} of {} files)".format((current_file / file_count) * 100, current_file, file_count))
            for seq_record in SeqIO.parse(filepath, "genbank"):
                organism = seq_record.annotations["organism"]
                print(f"organism = {organism}")
                #for annotation in seq_record.annotations:
                #    print(f"{annotation} = {seq_record.annotations[annotation]}")
                if os.path.isfile(root+'/'+seq_record.id+'.csv'):
                    #print('file ', seq_record.id+'.csv exists!', sep='')
                    pass
                else:
                    #print(seq_record.id)
                    #print(repr(seq_record.seq))
                    #print(len(seq_record))

                    protein_table = pd.DataFrame(columns=['organism', 'locus', 'protein_id', 'product','translation'])
                    #organism = 
                    locus = seq_record.id
                    for feature in seq_record.features:
                        df_temp = pd.DataFrame(columns=['organism', 'locus', 'protein_id', 'product','translation'])
                        df_temp['organism'] = organism
                        df_temp['locus'] = locus
                        if feature.type == 'CDS':
                            #for qual in feature.qualifiers:
                                #print(f"{qual} = {feature.qualifiers[qual]}")
                            if 'product' in feature.qualifiers:
                                df_temp['product'] = feature.qualifiers['product']
                            else:
                                df_temp['product'] = np.nan
                            if 'protein_id' in feature.qualifiers:
                                df_temp['protein_id'] = feature.qualifiers['protein_id']
                            else:
                                df_temp['protein_id'] = np.nan
                            if 'translation' in feature.qualifiers:
                                df_temp['translation'] = feature.qualifiers['translation']
                            else:
                                df_temp['translation'] = np.nan
                            protein_table = protein_table.append(df_temp, ignore_index=True)
                    print(protein_table.shape)
                    print('file saved to ',root,'/',seq_record.id,'.csv', sep='')
                    protein_table.to_csv(root+'/'+seq_record.id+'.csv', index=False)


'''
for seq_record in SeqIO.parse(homedir+"GCF_000006175.1/GCF_000006175.1_ASM617v2_genomic.gbff", "genbank"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))
    #for annotation in seq_record.annotations:
    #    print(f"{annotation} = {seq_record.annotations[annotation]}")
    protein_table = pd.DataFrame(columns=['protein_id', 'product','translation'])
    
    for feature in seq_record.features:
        df_temp = pd.DataFrame(columns=['protein_id', 'product','translation'])
        if feature.type == 'CDS':
            #for qual in feature.qualifiers:
                #print(f"{qual} = {feature.qualifiers[qual]}")
            if 'product' in feature.qualifiers:
                df_temp['product'] = feature.qualifiers['product']
            else:
            	df_temp['product'] = np.nan
            if 'protein_id' in feature.qualifiers:
                df_temp['protein_id'] = feature.qualifiers['protein_id']
            else:
                df_temp['protein_id'] = np.nan
            if 'translation' in feature.qualifiers:
                df_temp['translation'] = feature.qualifiers['translation']
            else:
                df_temp['translation'] = np.nan
            protein_table = protein_table.append(df_temp, ignore_index=True)
    print(protein_table.shape)
    protein_table.to_csv(homedir+"GCF_000006175.1/drafttable1.csv")

'''
