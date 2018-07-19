import os
from datetime import datetime

from Bio import SeqIO
import numpy as np
import pandas as pd

# Specify data location
homedir = "/work/data/refseq/"

start_time = datetime.now()

file_count = 0
for root, dirs, files in os.walk(homedir):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".gbff"):
            file_count = file_count + 1

current_file = 0
f = open('seq_dic.csv', 'w')
f.write("organism,locus,product_id,product,translation\n")
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
                for feature in seq_record.features:
                    if feature.type == 'CDS':
                        f.write(f"{organism},{seq_record.id},{feature.qualifiers['product'][0] if 'product' in feature.qualifiers else np.nan},{feature.qualifiers['protein_id'][0] if 'protein_id' in feature.qualifiers else np.nan},{feature.qualifiers['translation'][0] if 'translation' in feature.qualifiers else np.nan}\n")
                        # commented out the line below to speed up code
                        # f.flush()
f.close()

stop_time = datetime.now()
total_time = stop_time - start_time
print("run time was:", total_time)
