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
