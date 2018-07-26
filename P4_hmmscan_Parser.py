import os
from datetime import datetime

from Bio import SeqIO
from Bio import SearchIO
import numpy as np
import pandas as pd

hmmin = "Pfam_table.txt"

for qresult in SearchIO.parse(hmmin, 'hmmer3-tab'):
    for item in qresult.hits:
        print(item)
        print(item.evalue) # this will print the evalue of each domain hit

stop_time = datetime.now()
total_time = stop_time - start_time
print("run time was:", total_time)
