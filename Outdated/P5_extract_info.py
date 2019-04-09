import os
from Bio import SeqIO
from Bio import SearchIO

hmmin="Pfam_table.txt"

f=open("Pfam_table2.csv", "w")
f.write("protein_id\tfamily\n")
print("processing...")

for qresult in SearchIO.parse(hmmin, 'hmmer3-tab'):
    cut = 0.01
    for item in qresult.hits:
        '''
        val = item.evalue
        if val < cut:
        '''
        f.write(f"{qresult.id}\t{item.id}\n")
f.close()
