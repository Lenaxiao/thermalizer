from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
import pandas as pd
import os.path

thermal = "16S_thermophile.fasta"
meso = "16S_mesophile.fasta"
output = "16S_blast.xml"
_new = False

if _new:
	# 16S gene blast
	NcbiblastnCommandline(query=thermal, subject=meso, outfmt=5, out=output)()[0]
	print("blast finished!")

blast_records = NCBIXML.parse(open(output, "r"))
f = open("16S_blast_org.csv", 'w')
f.write("query_seq,hit_seq,hit_len,identity,score,evalue\n")
for blast_record in blast_records:
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps: 
            if hsp.gaps != 0 and blast_record.query != alignment.hit_def:
                f.write(f"{blast_record.query},{alignment.hit_def},{hsp.align_length},{hsp.identities},{hsp.score},{hsp.expect}\n")
f.close()