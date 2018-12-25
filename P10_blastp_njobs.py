
# coding: utf-8

# In[ ]:


from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
import pandas as pd
import os
from joblib import Parallel, delayed
import multiprocessing


# In[ ]:


meso_file = "best_hit_org/hit_meso.csv"
thermal_file = "best_hit_org/query_thermal.csv"
meso_fold = "meso_protein/"
thermal_fold = "thermal_protein/"
meso_fst_fold = "meso_fasta/"
thermal_fst_fold = "thermal_fasta/"
blast_dir = "blastp_result/"


# In[ ]:


if not os.path.exists(thermal_fst_fold):
    os.mkdir(thermal_fst_fold)
if not os.path.exists(meso_fst_fold):
    os.mkdir(meso_fst_fold)
if not os.path.exists(blast_dir):
    os.mkdir(blast_dir)


# In[ ]:


def read_txt(fname):
    with open(fname, 'r') as f:
        content = [line.rstrip() for line in f]
        return content


# In[ ]:


meso = read_txt(meso_file)
thermal = read_txt(thermal_file)


# In[ ]:


def csv_to_fasta(file, to_file):
    f = open(file, 'r').readlines()
    fst = open(to_file, 'w')
    
    for i in range(len(f)):
        lines = f[i].split("\t")
        fst.write(f">{lines[3]}\n{lines[-1]}")
    fst.close()


# In[ ]:


def blastp(i, meso, thermal, meso_file, thermal_file, meso_fold,
           thermal_fold, meso_fst_fold, thermal_fst_fold, blast_dir):
    
    if i == 0:
        print("Start:")
    elif i % 10 == 0:
        print("{0:.1f}% Completed!".format(i / len(meso) * 100))
    elif i == len(meso):
        print("100% Completed!")
        
    meso_pro = meso_fold + meso[i] + ".csv"
    thermal_pro = thermal_fold + thermal[i] + ".csv"
    
    meso_fst = meso_fst_fold + meso[i] + ".fasta"
    thermal_fst = thermal_fst_fold + thermal[i] + ".fasta"
    
    filename = blast_dir + meso[i] + "-" + thermal[i]
    
    if os.path.isfile(meso_pro) and not os.path.exists(meso_fst):
        csv_to_fasta(meso_pro, meso_fst)

    if os.path.isfile(thermal_pro) and not os.path.exists(thermal_fst):
        csv_to_fasta(thermal_pro, thermal_fst)
    
    if os.path.exists(meso_fst) and os.path.exists(thermal_fst) and not os.path.exists(filename+".csv"):
        # protein blast
        NcbiblastpCommandline(query=thermal_fst, subject=meso_fst, outfmt=5, out=filename+".xml")()[0]
        blast_records = NCBIXML.parse(open(filename+".xml", "r"))
        f = open(filename+".csv", 'w')
        f.write("query_seq,hit_seq,hit_len,identity,score,evalue\n")
        for blast_record in blast_records:
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps: 
                    if hsp.gaps != 0:
                        f.write(f"{blast_record.query},{alignment.hit_def},{hsp.align_length},{hsp.identities},{hsp.score},{hsp.expect}\n")
        f.close()
        os.remove(filename + ".xml")


# In[ ]:


inputs = range(len(meso))
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores-5)(delayed(blastp)(i, meso, thermal,meso_file, thermal_file, meso_fold, thermal_fold, meso_fst_fold, thermal_fst_fold, blast_dir) for i in inputs)

