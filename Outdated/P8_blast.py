from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
import pandas as pd

blast_dir = "blast_result"
if not os.path.exists("blast_result"):
    os.mkdir(blast_dir)

for subdirs, dirs, files in os.walk('temped_protein_family')
    for file in files:

        # combine rows have same protein
        df_ori = pd.read_csv(file)
        foo = lambda x: "/".join(x)
        df = df_ori.groupby(by=['product_id', 'translation']).agg({'resistance': foo}).reset_index()

        # csv --> fasta
        f = open(blast_dir+"/"+tofile, "w")
        for index, row in df.iterrows():
            seq = row['translation']
            f.write(f">{row['product_id']} {row['resistance']}\n{row['translation']}\n")
        f.close()


        # blast
        thresh = 0.001  # UNDEFINED
        NcbiblastpCommandline(query=tofile, subject=tofile, outfmt=5, out=filename+".xml")()[0]
        blast_records = NCBIXML.parse(open(filename+".xml", "r"))

        new_f = open(new_file, 'w')
        new_f.write("query_seq,hit_seq,score,evalue,query,match\n")
        for blast_record in blast_records:
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps: 
                    if hsp.gaps != 0:
                        new_f.write(f"{blast_record.query},{alignment.hit_def},{hsp.score},{hsp.expect},{hsp.query},{hsp.match}\n")