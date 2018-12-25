import os

protein = "seq_dic.tsv"
thermal = "best_hit_org/query_thermal.csv"
meso = "best_hit_org/hit_meso.csv"
thermal_dir = "thermal_protein"
meso_dir = "meso_protein"

makedir = True  # set to False if you don't want to make directory

if makedir and not os.path.isdir(thermal_dir) and not os.path.isdir(meso_dir):
	os.mkdir(thermal_dir)
	os.mkdir(meso_dir)

fp = open(protein, "r")
ft = open(thermal, "r")
fm = open(meso, "r")

thermal_org = ft.read().splitlines()  # thermophiles
meso_org = fm.read().splitlines()  # mesophiles

for index, line in enumerate(fp):
	locus = line.split("\t")[1]  # locus
	if locus in thermal_org:
		filename = thermal_dir+"/"+locus+".csv"
		f = open(filename, "a")
		if os.path.exists(filename):
			f.write(line)
		else:
			f.write("organism\tlocus\tproduct\tproduct_id\ttranslation\n")
		f.close()
	elif locus in meso_org:
		filename = meso_dir+"/"+locus+".csv"
		f = open(filename, "a")
		if os.path.exists(filename):
			f.write(line)
		else:
			f.write("organism\tlocus\tproduct\tproduct_id\ttranslation\n")
		f.close()