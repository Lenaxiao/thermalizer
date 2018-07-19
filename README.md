# thermalizer
--
### Automated translation from mesophilic AA sequence to thermally stabilized variant
--

Some notes:
* The environment for this tool can be reconstructed with conda.  It is saved in `environment.yml`

Overview of files:

|File|Description|
|...|...|
|P1_TableGenerator.py|Extracts all protein sequences from the refseq bacterial and archeael genomes|
|P2_Table2Fasta|Creates a fasta file with the sequences from P1|
