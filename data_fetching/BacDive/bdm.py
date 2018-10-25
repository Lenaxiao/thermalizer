#! /usr/bin/env python

import requests
from requests.auth import HTTPBasicAuth
import csv
#import yaml


class BacdiveClient(object):
    def __init__(self, credentials):
        self.headers = {'Accept': 'application/json'}
        USERNAME = credentials['login']
        PASSWORD = credentials['password']
        self.credentials = HTTPBasicAuth(USERNAME, PASSWORD)

    def getLinksByGenus(self, genus):
        response = requests.get(
            'http://bacdive.dsmz.de/api/bacdive/taxon/%s/' % (genus ),
            headers=self.headers,
            auth=self.credentials
        )
        if response.status_code == 200:
            results = response.json()
            return results

    def getLinksBySpecies(self, genus, species_epithet):
        response = requests.get(
            'http://bacdive.dsmz.de/api/bacdive/taxon/%s/%s/' % (genus,species_epithet),
            headers=self.headers,
            auth=self.credentials
        )
        if response.status_code == 200:
            results = response.json()
            return results

    def getLinksBySubspecies(self, genus, species_epithet, subspecies_epithet):
        response = requests.get(
            'http://bacdive.dsmz.de/api/bacdive/taxon/%s/%s/%s/' % (genus,species_epithet,subspecies_epithet),
            headers=self.headers,
            auth=self.credentials
        )
        if response.status_code == 200:
            results = response.json()
            return results

    def getLinksBySeqAccNum(self, seq_acc_num):          
        response = requests.get(
            'http://bacdive.dsmz.de/api/bacdive/sequence/%s/' % (seq_acc_num),
            headers=self.headers,
            auth=self.credentials
        )
        if response.status_code == 200:
            results = response.json()
            return results

    def getDataFromURL(self, url):
        response = requests.get(url, headers=self.headers, 
                                auth=self.credentials)
        if response.status_code == 200:
            results = response.json()
            return results

    
    def run(self):
        f = open('Bacdive.csv', 'w')
        f1 =  open('genus_table.csv', 'r')
        reader = csv.reader(f1)
        genus_list = list(reader)[0]
        count = 0
        for genus_name in genus_list:
            genus = self.getLinksByGenus(genus_name) 
            #print(f'genus is {genus}\n')
            if genus == None:
            	continue
            for result in genus['results']:
                url = result['url']
                summary = self.getDataFromURL(url)
                genus = None
                species = None
                subspecies = None
                tax_id = None
                gt = None
                gr = None
                gt_min = None
                gt_max = None
                if 'molecular_biology' in summary:
                    mb = summary['molecular_biology']
                    if 'sequence' in mb:
                        sq = mb['sequence'][0]
                        if 'NCBI_tax_ID' in sq:
                            tax_id = sq['NCBI_tax_ID']
                if 'taxonomy_name' in summary:
                    tax = summary['taxonomy_name']
                    if 'strains_tax_PNU' in tax:
                        taxlin = tax['strains_tax_PNU'][0]  #so that taxlin is dic
                        #print(f'taxlin is {taxlin}\n')      #shows the difference
                        #print(f"taxlin['species_epithet'] is {taxlin['species_epithet']}\n")
                        if 'genus' in taxlin:
                            genus = taxlin['genus']
                        if 'species_epithet' in taxlin:
                            species = taxlin['species_epithet']
                        if 'subspecies_epithet' in taxlin:
                            subspecies = taxlin['subspecies_epithet']
                if 'culture_growth_condition' in summary:
                    cgc = summary['culture_growth_condition'];
                    if 'culture_temp' in cgc:
                        ct = cgc['culture_temp']
                        if 'temp' in ct[0]:   #the first one always growth temperature
                            gt = ct[0]['temp']
                        if 'temperature_range' in ct[0]:
                            gr = ct[0]['temperature_range']
                        for i in range(len(ct)):
                            if ct[i]['test_type'] == 'maximum':
                                gt_max = ct[i]['temp']
                            if ct[i]['test_type'] == 'minimum':
                                gt_min = ct[i]['temp']
                f.write(str("{},{},{},{},{},{},{},{},{}\n".format(genus, species, subspecies, tax_id, gt, gt_min, gt_max, gr,url)))
                f.flush()
            count += 1
            print("{0:.2f}'%' completed".format(count/len(genus_list)*100))
        f.close()
#        species = self.getLinksBySpecies('Methylomonas','methanica')
#        print(species)
#        subspecies = self.getLinksBySubspecies('Bacillus','subtilis', 'subtilis')
#        print(subspecies)
#        sec_acc_num = self.getLinksBySeqAccNum('ALAS01000001')
#        print(sec_acc_num)

if __name__ == '__main__':
    #stream = open('credentials.yaml', 'r')
    #credentials = yaml.load(stream)
    credentials = {'login':'zhangt04@uw.edu','password':'0000000p'}
    BacdiveClient(credentials).run()
