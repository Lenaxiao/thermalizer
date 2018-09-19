#import env
import pandas as pd
import urllib.request
import re

#creat URL
df = pd.read_csv('org_temp_sub.csv',index_col=0,sep=',',header=0)
result_list = []

for i in range(len(df)):
	tem_list = df.iloc[i][0].split()
	join_list = '+'.join(tem_list)
	URL = "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?name={}".format(join_list)

	#visit the website and search with regex
	try:
		page = urllib.request.urlopen(URL).read()
	except :
		#result_list.append('404')
		txid='404ERROR'
	else:
		#soup = BeautifulSoup(page, "html.parser")
		text = page.decode('utf-8')
		txid = re.findall(r'Taxonomy ID: (\d*)',text)
	if txid:
		result_list.append(txid)
	else:
		result_list.append(0)

	#timer fuc
	print("{0:.2f}'%' completed".format(i/len(df)*100))

#in result_list '404' represent no such page on ncbi, '0' represent no txid found , \d* represent the txid
print(result_list)

if len(result_list)==len(df):
    print('right length!')
    df['Txid'] = result_list

df.to_csv('org_txid.csv', sep='\t', encoding='utf-8')