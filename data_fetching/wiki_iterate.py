import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import re

# take in organism data from csv file
df = pd.read_csv('org_table.csv',sep='\t',header=-1)
df.columns=['organism']
df.drop(3594,inplace=True)   #list index error here
result_list = []
# iterate through the organism names and the correspoding URL
for i in range(len(df)):
    tem_list = df.iloc[i][0].split()
    URL = "https://en.wikipedia.org/wiki/{}_{}".format(tem_list[0],tem_list[1])
    #print(URL)

    # avoid HTTPError exception upon url request that terminate the loop
    try:
        page = urllib.request.urlopen(URL).read()
    except :
        #result_list.append('404')
        tem='404ERROR'
    else:
        #soup = BeautifulSoup(page, "html.parser")
        text = page.decode('utf-8')
        tem = re.findall(r'(\d?\d\d)&#160;°C',text)
    if tem:
        result_list.append(tem)
    else:
        result_list.append(0)  
    print("{0:.2f}'%' completed".format(i/len(df)*100))

#in result_list '404' represent no wikipage, '0' represent no '°C' , (37,45) represent the searching result
print(result_list)

with open("temp.txt", "w") as output:
    output.write(str(result_list))

if len(result_list)==len(df):
    print('right length!')
    df['tem'] = result_list

df.to_csv('org_temp.csv', sep='\t', encoding='utf-8')