import urllib.request
from bs4 import BeautifulSoup
import re
#from fuzzywuzzy import process

URL = "https://en.wikipedia.org/wiki/Escherichia_coli"
page = urllib.request.urlopen(URL).read()
print("fetched URL")
soup = BeautifulSoup(page, "html.parser")

#besthit = process.extractOne("°C", page)
#print(besthit)

text = page.decode('utf-8')
### loc = text.find("°C")
### t = text[loc-8:loc+3]
### print(t.replace("&#160;", ""))

tem = re.findall(r'(\d?\d\d)&#160;°C',text)
if tem:
	print(tem)
else:
	print('nothing match')

#print(soup.find_all("table"))
