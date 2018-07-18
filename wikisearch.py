import urllib.request
from bs4 import BeautifulSoup
#from fuzzywuzzy import process

URL = "https://en.wikipedia.org/wiki/Escherichia_coli"
page = urllib.request.urlopen(URL).read()
print("fetched URL")
soup = BeautifulSoup(page, "html.parser")

#besthit = process.extractOne("°C", page)
#print(besthit)

text = page.decode('utf-8')
loc = text.find("°C")
t = text[loc-8:loc+3]
t.replace("&#160;", "")
print(t)

#print(soup.find_all("table"))
