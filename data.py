import requests
from bs4 import BeautifulSoup
import io
import re

def get_poems():
    url = "https://en.wikipedia.org/wiki/List_of_Emily_Dickinson_poems"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    poem_list_div = soup.find("div", class_="mw-parser-output")
    links = poem_list_div.find_all("a")

    links_poems = []
    for link in links:
        href = link.get("href")
        links_poems.append(href)
        # print(href)
    list_poems = links_poems[13:-53]

    with io.open("poems.txt", "w", encoding="utf-8") as file:
        for url in list_poems:
            try:
                poem_response = requests.get(url)
                soup = BeautifulSoup(poem_response.content, "html.parser")
                poem_div = soup.find("div", {"class": "poem"})
                poem_txt = poem_div.get_text("/n")

                file.write(poem_txt + "/n/n")
                print("poem text saved")
            except AttributeError:
                print("Error: poem content not found for URL:", url)
    print("Done")


if __name__ == "__main__":
    get_poems()


# Open file and read data
with open("poems.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------------
# Cleans the text from line breaks
pattern = r"/n|/n/n"
new_text = re.sub(pattern, "", text)

# Write the new text to a file
with open("poems_clean.txt", "w", encoding="utf-8") as f:
    f.write(new_text)
