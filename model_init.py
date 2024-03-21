from txtai.embeddings import Embeddings
import pandas as pd

def get_data():
    # data = []
    title_data = []
    content_data = []
    link_data = []
    # df = pd.read_csv('https://raw.githubusercontent.com/Anway-Agte/PA-Search-Engine/ai-search/PA_LENR/LENR_metadata_csv.csv', usecols=['document_link', 'abstract', 'title'])
    df = pd.read_csv('https://raw.githubusercontent.com/Anway-Agte/PA-Search-Engine/main/PA_LENR/data%20(2).csv', usecols=['document_link', 'abstract', 'title'], nrows=1000)


    for index, row in df.iterrows():
        abstract = re.sub('[()]', '', str(row['abstract']))
        title = str(index+1) + ". " + re.sub('[()]', '', str(row['title']))
        # data.append({"index": index, "abstract": abstract, "title": title, "link": row['document_link']})
        content_data.append((index, abstract, None))
        link_data.append(row['document_link'])
        title_data.append(title)
    return content_data, link_data, title_data


content_data, link_data, title_data = get_data()
emb_data = []
for row in content_data:
    emb_data.append(row)
embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True, objects=True)
# self.embeddings.index(zip(content_data, link_data, title_data))
embeddings.index(emb_data)
embeddings.save("models")