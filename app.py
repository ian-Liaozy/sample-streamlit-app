"""
Basic similarity search example. Used in the original txtai demo.

Requires streamlit to be installed.
  pip install streamlit
"""

import os

import streamlit as st

from txtai.embeddings import Embeddings

# import urllib3
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# from elasticsearch import Elasticsearch


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

def get_pdf_data():
    pdf_data = []
    


class Application:
    """
    Main application.
    """

    def __init__(self):
        """
        Creates a new application.
        """

        # Create embeddings model, backed by sentence-transformers & transformers
        # self.embeddings = Embeddings()
        self.init = 0
        # self.content_data = []
        # self.link_data = []
        # self.title_data = []
        # self.es = Elasticsearch("http://localhost:9200")

    # def get_data_from_es(self):
    #     # Query Elasticsearch to get documents
    #     response = self.es.search(index="papers_index", body={"query": {"match_all": {}}})
    #     titles, abstracts, links = [], [], []
    #     for hit in response['hits']['hits']:
    #         source = hit["_source"]
    #         titles.append(source["title"])
    #         abstracts.append(source["abstract"])
    #         links.append(source["link"])
    #     return abstracts, links, titles


    def run(self):
        """
        Runs a Streamlit application.
        """
        # content_data = ["US tops 5 million confirmed virus cases",
        #     "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        #     "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        #     "The National Park Service warns against sacrificing slower friends in a bear attack",
        #     "Maine man wins $1M from $25 lottery ticket",
        #     "Make huge profits without work, earn up to $100,000 a day"]
        # link_data = ["Link 1", "Link 2", "Link 3", "Link 4", "Link 5", "Link 6"]
        # title_data = ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5", "Title 6"]
        # emb_data = []
        # for idx, content in enumerate(content_data):
        #     emb_data.append((idx, content, None))
        # self.embeddings.index(emb_data)
        # if self.init == 0:
        #     content_data, link_data, title_data = get_data()
        #     self.embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True, objects=True)
        #     # self.embeddings.index(zip(content_data, link_data, title_data))
        #     self.embeddings.index(emb_data)
        #     self.embeddings.save("index")
        #     self.init = 1
        # else:
        #     self.embeddings.load("index")
        # # content_data, link_data, title_data = self.get_data_from_es()
        content_data, link_data, title_data = get_data()
        # # self.embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True, objects=True)
        # # self.embeddings.index(zip(content_data, link_data, title_data))
        
        
        # self.embeddings.load("models")
        embeddings = Embeddings()
        embeddings.load(provider="huggingface-hub", container="1anliao/lenr-paper-semantic-search")
            
        # self.embeddings.save("index")

        st.title("Semantic Based Similarity Search")
        st.markdown("This application runs a basic similarity search that identifies the best matching row for a query.")
        st.text_area("Source Papers", value='\n'.join(title_data), key="source_papers_1")
        keyword = st.text_input("Query")
        # keyword = st.text_input("Enter a keyword for search", key="keyword_input")
        semantic_search = st.button("Semantic Search")
        # keyword_search = st.button("Keyword Search")
        # if semantic_search:
        if keyword:
            # Get index of best section that best matches query
            # top 5 results:
            # top_list = self.embeddings.similarity(query, content_data)[:5]
            top_list = embeddings.search(keyword, 5)
            for idx, result in enumerate(top_list):
                st.write("Title: ", title_data[int(result["id"])], "\n Abstract: ", result["text"], "\n Score = ", result["score"], "\n", link_data[int(result["id"])])
        else:
            st.warning("Please enter the query or keyword for search")


        # st.title("Keyword Based Similarity Search")
        # st.markdown("This application runs a basic similarity search that identifies the best matching row for a query based on keywords.")
        # st.text_area("Source Papers", value='\n'.join(self.title_data), key="source_papers_2")
                

        # if keyword_search:
        #     if keyword:
        #         vectorizer = TfidfVectorizer(stop_words='english')
        #         tfidf_matrix = vectorizer.fit_transform(self.content_data)
        #         keyword_vector = vectorizer.transform([keyword])
        #         cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
        #         top_5_idx = np.argsort(cosine_similarities)[-5:]
        #         top_5_sorted_idx = sorted(top_5_idx, key=lambda i: cosine_similarities[i], reverse=True)

        #         st.write("Top 5 similar documents (by index):")
        #         for idx in top_5_sorted_idx:
        #             st.write("Abstract: ", self.content_data[idx], "\n Score = ", cosine_similarities[idx], "\n", self.link_data[idx])
            
        #     else:
        #         st.warning("Please enter the query or keyword for search")

        # data = get_data()
        # Second text area
        # Keyword input
        # content_data = [row['abstract'] for row in data]
        # link_data = [row['link'] for row in data]
        # Process search
        # if st.button("Find Similar Documents"):
            



def model_init():
    # content_data = ["US tops 5 million confirmed virus cases",
    #         "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
    #         "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
    #         "The National Park Service warns against sacrificing slower friends in a bear attack",
    #         "Maine man wins $1M from $25 lottery ticket",
    #         "Make huge profits without work, earn up to $100,000 a day"]
    # link_data = ["Link 1", "Link 2", "Link 3", "Link 4", "Link 5", "Link 6"]
    # title_data = ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5", "Title 6"]
    content_data, link_data, title_data = get_data()
    emb_data = []
    for row in content_data:
        emb_data.append(row)
    embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True, objects=True)
    # self.embeddings.index(zip(content_data, link_data, title_data))
    embeddings.index(emb_data)
    embeddings.save("models")
    return content_data, link_data, title_data

def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application

    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # content_data, link_data, title_data = model_init()
    # Create and run application
    app = create()
    # app.content_data = content_data
    # app.link_data = link_data
    # app.title_data = title_data
    app.run()