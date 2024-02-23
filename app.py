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
        content_data.append(abstract)
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
        self.embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
        # self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

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
        content_data, link_data, title_data = get_data()
        # content_data, link_data, title_data = self.get_data_from_es()



        st.title("Semantic Based Similarity Search")
        st.markdown("This application runs a basic similarity search that identifies the best matching row for a query.")
        st.text_area("Source Papers", value='\n'.join(title_data), key="source_papers_1")
        query = st.text_input("Query")

        if query:
            # Get index of best section that best matches query
            # top 5 results:
            top_list = self.embeddings.similarity(query, content_data)[:5]
            for uid, score in top_list:
                st.write("Abstract: ", content_data[uid], "\n Score = ", score, "\n", link_data[uid])


        st.title("Keyword Based Similarity Search")
        st.markdown("This application runs a basic similarity search that identifies the best matching row for a query based on keywords.")
        st.text_area("Source Papers", value='\n'.join(title_data), key="source_papers_2")
        keyword = st.text_input("Enter a keyword for search", key="keyword_input")        

        if st.button("Find Similar Documents"):

            if keyword:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(content_data)
                keyword_vector = vectorizer.transform([keyword])
                cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
                top_5_idx = np.argsort(cosine_similarities)[-5:]
                top_5_sorted_idx = sorted(top_5_idx, key=lambda i: cosine_similarities[i], reverse=True)

                st.write("Top 5 similar documents (by index):")
                for idx in top_5_sorted_idx:
                    st.write("Abstract: ", content_data[idx], "\n Score = ", cosine_similarities[idx], "\n", link_data[idx])
            
            else:
                st.warning("Please enter the query or keyword for search")

        # data = get_data()
        # Second text area
        # Keyword input
        # content_data = [row['abstract'] for row in data]
        # link_data = [row['link'] for row in data]
        # Process search
        # if st.button("Find Similar Documents"):
            





@st.cache_resource(ttl=60 * 60, max_entries=3, show_spinner=False)
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application

    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create and run application
    app = create()
    app.run()