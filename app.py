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


def get_data():
    data = []
    # df = pd.read_csv('https://raw.githubusercontent.com/Anway-Agte/PA-Search-Engine/ai-search/PA_LENR/LENR_metadata_csv.csv', usecols=['document_link', 'abstract', 'title'])
    df = pd.read_csv('https://raw.githubusercontent.com/Anway-Agte/PA-Search-Engine/main/PA_LENR/data%20(1).csv', usecols=['document_link', 'abstract', 'title'])
    for index, row in df.iterrows():
        abstract = re.sub('[()]', '', str(row['abstract']))
        title = str(index+1) + ". " + re.sub('[()]', '', str(row['title']))
        data.append({"index": index, "abstract": abstract, "title": title, "link": row['document_link']})
    return data

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

    def run(self):
        """
        Runs a Streamlit application.
        """

        st.title("Similarity Search")
        st.markdown("This application runs a basic similarity search that identifies the best matching row for a query.")

        # data = [
        #     "US tops 5 million confirmed virus cases",
        #     "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        #     "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        #     "The National Park Service warns against sacrificing slower friends in a bear attack",
        #     "Maine man wins $1M from $25 lottery ticket",
        #     "Make huge profits without work, earn up to $100,000 a day",
        # ]
        data = get_data()

        # data = st.text_area("Data", value="\n".join(data))
        st.text_area("Source Papers", value='\n'.join(row['title'] for row in data))
        query = st.text_input("Query")

        # data = data.split("\n")
        content_data = [row['abstract'] for row in data]
        link_data = [row['link'] for row in data]

        if query:
            # Get index of best section that best matches query
            # top 5 results:
            top_list = self.embeddings.similarity(query, content_data)[:5]
            for uid, score in top_list:
                st.write("Abstract: ", content_data[uid], "\n Score = ", score, "\n", link_data[uid])


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