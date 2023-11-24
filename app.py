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
    df = pd.read_csv('https://raw.githubusercontent.com/Anway-Agte/PA-Search-Engine/ai-search/PA_LENR/LENR_metadata_csv.csv', usecols=['document_link', 'abstract', 'title'])

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
        
        # data = ['Isoperibolic electrode calorimetry has demonstrated that four times as much heat is generated at the anode then at the cathode in D2O. Experiments recognized that silica affected some results. Experiments in K2CO3 reported here identifies silica as both a contributor to excess heat generation and as a factor in modifying the cell calibration constant. Implications for cold fusion will be discussed.',
        #         'In any discussion of the origin, measurement or description of the anomalous power prod ucing process which occurs in connection with the electrochemical loading of deuterium into palladium, knowledge of the thermodynamic behaviour of the system is clearly of importance. More particularly, since the formation of highly l oaded palladium is implicated as a necessary (but itself insufficient) condition for the observation of anomalous power, thermodynamic considerations relating to the attainment of high l oadings are of interest. Here, it is intended to review, at a general level , those aspects of the thermodynamic nature of the system, both equilibrium and non-equilibrium, which appear to bear most directly,on the question of excess power producti on in relati o n to the attai nment of high loadings.',
        #         'Measurement of electrical resistance is a convenient method for the determination of composition in a number of metal?hydrogen systems. For the fi-phase of the H?Pd system, pertinent data from the literature are employed in order to construct a complete resistance-loading function at 298 K.',
        #         'U.S. House of Representatives, Hearing before the Committee on Science, Space and Technology on cold fusion, April 1989.',
        #         'An experimental system has been developed to grow pure titanium films on tungsten substrates. The physicochemical properties of these films have been widely studied and ad hoc samples can be used for Cold Fusion experiments avoiding their contact with atmosphere. Different Cold Fusion experiments are proposed in a new experimental setup that allows deuterium gas loading of the film whi le electrical current is applied through them. Thus, an experimental configuration similar to an electrochemical loading is attained.',
        #         'A complete set of NRS Nuclear Reactions in Solids experiments has been performed on the Ti-D system checking as triggering mechanisms of these phenomena the imposition of electric fields and the crossing of the cS-E and p-cS phase boundaries. The experiments were accomplished using a high pure iodide-titanium film as the initial metal matrix. Neutron measurements were monitored while doing these experiments and no clear evidence of the nuclear fusion reaction D+D---+3He+n has been detected, the upper detection limit for this reaction being lamda = 3 x 10^-21 f/pds.',
        #         'Abstract. -- To study the electron screening of nuclear reactions in metallic environments, angular distributions and thick target yields of the fusion reactions  3He have been measured on deuterons implanted in three different metal targets for beam energies ranging from 5 to 60 keV. The experimentally determined values of the screening energy are about one order of magnitude larger than the value achieved in a gas target experiment and significantly larger than the theoretical predictions. A clear target material dependence of the screening energy has been established.',
        #         ]

        # data = st.text_area("Data", value="\n".join(data))
        st.text_area("Source Papers", value='\n'.join(row['title'] for row in data))
        query = st.text_input("Query")

        # data = data.split("\n")
        content_data = [row['abstract'] for row in data]
        link_data = [row['link'] for row in data]

        if query:
            # Get index of best section that best matches query
            # top 5 results:
            for i in range(5):
                uid, score = self.embeddings.similarity(query, content_data)[i]
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