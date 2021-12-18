import streamlit as st

from preprocessing_page import load_preprocessing
from eda_page import loadEDA
from contentbasedconsinesimilarities_page import loadContentBasedConsineSimilarities
from contentbasedgensim_page import loadContentBasedGenSim
from collaborativefiltering_recommender_page import loadCollaborativeFiltering_Recommender

st.markdown(
    """
        <style>
        /* The . with the boxed represents that it is a class */
        .boxed {
            background: lightgrey;
            color: black;
            border: 3px solid black;
            margin: 0px auto;
            width: 800px;
            padding: 10px;
            border-radius: 10px;
        }
        ol.s {list-style-type: square;}
        </style>
    """, unsafe_allow_html=True
)


page = st.sidebar.selectbox(
    "Menu", ("Preprocessing", "EDA", "ContentBased ConsineSimilarities", "ContentBased GenSim", "CollaborativeFiltering Recommender"))

if (page == "Preprocessing"):
    load_preprocessing()
elif (page == "EDA"):
    loadEDA()
elif (page == "ContentBased ConsineSimilarities"):
    loadContentBasedConsineSimilarities()

elif (page == "ContentBased GenSim"):
    loadContentBasedGenSim()

else:
    # loadCollaborativeFiltering_Recommender()
    st.write("TBD")
