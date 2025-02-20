import streamlit as st
import pandas as pd
from paper_recc import get_recommendations
from autocorrect import Speller

st.markdown("""
    <style>
    .stTextInput input {
        color: #bfbbbb;
    }
    .stNumberInput input {
        color: #bfbbbb;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Scientific Paper Finder")
st.write("Find top papers from arXiv based on your topic.")

with st.expander(f"Original dataframe"):
        url = "https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts"
        st.write("[Link to original dataframe](%s) (unfiltered for the first 500)" % url)
        st.write(pd.read_csv("filtered_arxiv_data.csv").drop(columns=['Unnamed: 0','terms']))

spell = Speller()
st.header("🔎 Search Settings")
input = st.text_input('What scientific topics do you want to search for?', value='Medical imaging with machine learning')
x = st.number_input('How many top search results do you want to see?', min_value=1, max_value=50, value=5)

st.header("📑 Search Results")
if spell(input) == input:
    st.markdown(f"#### Retrieving {x} results for {input}")
else:
    st.markdown(f"#### Retrieving {x} results for {spell(input)} instead of {input}")
df = get_recommendations(spell(input), x, data_path="filtered_arxiv_data.csv")
for index, row in df.iterrows():
    with st.expander(f"{row['Title']}"):
        st.write(f"{row['Abstract']}")