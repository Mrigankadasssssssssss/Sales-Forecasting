import streamlit as st
import pandas as pd
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE))

from src.nlp.data_chat_agent import get_data_agent
from src.data_pipeline import run as run_pipeline

PROC = BASE / "data" / "processed"

st.set_page_config(page_title="💬 Data Chat", layout="wide")
st.title("💬 Chat with Your Sales Data")


proc_file = PROC / "monthly_by_category.csv"
if not proc_file.exists():
    st.warning("⚠️ Processed data not found. Running pipeline to generate it...")
    run_pipeline()

if not proc_file.exists():
    st.error(f"❌ Still no processed file at: {proc_file}")
    st.stop()


df = pd.read_csv(proc_file, parse_dates=["ds"])
agent = get_data_agent(df)

user_query = st.chat_input("Ask me about sales trends, profits, categories...")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    try:
        response = agent.run(user_query)
        st.session_state.chat_history.append({"role": "assistant", "content": str(response)})
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
