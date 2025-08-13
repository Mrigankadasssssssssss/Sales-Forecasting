import os
from dotenv import load_dotenv
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set.")

def get_data_agent(df: pd.DataFrame):
    """Creates a LangChain Pandas agent for chatting with sales data."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY, temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    return agent
