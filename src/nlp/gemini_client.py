from dotenv import load_dotenv
load_dotenv()  # Load variables from .env if present

import os
import google.generativeai as genai

# Load API key from environment
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set. Please set it in your .env file or as an environment variable.")

# Configure Gemini SDK
genai.configure(api_key=API_KEY)

# Pick model (can switch to gemini-pro for better quality)
MODEL_NAME = "gemini-2.0-flash-001"

def generate_insights(category: str, kpis: dict, forecast_points: list):
    """
    Generates AI-written sales insights for a category.

    Args:
        category (str): Product category name
        kpis (dict): Key metrics { 'total_revenue':..., 'total_profit':..., 'avg_monthly_revenue':... }
        forecast_points (list): List of dicts [{'month': 'YYYY-MM', 'forecast_revenue':..., 'forecast_profit':...}, ...]

    Returns:
        str: AI-generated insight
    """
    prompt = f"""
    You are a sales analyst AI. Analyze the following data for the category '{category}'.

    **Key Metrics**
    Total Revenue: {kpis['total_revenue']:,}
    Total Profit: {kpis['total_profit']:,}
    Avg Monthly Revenue: {kpis['avg_monthly_revenue']:,}

    **Forecast**
    {forecast_points}

    Please write a concise, business-friendly summary (max 150 words) 
    highlighting trends, risks, and opportunities. 
    Focus on actionable insights.
    """

    try:
        response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Could not generate insights: {e}"
