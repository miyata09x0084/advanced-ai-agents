from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Single Agent
# Takes the initial specification (from user query) and answers in one sentence.
single_agent = LlmAgent(
    name="SingleAgent",
    model=GEMINI_MODEL,
    instruction="""You are a helpful assistant.
    You will answer the user's query in one sentence.
    """,
    description="A single agent that answers user queries in one sentence.",
)

root_agent = single_agent