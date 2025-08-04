from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.agents.callback_context import CallbackContext
import os
from dotenv import load_dotenv
from .opik_tracer import OpikTracer

load_dotenv()

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_PRO_MODEL = "gemini-2.5-pro-preview-05-06"
WEB_SEARCHER_AGENT_NAME = "WebSearch"
SUMMARIZER_AGENT_NAME = "Summarizer"
SEARCH_ASSISTANT_AGENT_NAME = "SearchAssistant"
SEARCH_PLANNER_AGENT_NAME = "SearchPlanner"
ROOT_AGENT_NAME = "ReportWriter"

# for observability
tracer = OpikTracer(project_name="search_agent_v2", agent_name="SearchAgent", root_agent_name=ROOT_AGENT_NAME)

# tools
web_searcher = LlmAgent(
    model=GEMINI_MODEL, 
    name=WEB_SEARCHER_AGENT_NAME, 
    description="Performs web searches on the web for facts.", 
    tools=[google_search],
    before_agent_callback=tracer.before_agent_callback,
    after_agent_callback=tracer.after_agent_callback,
    before_model_callback=tracer.before_model_callback,
    after_model_callback=tracer.after_model_callback,
)
summarizer = LlmAgent(
    model=GEMINI_MODEL, 
    name=SUMMARIZER_AGENT_NAME, 
    description="Summarizes text obtained from web searches.",
    before_agent_callback=tracer.before_agent_callback,
    after_agent_callback=tracer.after_agent_callback,
    before_model_callback=tracer.before_model_callback,
    after_model_callback=tracer.after_model_callback,
)

# search assistant
search_assistant = LlmAgent(
    name=SEARCH_ASSISTANT_AGENT_NAME,
    model=GEMINI_MODEL,
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="Finds and summarizes information on a topic. Use the WebSearch tool to search the web and the Summarizer tool to summarize text obtained from web searches.",
    tools=[agent_tool.AgentTool(agent=web_searcher), agent_tool.AgentTool(agent=summarizer)],
    before_agent_callback=tracer.before_agent_callback,
    after_agent_callback=tracer.after_agent_callback,
    before_model_callback=tracer.before_model_callback,
    after_model_callback=tracer.after_model_callback,
)

# search planning
search_planner = LlmAgent(
    name=SEARCH_PLANNER_AGENT_NAME,
    model=GEMINI_PRO_MODEL,
    instruction="You are a helpful assistant. You will plan a search strategy.",
    description="Plans a search strategy. Use the SearchAssistant to find and summarize information.",
    before_agent_callback=tracer.before_agent_callback,
    after_agent_callback=tracer.after_agent_callback,
    before_model_callback=tracer.before_model_callback,
    after_model_callback=tracer.after_model_callback,
)

root_agent = LlmAgent(
    name=ROOT_AGENT_NAME,
    model=GEMINI_MODEL,
    instruction="You are a helpful assistant. You will write a report on topic X. Use the SearchPlanner to plan and refine the search strategy. Use the SearchAssistant to find and summarize information based on the search strategy. Output *only* the final report.",
    description="Writes a report based on information gathered from the SearchPlanner.",
    tools=[agent_tool.AgentTool(agent=search_planner), agent_tool.AgentTool(agent=search_assistant)],
    before_agent_callback=tracer.before_agent_callback,
    after_agent_callback=tracer.after_agent_callback,
    before_model_callback=tracer.before_model_callback,
    after_model_callback=tracer.after_model_callback,
)

# test
# Can you give me a report on the latest model by OpenAI : GPT-4.1 Please include the cost, the capabilities, how it differs from gpt-4o, and more important dev details.


