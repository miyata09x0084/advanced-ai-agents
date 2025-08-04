from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools import google_search
from google.adk.runners import Runner

# tools
web_searcher = LlmAgent(model="gemini-2.0-flash", name="WebSearch", description="Performs web searches on the web for facts.", tools=[google_search])
summarizer = LlmAgent(model="gemini-2.0-flash", name="Summarizer", description="Summarizes text obtained from web searches.")

# search assistant
search_assistant = LlmAgent(
    name="SearchAssistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="Finds and summarizes information on a topic. Use the WebSearch tool to search the web and the Summarizer tool to summarize text obtained from web searches.",
    tools=[agent_tool.AgentTool(agent=web_searcher), agent_tool.AgentTool(agent=summarizer)]
)

# search planning
search_planner = LlmAgent(
    name="SearchPlanner",
    model="gemini-2.5-pro-exp-03-25",
    instruction="You are a helpful assistant. You will plan a search strategy.",
    description="Plans a search strategy. Use the SearchAssistant to find and summarize information."
)

# report writer
report_writer = LlmAgent(
    name="ReportWriter",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. You will write a report on topic X. Use the SearchPlanner to plan and refine the search strategy. Use the SearchAssistant to find and summarize information based on the search strategy. Output *only* the final report.",
    description="Writes a report based on information gathered from the SearchPlanner.",
    tools=[agent_tool.AgentTool(agent=search_planner), agent_tool.AgentTool(agent=search_assistant)]
)

root_agent = report_writer

# test
# Can you give me a report on the latest model by OpenAI : GPT-4.1 Please include the cost, the capabilities, how it differs from gpt-4o, and more important dev details.


