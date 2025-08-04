# Start with the Quick Start Guide

Go over important details regarding setting up environment, API Keys, etc. 
https://google.github.io/adk-docs/get-started/quickstart/

# Demo Simple Agent with ADK

run agent: adk run adk_agents/single_agent

# Conversational Context

Go through what is session, state, and memory

- Session tracks individual conversation threads (e.g. identification, events, temp data); short-term memory
- State (the session's scratchpad) is where agents store and update dynamic details during a conversation; short-term memory
- Memory enables agent to recall information from past conversations or access external KBs; long-term memory

https://google.github.io/adk-docs/sessions/

# Demo Sequential Agent with ADK

Introduce important concepts and build a quick multi-agent system:

run agent: adk run adk_agents/sequential_agent


# Demo Search Agent with ADK

run agent: adk run adk_agents/search_agent

