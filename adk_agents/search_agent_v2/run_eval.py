import os
import json
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.genai import types
from google.adk.sessions import InMemorySessionService
from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import base_metric, score_result
from google import genai
from dotenv import load_dotenv

load_dotenv()

from .agent import root_agent, tracer
import random

# for LLM-as-a-judge
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Warning: GOOGLE_API_KEY not found. LLM-as-a-judge may fail.")
# Initialize client, handling potential failure if API_KEY is missing
try:
    llm_client = genai.Client(api_key=API_KEY) if API_KEY else None
except Exception as e:
    print(f"Error initializing genai.Client: {e}. LLM-as-a-judge may fail.")
    llm_client = None

# Constants
APP_NAME = "search_agent"
USER_ID = "dev_user_01"
SESSION_ID = "pipeline_session_01"
LLM_AS_A_JUDGE_MODEL = "gemini-2.0-flash"

LLM_JUDGE_CRITERIA = """
Evaluate the quality of the "Actual Response" compared to the "Expected Response" based on the following criteria:
1. Correctness: Does the actual response accurately address the user's query or task defined by the "User Query"?
2. Completeness: Does the actual response provide all necessary information that would be expected for the query, similar to what's in the "Expected Response"?
3. Relevance: Is the actual response directly relevant to the query, without unnecessary or hallucinated information?
4. Clarity: Is the actual response clear, concise, and easy to understand?

Based on these criteria, provide a single numerical score from 1 to 10, where 1 is very poor and 10 is excellent.
Respond with ONLY the numerical score (e.g., 7).
"""


def create_session(user_message_content):

    FINAL_SESSION_ID = SESSION_ID + str(random.randint(0, 100))

    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=FINAL_SESSION_ID)

    print("Initializing agent runner...")
    runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)

    # trigger agent
    events = runner.run(user_id=USER_ID, session_id=FINAL_SESSION_ID, new_message=types.Content(role='user', parts=[types.Part(text=user_message_content)]))

    # gather actual tool calls and final result
    tool_calls = []
    result = ""
    for event in events:
        if event.is_final_response():
            print("Reached final event!")
                    
    # actual output is obtained from the tracer
    trace_final_result = tracer.last_dataset_entry["reference"]

    print("====== FINAL RESPONSE ==============")
    print(trace_final_result)
    print("====================================")

    # actual tool calls are obtained from the tracer
    tool_calls = tracer.last_dataset_entry["expected_tool_use"]

    print("====== TOOL CALLS ==============")
    print(tool_calls)
    print("================================")
    return tool_calls, trace_final_result


def get_tool_names(tool_call_data):
    names = set()
    if tool_call_data is None:
        return names

    if isinstance(tool_call_data, str):
        try:
            # Attempt to load if it's a JSON string representation of a list
            tool_call_data = json.loads(tool_call_data)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, and it's a simple string, treat as a single tool name
            # This case might need refinement based on actual string format for single tools
            # For now, if it's a non-JSON string, we'll add it directly if it's non-empty
            if tool_call_data.strip(): # Avoid adding empty strings
                names.add(tool_call_data.strip())
            return names # Return early if it was a simple string (not a list)
    
    if not isinstance(tool_call_data, list):
        print(f"Warning: tool_call_data is not a list after potential parsing: {type(tool_call_data)}. Data: {str(tool_call_data)[:100]}")
        return names

    for tc in tool_call_data:
        if isinstance(tc, dict) and tc.get("name"):
            names.add(tc["name"])
        elif isinstance(tc, str) and tc.strip(): # Handles if dataset stores tool calls as list of non-empty names
            names.add(tc.strip())
        # Silently ignore malformed entries or log them if verbose debugging is needed
    return names


def evaluate(dataset_item, predicted_tool_calls, predicted_final_response):
    user_input = dataset_item["input"]
    expected_tool_use = dataset_item["expected_tool_use"]
    expected_final_result = dataset_item["reference"]

    llm_score = 0
    tool_match_score = 0.0

    # 1. LLM-as-a-judge
    if llm_client:
        llm_judge_prompt = f"{LLM_JUDGE_CRITERIA}\n\nUser Query:\n{user_input}\n\nExpected Response:\n{expected_final_result}\n\nActual Response:\n{predicted_final_response}\n\nScore (1-10):"
        try:
            response = llm_client.models.generate_content( model=LLM_AS_A_JUDGE_MODEL, contents=llm_judge_prompt)
            llm_score = int(response.text.strip())
            if not (1 <= llm_score <= 10):
                print(f"Warning: LLM score {llm_score} out of range (1-10). Clamping to 0.")
                llm_score = 0 # Or handle as error / clamp to min/max valid
        except Exception as e:
            print(f"Error during LLM-as-a-judge call: {e}")
            llm_score = 0  # Default or error score
    else:
        print("LLM client not available. Skipping LLM-as-a-judge.")

    # 2. Tool Call Match (Jaccard Index for tool names)
    # measures the similarity between the predicted tool calls and the expected tool calls; higher the better
    actual_tool_names = get_tool_names(predicted_tool_calls)
    expected_tool_names_set = get_tool_names(expected_tool_use)

    if not isinstance(actual_tool_names, set):
         actual_tool_names = set(actual_tool_names) # ensure it's a set
    if not isinstance(expected_tool_names_set, set):
        expected_tool_names_set = set(expected_tool_names_set) # ensure it's a set

    intersection_count = len(actual_tool_names.intersection(expected_tool_names_set))
    union_count = len(actual_tool_names.union(expected_tool_names_set))

    if union_count == 0:
        tool_match_score = 1.0 if intersection_count == 0 else 0.0
    else:
        tool_match_score = intersection_count / union_count

    print(f"--- Evaluation for Input: {user_input[:100]}... ---")
    print(f"LLM-as-a-judge Score (1-10): {llm_score}")
    print(f"Tool Match Score (Jaccard): {tool_match_score:.2f}")
    
    return llm_score, tool_match_score


def main():
    load_dotenv() # Load environment variables from .env file

    # configure Opik eval dataset
    opik_client = Opik(project_name="search_agent_v2") # Renamed client to avoid conflict
    dataset = opik_client.get_dataset(name="search_agent_v2")

    # set trace to eval mode so it skip storing traces
    tracer.eval_mode = True
    
    total_llm_score = 0
    total_tool_match_score = 0.0
    num_items_evaluated = 0

    # evaluate for each item in the dataset
    dataset_list = json.loads(dataset.to_json())
    print(f"Starting evaluation for {len(dataset_list)} items...")

    for item in dataset_list:
        print(f"\nProcessing item with input: {item['input'][:100]}...")
        try:
            predicted_tool_calls, predicted_final_response = create_session(item["input"])
            llm_score, tool_match_score = evaluate(item, predicted_tool_calls, predicted_final_response)
            
            total_llm_score += llm_score
            total_tool_match_score += tool_match_score
            num_items_evaluated += 1
        except Exception as e:
            print(f"Error processing dataset item: {item['input'][:100]}... Error: {e}")
            # Optionally, decide if you want to skip this item or assign default error scores

    if num_items_evaluated > 0:
        avg_llm_score = total_llm_score / num_items_evaluated
        avg_tool_match_score = total_tool_match_score / num_items_evaluated
        print("\n--- Overall Evaluation Summary ---")
        print(f"Items Evaluated: {num_items_evaluated}")
        print(f"Average LLM-as-a-judge Score: {avg_llm_score:.2f}/10")
        print(f"Average Tool Match Score (Jaccard): {avg_tool_match_score:.2f}")
    else:
        print("No items were successfully evaluated from the dataset.")
    
if __name__ == "__main__":
    main()