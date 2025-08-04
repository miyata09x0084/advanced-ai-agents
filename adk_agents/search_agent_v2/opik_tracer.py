import json
import os
from opik import Opik
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from typing import Optional, Dict
from opik import context_storage, datetime_helpers
from datetime import datetime

client = Opik(project_name="search_agent_v2")
dataset = client.get_dataset(name="search_agent_v2")

class OpikTracer:
    def __init__(self, project_name, agent_name="Agent", root_agent_name="ReportWriter"):
        self.project_name = project_name
        self.client = Opik(project_name=project_name)
        self.agent_name = agent_name  # Store agent_name for reinitialization
        self.root_agent_name = root_agent_name
        self.trace = None # Initialize trace to None, create on first input
        self.active_callbacks = {}
        self.callback_count = 0
        self.trace_ended = True # Start conceptually ended, wait for input
        self.last_user_input = None
        self.current_trace_user_input = None
        self.current_trace_tool_calls = []
        self.last_dataset_entry = None  # Attribute to store the last dataset entry
        self.eval_mode = False # set eval mode to allow automatic evaluation

    def _reset_trace_state(self):
        """Resets the tracking state for a new trace."""
        self.active_callbacks = {}
        self.callback_count = 0
        self.trace_ended = False # Mark as active once started
        self.current_trace_user_input = None
        self.current_trace_tool_calls = []
        self.last_dataset_entry = None  # Reset last dataset entry

    def before_agent_callback(self, callback_context: CallbackContext):
        
        # Extract user text from the Content object
        user_text = ""
        if callback_context.user_content:
            # Navigate through the Content object structure to get the text
            try:
                # Access the text from the first part of the user content
                if hasattr(callback_context.user_content, 'parts') and callback_context.user_content.parts:
                    for part in callback_context.user_content.parts:
                        if hasattr(part, 'text') and part.text:
                            user_text = part.text
                            break
            except Exception as e:
                print(f"Error extracting user text: {e}")
        
        # Handle trace initialization for both interactive and automated runs
        if callback_context.agent_name == self.root_agent_name and user_text:
            # If no trace exists yet for this interaction cycle, create one
            if self.trace is None:
                print(f"First root agent input detected: {user_text[:50]}... Initializing new trace.")
                self._reset_trace_state() # Reset state variables

                # Create the trace object
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                trace_name = f"{self.agent_name}-{timestamp}"
                try:
                    self.trace = self.client.trace(
                        name=trace_name,
                        end_time=datetime_helpers.local_timestamp(), # May not be needed, but Opik examples use it
                    )
                    print(f"Initialized Opik trace: {trace_name}")
                    # Capture the initial user input immediately after trace creation
                    self.current_trace_user_input = user_text
                    print(f"Captured initial trace input: {self.current_trace_user_input[:50]}...")
                    
                    # Initialize dataset entry structure early
                    # This ensures it exists even if callbacks complete out of expected order
                    self.last_dataset_entry = {
                        "reference": "",  # Will be populated in after_agent_callback
                        "input": self.current_trace_user_input,
                        "expected_tool_use": self.current_trace_tool_calls,
                        "expected_intermediate_agent_responses": []
                    }
                except Exception as e:
                    print(f"Error creating Opik trace: {e}")
                    self.trace = None # Ensure trace is None if creation failed
                    self.trace_ended = True # Revert state if creation failed
                    return None # Stop processing if trace couldn't be created

            # Keep last_user_input for compatibility
            self.last_user_input = user_text
        
        # --- Capture tool calls (only if trace exists) ---
        elif self.trace and callback_context.agent_name != self.root_agent_name and user_text:
            tool_call_data = {
                "tool_name": callback_context.agent_name,
                "tool_input": {
                    "request": user_text  # Assuming user_text is the relevant input
                }
            }
            self.current_trace_tool_calls.append(tool_call_data)
            # Update the dataset entry with the latest tool calls
            if self.last_dataset_entry:
                self.last_dataset_entry["expected_tool_use"] = self.current_trace_tool_calls
            print(f"Captured tool call: {callback_context.agent_name} with input: {user_text[:50]}...")
        # --- END Capture tool calls ---
        
        # Increment callback counter and track this agent callback (only if trace exists)
        if self.trace:
            self.callback_count += 1
            callback_id = self.callback_count
            self.active_callbacks[callback_id] = callback_context.agent_name
            
            # create a trace span (only if trace exists)
            self.trace.span(
                name="Start of " + callback_context.agent_name,
                input={"text": callback_context.user_content},
            )

        return None

    def after_agent_callback(self, callback_context: CallbackContext):
        # Only process if a trace exists
        if not self.trace:
            return None
            
        print("\n" + callback_context.agent_name + " finished executing!\n")
        
        # Assuming callback_context is your object
        invocation_context = callback_context._invocation_context
        session = invocation_context.session
        events = session.events

        # To access the text from a specific event (e.g., the last event)
        content = None
        if events:
            last_event = events[-1]
            content = last_event.content
    
        # Check if there are parts with text
        output_text = ""
        if content and content.parts:
            for part in content.parts:
                # Handle different part types if necessary, focusing on text
                if hasattr(part, 'text') and part.text:
                    print(f"Final response event text: {part.text}")
                    output_text = part.text
                    break
                # Potentially handle function_call or function_response if needed for 'reference'
                elif hasattr(part, 'function_response') and part.function_response:
                     # Example: Try to serialize or get a string representation
                     try:
                         output_text = json.dumps(part.function_response) # Or a more readable format
                         print(f"Event function_response: {output_text}")
                     except Exception:
                         output_text = str(part.function_response)
                     break
                     
        # Always update the reference in last_dataset_entry when we have output text
        # This ensures it's available for both interactive and automated runs
        if output_text and self.last_dataset_entry is not None:
            self.last_dataset_entry["reference"] = output_text

        # create a trace span (only if trace exists)
        if self.trace:
            self.trace.span(
                name="End of " + callback_context.agent_name,
                input={"text": callback_context.user_content},
                output={"text": output_text}, # Use the extracted output_text
            )
        
        # Find the callback ID for this agent and remove it from active callbacks
        callback_id_to_remove = None
        for callback_id, agent_name in self.active_callbacks.items():
            if agent_name == callback_context.agent_name:
                callback_id_to_remove = callback_id
                break
                
        if callback_id_to_remove:
            del self.active_callbacks[callback_id_to_remove]
            
        # Check if this is the last callback and it's from the root agent
        if not self.active_callbacks and callback_context.agent_name == self.root_agent_name:
            print(f"Trace completed for {self.root_agent_name}.")

            # --- Prompt to save dataset (only if trace existed) ---
            if self.current_trace_user_input: # Only prompt if we have initial input
                
                # Update the final dataset entry with the latest information
                # We already initialized this structure in before_agent_callback
                # Now we're just ensuring all fields are up to date
                self.last_dataset_entry = {
                    "reference": output_text,
                    "input": self.current_trace_user_input,
                    "expected_tool_use": self.current_trace_tool_calls,
                    "expected_intermediate_agent_responses": [] # Keep empty as requested
                }

                # Skip saving to dataset or file if eval_mode is True
                if not self.eval_mode:
                    # save to file first
                    dataset_filename = "search_agent_v2/opik_dataset.json"
                    try:
                        data = []
                        if os.path.exists(dataset_filename):
                                # Check if file is empty before trying to load
                            if os.path.getsize(dataset_filename) > 0:
                                with open(dataset_filename, 'r') as f:
                                    try:
                                        data = json.load(f)
                                        # Ensure data is a list
                                        if not isinstance(data, list):
                                            print(f"Warning: Existing data in {dataset_filename} is not a list. Overwriting with new list.")
                                            data = []
                                    except json.JSONDecodeError:
                                        print(f"Warning: Could not decode JSON from {dataset_filename}. Starting fresh.")
                                        data = []
                            else:
                                print(f"{dataset_filename} is empty. Initializing new list.")
                        else:
                            print(f"{dataset_filename} not found. Creating new file.")


                        data.append(self.last_dataset_entry)

                        with open(dataset_filename, 'w') as f:
                            json.dump(data, f, indent=2)
                        print(f"Trace data appended to {dataset_filename}")

                    except Exception as e:
                        print(f"Error saving trace data to {dataset_filename}: {e}")
                    
                    # now prompt to save to opik
                    save_choice = input("Save this trace as an Opik dataset entry? (y/n): ").lower()
                    if save_choice == 'y':
                        # add to opik first
                        try:
                            dataset.insert([self.last_dataset_entry])
                            print(f"Trace data added to Opik dataset: {self.last_dataset_entry}")
                        except Exception as e:
                            print(f"Error adding trace data to Opik dataset: {e}")
                    else:
                        print("Trace data not saved.")
                else:
                    print("Eval mode is enabled, skipping dataset and file saving.")
            else:
                print("No initial user input captured for this trace, skipping save prompt.")
            # --- END Prompt to save dataset ---

            # End the Opik trace regardless of saving choice
            print(f"Ending Opik trace.")
            if self.trace: # Check again before ending
                try:
                    self.trace.end()
                except Exception as e:
                    print(f"Error ending Opik trace: {e}")
                finally:
                    # Mark the trace as ended and clear the trace object for the next cycle
                    self.trace_ended = True
                    self.trace = None # Set trace to None AFTER ending

        return None

    def before_model_callback(self, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        if not self.trace: return None # Guard: Only run if trace is active
        print("Started model call")
        
        # Inspect the last user message in the request contents
        last_user_message = ""
        if llm_request.contents and llm_request.contents[-1].role == 'user':
            if llm_request.contents[-1].parts:
                last_user_message = llm_request.contents[-1].parts[0].text
        
        # create a llm trace span (only if trace exists)
        if self.trace:
            self.trace.span(
                name=llm_request.model,
                type="llm",
                input={"text": last_user_message},
            )

        return None

    def after_model_callback(self, callback_context: CallbackContext, llm_response: LlmResponse):
        if not self.trace: return None # Guard: Only run if trace is active
        print("Finished model call\n")
        
        # create a llm trace span (only if trace exists)
        if self.trace:
            self.trace.span(
                name="response",
                output={"text": llm_response.content},
            )

        return None