import os
import sys
import json
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv
from exa_py import Exa
from utils import *

"""
Augmented LLM Script with Function Calling

This script implements the architecture shown in the diagram:
- Input (User Query) -> LLM -> Output (Response)
- With LLM connected to external tools (Exa search in this case)
- Uses Function Calling to determine if search is needed

Flow:
1. User provides a query
2. LLM determines if external information is needed (using function calling)
3. If needed, LLM uses Exa search to retrieve relevant information
4. LLM incorporates the retrieved information into its response
5. Final response is returned to the user
"""

# Load environment variables
load_dotenv()

# Exa API configuration
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")

class ExaSearchTool:
    """Tool for performing web searches using Exa API via the official exa-py package"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            print("Warning: EXA_API_KEY not set. Search functionality will not work.")
        else:
            self.exa_client = Exa(api_key=self.api_key)
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using Exa API via the official exa-py package
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title, url, and text
        """
        if not self.api_key:
            return [{"title": "Error", "url": "", "text": "API key not provided"}]
        
        try:
            # Use the exa-py client for search_and_contents which gets both search results and their contents
            results = self.exa_client.search_and_contents(
                query=query,
                num_results=num_results,
                use_autoprompt=True,
                text=True,
                type="keyword"
            )
            
            # Format results
            formatted_results = []
            for result in results.results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "text": result.text or ""
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error performing search: {e}")
            return [{"title": "Error", "url": "", "text": f"Search failed: {str(e)}"}]

class AugmentedLLM:
    """LLM augmented with external tool access"""
    
    def __init__(self, search_tool: ExaSearchTool, model: str = "gpt-4o", verbose: bool = False):
        self.search_tool = search_tool
        self.model = model
        self.conversation_history = []
        self.verbose = verbose
        self.last_search_results = []
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for inclusion in prompt"""
        if not results:
            return "No results found."
        
        formatted_text = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted_text += f"[{i}] {result['title']}\n"
            formatted_text += f"URL: {result['url']}\n"
            formatted_text += f"Summary: {result['text']}\n\n"
        
        return formatted_text
    
    def _search_decision_and_query(self, query: str) -> tuple:
        """
        Determine if search should be used and generate optimized search query using parallel function calling
        
        Returns:
            tuple: (should_search, optimized_query)
        """
        # Define the tool functions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "perform_search",
                    "description": "Determine if web search is needed to answer the user's query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "should_search": {
                                "type": "boolean",
                                "description": "Whether external search is needed to provide an accurate and up-to-date answer."
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why search is or isn't needed."
                            }
                        },
                        "required": ["should_search", "reasoning"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_search_query",
                    "description": "Generate an optimized search query to find the most relevant information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "optimized_query": {
                                "type": "string",
                                "description": "An optimized version of the user's query crafted for web search."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of how the query was optimized."
                            }
                        },
                        "required": ["optimized_query", "explanation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]
        
        # Prepare the system message
        system_message = """
        You are an AI assistant that performs two functions:
        
        1. DETERMINE if external search is needed based on these criteria:
           - The query asks about current events or recent information
           - The query requests factual information that might change over time
           - The query is about a specific subject you might have limited information about
           
           SEARCH IS NOT NEEDED if:
           - The query is about general knowledge that doesn't change
           - The query is asking for logical reasoning, opinions, or creative content
           - The query is about well-established concepts or definitions
        
        2. OPTIMIZE the search query by:
           - Focusing on the key information needs
           - Removing unnecessary words and phrases
           - Using specific terms that will yield better search results
           - Adding relevant keywords if the original query is ambiguous
        
        Perform BOTH functions for every user query.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        # Call the model to determine if search is needed and to optimize the query
        # Note: Setting parallel_tool_calls=True to allow multiple function calls in parallel
        response = get_chat_completion(
            messages,
            model=self.model,
            tools=tools,
            tool_choice="auto",  # Allow the model to decide which tools to use
            parallel_tool_calls=True  # Enable parallel function calling
        )
        
        # Default values
        should_search = False
        optimized_query = query  # Default to original query
        search_reasoning = ""
        query_explanation = ""
        
        # Parse the tool calls from the response
        tool_calls = getattr(response, 'tool_calls', None)
        if tool_calls and len(tool_calls) > 0:
            for tool_call in tool_calls:
                try:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "perform_search":
                        should_search = function_args.get('should_search', False)
                        search_reasoning = function_args.get('reasoning', "No reasoning provided")
                        
                    elif function_name == "optimize_search_query":
                        optimized_query = function_args.get('optimized_query', query)
                        query_explanation = function_args.get('explanation', "No explanation provided")
                        
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    print(f"Error parsing function call: {e}")
            
            if self.verbose:
                print(f"Search decision: {should_search}, Reasoning: {search_reasoning}")
                print(f"Optimized query: '{optimized_query}', Explanation: {query_explanation}")
        else:
            # If no tool call was made, fallback to assuming no search is needed
            print("No tool calls were made by the model")
        
        return should_search, optimized_query
    
    def _should_use_search(self, query: str) -> bool:
        """Determine if search should be used for this query (legacy method)"""
        should_search, _ = self._search_decision_and_query(query)
        return should_search
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to the user query, using search when appropriate
        
        Args:
            query: User query
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        search_used = False
        search_results = []
        optimized_query = query  # Default to original query
        
        # Add user query to conversation
        self.conversation_history.append({"role": "user", "content": query})
        
        # Determine if search should be used and get optimized query
        should_search, optimized_query = self._search_decision_and_query(query)
        if should_search:
            search_used = True
            print(f"Searching for information about: '{optimized_query}'")
            search_results = self.search_tool.search(optimized_query)
            self.last_search_results = search_results  # Store search results
            search_text = self._format_search_results(search_results)
            
            # Display search results in terminal if verbose mode is on
            if self.verbose:
                print("\nSearch Results:")
                print("-" * 50)
                for i, result in enumerate(search_results, 1):
                    print(f"[{i}] {result['title']}")
                    print(f"URL: {result['url']}")
                    print(f"Summary: {result['text'][:150]}..." if len(result['text']) > 150 else f"Summary: {result['text']}")
                    print("-" * 30)
                print("-" * 50)
            
            # Prepare system message with search results
            system_message = f"""
            You are an AI assistant augmented with the ability to search the web for information.
            For the user's query, you have the following search results that may contain relevant information.
            Use these results to inform your response, and cite sources when appropriate.
            
            {search_text}
            
            IMPORTANT GUIDELINES:
            1. Provide DIRECT and SPECIFIC answers based on the search results - do not just tell the user where they can find information.
            2. Extract precise facts, dates, numbers, and details from the search results.
            3. If a search result contains the exact answer (like a date, time, location, or fact), state it explicitly.
            4. Do NOT respond with phrases like "To find out..." or "You can check..."
            5. If the search results don't contain relevant information, use your own knowledge but be direct.
            6. Cite sources by referencing the search result number when providing information.
            
            Example of a BAD response: "To find out when the Lakers play next, you can check their schedule on the NBA website."
            Example of a GOOD response: "According to [1], the Lakers play next on March 18, 2025 at 7:30 PM against the Phoenix Suns at Crypto.com Arena."
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
        else:
            # Standard prompt without external search results
            print("Answering without external search")
            system_message = """You are a helpful AI assistant. Provide a direct and specific answer to the user's query based on your knowledge.
            
            IMPORTANT GUIDELINES:
            1. Be direct and specific in your answers.
            2. Provide precise facts, dates, numbers, and details when available.
            3. Do NOT respond with phrases like 'To find out...' or 'You can check...'
            4. If you don't know the exact answer, say so clearly but still provide your best response.
            
            Example of a BAD response: "To find out about quantum physics, you can read books on the subject."
            Example of a GOOD response: "Quantum physics is the study of matter and energy at the most fundamental level. It describes how particles like electrons and photons behave in ways that differ from classical physics."""
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
        
        # Generate response
        response = get_chat_completion(messages, model=self.model)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        execution_time = time.time() - start_time
        
        return {
            "query": query,
            "response": response,
            "search_used": search_used,
            "num_search_results": len(search_results) if search_used else 0,
            "execution_time": execution_time
        }

def main():
    """Main function to run the augmented LLM"""
    # Initialize tools
    search_tool = ExaSearchTool(api_key=EXA_API_KEY)
    augmented_llm = AugmentedLLM(search_tool=search_tool, verbose=True)  # Enable verbose mode
    
    print("=" * 50)
    print("Augmented LLM with Exa Search (Function Calling)")
    print("=" * 50)
    print("Type 'exit' to quit")
    print()
    
    while True:
        # Get user query
        query = input("User: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Generate response
        result = augmented_llm.generate_response(query)
        
        # Print response with metadata
        print("\nAssistant:", result["response"])
        print("\n" + "-" * 50)
        print(f"Search used: {result['search_used']}")
        if result['search_used']:
            print(f"Search results: {result['num_search_results']}")
        
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
