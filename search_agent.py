"""
Search Agent: An orchestrator-workers workflow for complex search tasks.

This module implements a search agent that breaks down complex search queries into
subtasks, delegates them to specialized worker LLMs, and synthesizes the results.

Architecture:
1. Orchestrator: Analyzes query, creates subtasks
2. Workers: Execute specialized searches
3. Synthesizer: Integrates findings into a coherent response

Usage example:
    python search_agent.py "What's the latest from OpenAI"
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from dotenv import load_dotenv
from exa_py import Exa
from utils import get_chat_completion, get_structured_output_with_tools, get_current_datetime_tool_func

# Load environment variables
load_dotenv()

# API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")

# Default models
ORCHESTRATOR_MODEL = "gpt-4o"
WORKER_MODEL = "gpt-4o"
SYNTHESIZER_MODEL = "gpt-4o"

class SearchSubtask(BaseModel):
    """Represents a search subtask created by the orchestrator."""
    id: str
    query: str
    source_type: str  # e.g., "web", "news", "academic", "specialized"
    time_period: Optional[str] = None  # e.g., "recent", "past_year", "all_time"
    domain_focus: Optional[str] = None  # e.g., "technology", "science", "health"
    priority: int  # 1 (highest) to 5 (lowest)

class SubtaskListResponse(BaseModel):
    """Pydantic model to wrap the list of subtasks."""
    subtasks: List[SearchSubtask]

class SearchResult(BaseModel):
    """Represents the result of a search subtask."""
    subtask_id: str
    source_type: str
    results: List[Dict[str, Any]]
    summary: str
    execution_time: float

class SearchWorker:
    """Worker responsible for executing a specific search subtask."""
    
    def __init__(self, api_key: str, model: str = WORKER_MODEL):
        self.exa_client = Exa(api_key=api_key) if api_key else None
        self.model = model
        
    def _determine_category(self, query: str) -> Optional[str]:
        """Dynamically determine the appropriate category based on query content.
        
        Available categories: "news", "research paper", "pdf", "company", 
                             "github", "tweet", "personal site", "linkedin profile"
        """
        query_lower = query.lower()
        
        # Check for research/academic indicators
        if any(term in query_lower for term in ["research", "study", "paper", "academic", "journal", "publication"]):
            return "research paper"
            
        # Check for news indicators
        elif any(term in query_lower for term in ["news", "recent", "latest", "update", "report", "announcement"]):
            return "news"
            
        # Check for company information
        elif any(term in query_lower for term in ["company", "business", "corporation", "enterprise", "startup"]):
            return "company"
            
        # Check for code/technical content
        elif any(term in query_lower for term in ["code", "github", "repository", "programming", "algorithm", "implementation"]):
            return "github"
            
        # Check for social media content
        elif any(term in query_lower for term in ["tweet", "twitter", "social media"]):
            return "tweet"
            
        # Check for personal blog/site content
        elif any(term in query_lower for term in ["blog", "personal", "portfolio"]):
            return "personal site"
            
        # Check for professional profile content
        elif any(term in query_lower for term in ["linkedin", "profile", "resume", "cv", "professional", "career"]):
            return "linkedin profile"
            
        # Check for document content
        elif any(term in query_lower for term in ["pdf", "document", "whitepaper", "paper", "report"]):
            return "pdf"
            
        # No specific category detected
        return None
        
    def execute(self, subtask: SearchSubtask) -> SearchResult:
        """Execute a search subtask and return structured results."""
        start_time = time.time()
        
        # Customize search parameters based on subtask properties
        num_results = 8 if subtask.priority <= 2 else 5
        
        # Execute search based on source type
        if subtask.source_type == "web":
            raw_results = self._search_web(subtask.query, num_results)
        elif subtask.source_type == "news":
            raw_results = self._search_news(subtask.query, num_results)
        elif subtask.source_type == "academic":
            raw_results = self._search_academic(subtask.query, num_results)
        else:  # Default to web search
            raw_results = self._search_web(subtask.query, num_results)
            
        # Create summary of results using LLM
        summary = self._create_summary(subtask, raw_results)
        
        execution_time = time.time() - start_time
        
        return SearchResult(
            subtask_id=subtask.id,
            source_type=subtask.source_type,
            results=raw_results,
            summary=summary,
            execution_time=execution_time
        )
    
    def _search_web(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Execute a general web search with dynamic category selection."""
        if not self.exa_client:
            return [{"title": "Error", "url": "", "text": "API key not provided"}]
        
        # Dynamically determine category based on query content
        category = self._determine_category(query)
        print(f"Web search using category: {category if category else 'None'}")
        
        try:
            # Prepare API parameters, only including category if it's not None
            search_params = {
                "query": query,
                "num_results": num_results,
                "use_autoprompt": True,
                "text": True,
                "type": "keyword"
            }
            
            if category:
                search_params["category"] = category
                
            # Execute search with the configured parameters
            results = self.exa_client.search_and_contents(**search_params)
            
            formatted_results = []
            for result in results.results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "text": result.text or "",
                    "source": "web",
                    "category": category
                })
                
            return formatted_results
        except Exception as e:
            print(f"Error in web search: {e}")
            return [{"title": "Error", "url": "", "text": f"Search failed: {str(e)}", "source": "web"}]
    
    def _search_news(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Execute a news search focusing on recent content."""
        if not self.exa_client:
            return [{"title": "Error", "url": "", "text": "API key not provided"}]
        
        try:
            # Modify the query to focus on news content
            news_query = f"recent news about {query}"
            
            # News search with category parameter
            results = self.exa_client.search_and_contents(
                query=news_query,
                num_results=num_results,
                use_autoprompt=True,
                text=True,
                type="keyword",
                category="news"  # Specifically target news content
            )
            
            formatted_results = []
            for result in results.results:
                # Try to extract date if available
                date_info = None
                if hasattr(result, 'published_date'):
                    date_info = result.published_date
                
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "text": result.text or "",
                    "source": "news",
                    "date": date_info
                })
                
            return formatted_results
        except Exception as e:
            print(f"Error in news search: {e}")
            return [{"title": "Error", "url": "", "text": f"News search failed: {str(e)}", "source": "news"}]
    
    def _search_academic(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Execute an academic search focusing on research papers and journals."""
        if not self.exa_client:
            return [{"title": "Error", "url": "", "text": "API key not provided"}]
        
        try:
            # For academic searches, modify the query to target academic content
            academic_query = f"research paper academic study {query}"
            
            # Define academic domains to include
            academic_domains = [
                "arxiv.org", 
                "scholar.google.com", 
                "researchgate.net", 
                "academia.edu", 
                "ncbi.nlm.nih.gov", 
                "science.org", 
                "nature.com", 
                "sciencedirect.com", 
                "springer.com", 
                "ieee.org"
            ]
            
            results = self.exa_client.search_and_contents(
                query=academic_query,
                num_results=num_results,
                use_autoprompt=True,
                text=True,
                type="keyword",
                include_domains=academic_domains,
                category="research paper"  # Specify academic content category
            )
            
            formatted_results = []
            for result in results.results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "text": result.text or "",
                    "source": "academic"
                })
                
            return formatted_results
        except Exception as e:
            print(f"Error in academic search: {e}")
            return [{"title": "Error", "url": "", "text": f"Academic search failed: {str(e)}", "source": "academic"}]
    
    def _create_summary(self, subtask: SearchSubtask, results: List[Dict[str, Any]]) -> str:
        """Create a concise summary of the search results using an LLM."""
        if not results or len(results) == 0:
            return "No results found."
            
        # Format results for the LLM
        formatted_results = ""
        for i, result in enumerate(results, 1):
            formatted_results += f"[{i}] {result['title']}\n"
            formatted_results += f"URL: {result['url']}\n"
            # Limit text length to avoid token limits
            formatted_results += f"Content: {result['text'][:1000]}...\n\n" if len(result['text']) > 1000 else f"Content: {result['text']}\n\n"
        
        # Create prompt for the LLM
        system_message = """
        You are an expert research assistant. Your task is to analyze search results and create a concise, 
        informative summary that captures the key information relevant to the search query.
        
        Focus on extracting:
        1. Facts and data points relevant to the query
        2. Different perspectives or viewpoints presented
        3. Key consensus points across multiple sources
        4. Areas of disagreement or uncertainty
        5. Notable quotes or statements from authoritative sources
        
        Organize the information logically and provide proper attribution to sources.
        Be objective and comprehensive, including all relevant information even if it presents different views.
        """
        
        user_message = f"""
        Search Query: {subtask.query}
        Source Type: {subtask.source_type}
        
        Search Results:
        {formatted_results}
        
        Please provide a comprehensive summary of these search results that directly addresses the search query.
        Include specific facts, figures, and quotes when relevant, with source references [1], [2], etc.
        """
        
        try:
            # we are not using structured outputs here because these are dynamic summaries
            response = get_chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model
            )
            
            return response
        except Exception as e:
            print(f"Error creating summary: {e}")
            return f"Error summarizing results: {str(e)}"

class SearchOrchestrator:
    """Orchestrator that breaks down complex queries into subtasks."""
    
    def __init__(self, exa_api_key: str, model: str = ORCHESTRATOR_MODEL):
        self.exa_api_key = exa_api_key
        self.model = model
        self.worker = SearchWorker(api_key=exa_api_key, model=WORKER_MODEL)
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a complex search query and return comprehensive results."""
        start_time = time.time()
        
        print(f"\nProcessing query: {query}")
        print("=" * 50)
        
        # Step 1: Break down the query into subtasks
        subtasks = self._create_subtasks(query)
        print(f"Created {len(subtasks)} search subtasks")
        
        # Step 2: Execute subtasks in parallel
        search_results = self._execute_subtasks(subtasks)
        
        # Step 3: Synthesize results
        synthesizer = ResultSynthesizer(model=SYNTHESIZER_MODEL)
        final_response = synthesizer.synthesize(query, search_results)
        
        # Create the result object
        execution_time = time.time() - start_time
        
        result = {
            "query": query,
            "response": final_response,
            "subtasks": len(subtasks),
            "search_results": [
                {
                    "subtask_id": result.subtask_id,
                    "source_type": result.source_type,
                    "num_results": len(result.results),
                } for result in search_results
            ],
            "execution_time": execution_time
        }
        
        return result
    
    def _create_subtasks(self, query: str) -> List[SearchSubtask]:
        """Use an LLM and the utility function to create subtasks, handling tools and Pydantic parsing."""
        system_message = """
        You are an expert research planner. Your task is to break down a complex research query into 
        specific search subtasks, each focusing on a different aspect or source type.
        
        You have access to a tool to get the current date and time if needed for queries like 'latest news'.
        Use the tool if the user's query implies needing current information.

        For each subtask, provide:
        1. A unique string ID for the subtask (e.g., 'subtask_1', 'news_update')
        2. A specific search query that focuses on one aspect of the main query
        3. The source type to search (web, news, academic, specialized)
        4. Time period relevance (recent, past_year, all_time, or a specific date range using the tool)
        5. Domain focus if applicable (technology, science, health, etc.)
        6. Priority level (1-highest to 5-lowest)
        
        All fields (id, query, source_type, time_period, domain_focus, priority) are required for each subtask, except time_period and domain_focus which can be null if not applicable.
        
        Create 3-5 subtasks that together will provide comprehensive coverage of the topic.
        Focus on different aspects, perspectives, or sources of information.
        
        If you don't need the current date/time tool, respond DIRECTLY with the structured list of subtasks.
        If you DO need the tool, call it first. Then, using the tool's response, generate the final structured list.
        Your final response MUST conform to the Pydantic schema for a list of SearchSubtask objects.
        Specifically, return a JSON object with a single key "subtasks" containing the list: 
        { "subtasks": [ {subtask_1}, {subtask_2}, ... ] }
        """
        
        user_message = f"""
        Main research query: {query}
        
        Break this down into 3-5 specific search subtasks according to the required structure.
        Consider if the current date/time is needed to formulate the best subtasks.
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_datetime",
                    "description": "Get the current date and time (ISO 8601 format), useful for 'latest' or 'recent' queries.",
                    "parameters": { # No parameters needed for this tool
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

        # Define the functions the LLM can call
        available_functions = {
            "get_current_datetime": get_current_datetime_tool_func,
        }

        # Define the expected Pydantic response schema (a list of subtasks)
        try:
            # Use the new utility function
            parsed_subtasks = get_structured_output_with_tools(
                messages=messages,
                response_schema=SubtaskListResponse, # Pass the wrapper schema
                tools=tools,
                available_functions=available_functions,
                model=self.model,
                description="search subtask list"
            )

            if parsed_subtasks and isinstance(parsed_subtasks, SubtaskListResponse): # Check if the utility returned the correct wrapper object
                print(f"--- Orchestrator: Successfully created {len(parsed_subtasks.subtasks)} subtasks via utility ---")
                # Basic validation (optional, Pydantic handles structure)
                if all(isinstance(st, SearchSubtask) for st in parsed_subtasks.subtasks):
                     return parsed_subtasks.subtasks # Return the actual list
                else:
                     print("--- Orchestrator: Utility returned data but not in expected SearchSubtask list format ---")
                     raise ValueError("Invalid data structure returned by utility")
            else:
                # The utility function returned None (error occurred)
                print(f"--- Orchestrator: Utility function failed to return subtasks. Falling back. ---")
                raise ValueError("Utility function returned None")
        
        except Exception as e:
            print(f"--- Orchestrator: Error in _create_subtasks using utility: {e} ---")
            # Fallback to a basic subtask division
            fallback_subtasks = [
                SearchSubtask(
                    id="fallback_general",
                    query=query,
                    source_type="web",
                    priority=1
                ),
                SearchSubtask(
                    id="fallback_news",
                    query=f"recent news about {query}",
                    source_type="news",
                    time_period="recent",
                    priority=2
                ),
                SearchSubtask(
                    id="fallback_academic",
                    query=f"research on {query}",
                    source_type="academic",
                    priority=3
                )
            ]
            print(f"--- Orchestrator: Created {len(fallback_subtasks)} fallback search subtasks due to error ---")
            return fallback_subtasks

    def _execute_subtasks(self, subtasks: List[SearchSubtask]) -> List[SearchResult]:
        """Execute search subtasks, potentially in parallel."""
        results = []
        
        # Sort subtasks by priority
        sorted_subtasks = sorted(subtasks, key=lambda x: x.priority)
        
        # Execute subtasks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(subtasks), 3)) as executor:
            # Submit all tasks
            future_to_subtask = {
                executor.submit(self.worker.execute, subtask): subtask 
                for subtask in sorted_subtasks
            }
            
            # Process results as they complete
            for i, (future, subtask) in enumerate(future_to_subtask.items(), 1):
                try:
                    print(f"Executing subtask {i}/{len(subtasks)}: {subtask.query} [{subtask.source_type}]")
                    result = future.result()
                    results.append(result)
                    print(f"Completed subtask {i}/{len(subtasks)}")
                except Exception as e:
                    print(f"Subtask execution failed: {e}")
                    # Add a placeholder failed result
                    results.append(
                        SearchResult(
                            subtask_id=subtask.id,
                            source_type=subtask.source_type,
                            results=[{"title": "Error", "url": "", "text": f"Task execution failed: {str(e)}"}],
                            summary=f"Error executing subtask: {str(e)}",
                            execution_time=0.0
                        )
                    )
        
        return results

class ResultSynthesizer:
    """Synthesizes results from multiple search subtasks into a coherent response."""
    
    def __init__(self, model: str = SYNTHESIZER_MODEL):
        self.model = model
        
    def synthesize(self, original_query: str, search_results: List[SearchResult]) -> str:
        """Synthesize search results into a coherent response."""
        # Format the search results for the LLM
        formatted_results, source_list = self._format_results_for_synthesis(search_results)
        
        system_message = """
        You are an expert research synthesizer with the ability to integrate information from multiple sources.
        Your task is to create a comprehensive, well-structured response to the original query based on the 
        search results provided.
        
        Follow these guidelines:
        1. Directly address the original query with specific information from the search results
        2. Integrate information from all relevant sources, giving more weight to higher confidence results
        3. Highlight areas of consensus and note areas of disagreement or uncertainty
        4. Provide proper attribution to sources using [Source X] notation in the text
        5. Structure your response logically with clear sections and summaries
        6. Be comprehensive yet concise, focusing on the most relevant information
        7. Present a balanced view if the topic is controversial, showing multiple perspectives
        8. At the end of your response, include a "Sources:" section that lists all sources referenced
           using the format "Source X: [full source name]" with each source on a new line
        
        YOUR RESPONSE SHOULD BE DIRECTLY USEFUL AND INFORMATIVE - DO NOT USE PHRASES LIKE "ACCORDING TO THE SEARCH RESULTS" 
        OR "TO FIND OUT..." - INSTEAD, DIRECTLY PROVIDE THE INFORMATION.
        """
        
        user_message = f"""
        Original Query: {original_query}
        
        Search Results:
        {formatted_results}
        
        Source List (include these at the end of your response):
        {source_list}
        
        Please synthesize these search results into a comprehensive response that directly addresses 
        the original query. Integrate information from all sources, with proper attribution in the text.
        Remember to include the Sources section at the end listing all sources in the format specified.
        """
        
        try:
            response = get_chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model
            )
            
            # If the response doesn't already include a Sources section, add it
            if "Sources:" not in response and "SOURCES:" not in response.upper():
                response += f"\n\nSources:\n{source_list}"
                
            return response
        except Exception as e:
            print(f"Error synthesizing results: {e}")
            return f"Error synthesizing results: {str(e)}"
    
    def _format_results_for_synthesis(self, search_results: List[SearchResult]) -> Tuple[str, str]:
        """Format search results for the synthesis LLM.
        
        Returns:
            A tuple containing (formatted_results, source_list)
        """
        formatted_results = ""
        source_details = []
        
        for i, result in enumerate(search_results, 1):
            # Create a source identifier
            source_id = f"Source {i}"
            source_name = f"{result.source_type.upper()} SEARCH"
            
            # Store source details for the list
            source_detail = f"{source_id}: {source_name}"
            if result.results and len(result.results) > 0:
                top_sources = []
                for j, item in enumerate(result.results[:min(3, len(result.results))], 1):
                    source_url = item['url']
                    source_title = item['title']
                    top_sources.append(f"{source_id}.{j}: {source_title} ({source_url})")
                source_details.extend(top_sources)
            else:
                source_details.append(source_detail)
            
            # Format the main results
            formatted_results += f"{source_id}: {source_name}\n"
            formatted_results += f"Summary:\n{result.summary}\n\n"
            
            # Add specific details from high-confidence results
            if len(result.results) > 0:
                formatted_results += "Key points from individual results:\n"
                for j, item in enumerate(result.results[:3], 1):  # Limit to first 3 for brevity
                    formatted_results += f"  {i}.{j} {item['title']}\n"
                    # Extract a brief snippet
                    snippet = item['text'][:200] + "..." if len(item['text']) > 200 else item['text']
                    formatted_results += f"  URL: {item['url']}\n"
                    formatted_results += f"  Snippet: {snippet}\n\n"
            
            formatted_results += "-" * 40 + "\n\n"
        
        # Create the source list string
        source_list = "\n".join(source_details)
            
        return formatted_results, source_list

def main():
    """Main function to run the search agent."""
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("Usage: python search_agent.py \"your complex search query\"")
        return
    
    # Get the query from command line
    query = " ".join(sys.argv[1:])
    
    # Check for API keys
    if not EXA_API_KEY:
        print("Error: EXA_API_KEY not set in environment variables or .env file")
        return
    
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in environment variables or .env file")
        return
    
    # Initialize the orchestrator
    orchestrator = SearchOrchestrator(exa_api_key=EXA_API_KEY)
    
    # Process the query
    print("\n" + "=" * 50)
    print(f"Processing search query: {query}")
    print("=" * 50)
    
    result = orchestrator.process_query(query)
    
    print("\n" + "=" * 50)
    print("SEARCH RESULTS")
    print("=" * 50)
    print(f"\nQuery: {result['query']}")
    print(f"Number of subtasks: {result['subtasks']}")
    print(f"Execution time: {result['execution_time']:.2f} seconds")
    print("\nResponse:")
    print("-" * 50)
    print(result['response'])
    print("=" * 50)

if __name__ == "__main__":
    main()
