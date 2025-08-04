#!/usr/bin/env python3
"""
Marketing Copywriter - A sequential multi-agent system for course marketing

This script implements a sequential workflow with three specialized agents:
1. Search Agent: Gathers information from a course structure file
2. Writer Agent: Converts search results into engaging marketing copy
3. Editor Agent: Edits the marketing copy for grammar and conciseness

The agents operate in sequence, with each agent's output feeding into the next.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from utils import *
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configurations
DEFAULT_MODEL = "gpt-4o"

# Path to course structure file
COURSE_STRUCTURE_FILE = os.path.join(os.path.dirname(__file__), "data/course_structure.md")


# Pydantic models for structured output
class Topic(BaseModel):
    name: str = Field(description="The name or title of the topic")
    description: str = Field(description="A detailed description of what the topic covers")
    subtopics: List[str] = Field(description="List of subtopics covered in this topic", default_factory=list)

class CourseInfoSchema(BaseModel):
    course_title: str = Field(description="The title of the course")
    course_description: List[str] = Field(description="Lines of course description", default_factory=list)
    topics: List[Topic] = Field(description="List of topics covered in the course", default_factory=list)
    target_audience: List[str] = Field(description="Types of audience this course is suitable for", default_factory=list)

class SearchResult(BaseModel):
    """Class for storing search results from the SearchAgent."""
    course_title: str
    course_description: List[str]
    topics: List[Dict[str, Any]]
    target_audience: List[str]
    execution_time: float

class MarketingCopy(BaseModel):
    """Class for storing marketing copy from the WriterAgent."""
    tweet: str = Field(description="A tweet-length marketing message for the course.")
    execution_time: float

class EditedMarketingCopy(BaseModel):
    """Class for storing edited marketing copy from the EditorAgent."""
    tweet: str = Field(description="The final, edited tweet-length marketing message for the course.")
    execution_time: float


class SearchAgent:
    """Agent responsible for gathering information about a course from a structure file."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """Initialize the search agent."""
        self.model = model
    
    def read_course_structure(self, file_path: str) -> str:
        """Read course structure from the provided file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading course structure file: {e}")
            return f"Error: {str(e)}"
    
    def search(self, course_structure: str) -> SearchResult:
        """
        Search and analyze the course structure.
        
        Args:
            course_structure: Content of the course structure file
            
        Returns:
            SearchResult object containing course information
        """
        start_time = time.time()
        
        system_message = """
        You are a search agent that will analyze course structure and extract detailed information.
        
        Your tasks:
        1. Gather all the information including topics about the course
        2. Expand on each topic with detailed descriptions
        3. Figure out a specific target audience based on the course content
        
        Provide a comprehensive analysis of the course.
        
        Format your response as a valid JSON object with the following structure:
        {
            "course_title": "Title of the course",
            "course_description": ["Line 1 of description", "Line 2 of description", ...],
            "topics": [
                {
                    "name": "Topic name",
                    "description": "Detailed description",
                    "subtopics": ["Subtopic 1", "Subtopic 2", ...]
                },
                ...
            ],
            "target_audience": ["Audience type 1", "Audience type 2", ...]
        }
        """
        
        user_message = f"""
        Please analyze the following course structure and provide detailed information:
        
        {course_structure}
        """
        
        try:
            # Use the get_structured_output function from utils
            result = get_structured_output(
                query=user_message,
                system_message=system_message,
                response_schema=CourseInfoSchema,
                description="course information"
            )
            
            # Convert Pydantic model to SearchResult
            if result:
                search_result = SearchResult(
                    course_title=result.course_title,
                    course_description=result.course_description,
                    topics=[topic.model_dump() for topic in result.topics],  # Convert Pydantic Topic objects to dicts
                    target_audience=result.target_audience,
                    execution_time=time.time() - start_time
                )
                return search_result
            else:
                # Handle the case where structured output failed
                raise ValueError("Failed to extract structured information from course content")
            
        except Exception as e:
            print(f"Error in search process: {e}")
            return SearchResult(
                course_title="Error",
                course_description=[f"Error analyzing course structure: {str(e)}"],
                topics=[],
                target_audience=[],
                execution_time=time.time() - start_time
            )


class WriterAgent:
    """Agent responsible for writing marketing copy based on search results."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """Initialize the writer agent."""
        self.model = model
    
    def write(self, search_result: SearchResult) -> MarketingCopy:
        """
        Generate marketing copy based on search results.
        
        Args:
            search_result: SearchResult object containing course information
            
        Returns:
            MarketingCopy object containing the generated tweet
        """
        start_time = time.time()
        
        system_message = """
        You are a technical marketing manager whose goal is to write marketing copies.
        
        Your tasks:
        1. Use the detailed course content to write a tweet that announces the course
        2. Ensure that the tweet is 4 blocks, easy to read, and shows excitement
        3. Include a clear call to action to enroll in the course
        4. Proofread for grammatical errors
        
        Write a well-written tweet in markdown format, ready to be published.
        """
        
        # Format search results for the prompt
        course_info = {
            "title": search_result.course_title,
            "description": "\n".join(search_result.course_description),
            "topics": [
                f"- {topic['name']}: {topic['description']}" for topic in search_result.topics
            ],
            "audience": ", ".join(search_result.target_audience)
        }
        
        user_message = f"""
        Please write a promotional tweet for the following course:
        
        Course Title: {course_info['title']}
        
        Course Description:
        {course_info['description']}
        
        Key Topics:
        {chr(10).join(course_info['topics'])}
        
        Target Audience: {course_info['audience']}
        
        Remember to make the tweet engaging, with 4 distinct blocks of text, and include a clear call to action.
        """
        
        try:
            # Use the get_structured_output function from utils
            response = get_structured_output(
                query=user_message,
                system_message=system_message,
                response_schema=MarketingCopy,
                description="marketing copy"
            )
            
            return response
            
        except Exception as e:
            print(f"Error in writing process: {e}")
            return MarketingCopy(
                tweet=f"Error generating marketing copy: {str(e)}",
                execution_time=time.time() - start_time
            )


class EditorAgent:
    """Agent responsible for editing marketing copy for grammar and conciseness."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """Initialize the editor agent."""
        self.model = model
    
    def edit(self, marketing_copy: MarketingCopy) -> EditedMarketingCopy:
        """
        Edit marketing copy for grammar and conciseness.
        
        Args:
            marketing_copy: MarketingCopy object containing the tweet to edit
            
        Returns:
            EditedMarketingCopy object containing the edited tweet
        """
        start_time = time.time()
        
        system_message = """
        You are an editor whose goal is to edit a given tweet to ensure it is grammatically correct and concise.
        
        Your output should be a tweet in markdown format that is ready for publication.
        Avoid hashtags, emojis, and noisy details. Focus on clarity, grammar, and impactful messaging.
        """
        
        user_message = f"""
        Please edit the following marketing tweet for grammar and conciseness:
        
        {marketing_copy.tweet}
        """
        
        try:
            # Use the get_structured_output function from utils
            response = get_structured_output(
                query=user_message,
                system_message=system_message,
                response_schema=EditedMarketingCopy,
                description="edited marketing copy"
            )
            
            return response
            
        except Exception as e:
            print(f"Error in editing process: {e}")
            return EditedMarketingCopy(
                tweet=f"Error editing marketing copy: {str(e)}",
                execution_time=time.time() - start_time
            )


class MarketingCopywriterSystem:
    """Coordinates the sequential multi-agent workflow."""
    
    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = True):
        """Initialize the marketing copywriter system."""
        self.search_agent = SearchAgent(model=model)
        self.writer_agent = WriterAgent(model=model)
        self.editor_agent = EditorAgent(model=model)
        self.verbose = verbose
    
    def run(self, course_structure_file: str) -> Dict[str, Any]:
        """
        Run the complete marketing copywriter workflow.
        
        Args:
            course_structure_file: Path to the course structure file
            
        Returns:
            Dictionary containing the final marketing copy and execution metrics
        """
        start_time = time.time()
        
        if self.verbose:
            print("Starting Marketing Copywriter workflow...")
            print("=" * 50)
        
        # Step 1: Search agent analyzes course structure
        if self.verbose:
            print("\nStep 1: Search Agent")
            print("-" * 50)
        
        course_structure = self.search_agent.read_course_structure(course_structure_file)
        if course_structure.startswith("Error:"):
            return {"error": course_structure, "execution_time": time.time() - start_time}
        
        search_result = self.search_agent.search(course_structure)
        
        if self.verbose:
            print(f"Course Title: {search_result.course_title}")
            print(f"Target Audience: {', '.join(search_result.target_audience)}")
            print(f"Topics Found: {len(search_result.topics)}")
            print(f"Search execution time: {search_result.execution_time:.2f} seconds")
        
        # Step 2: Writer agent creates marketing copy
        if self.verbose:
            print("\nStep 2: Writer Agent")
            print("-" * 50)
        
        marketing_copy = self.writer_agent.write(search_result)
        
        if self.verbose:
            print("Initial Marketing Copy:")
            print(marketing_copy.tweet)
            print(f"Writing execution time: {marketing_copy.execution_time:.2f} seconds")
        
        # Step 3: Editor agent refines the marketing copy
        if self.verbose:
            print("\nStep 3: Editor Agent")
            print("-" * 50)
        
        edited_copy = self.editor_agent.edit(marketing_copy)
        
        if self.verbose:
            print("Edited Marketing Copy:")
            print(edited_copy.tweet)
            print(f"Editing execution time: {edited_copy.execution_time:.2f} seconds")
        
        # Prepare final results
        total_execution_time = time.time() - start_time
        
        result = {
            "course_title": search_result.course_title,
            "final_marketing_copy": edited_copy.tweet,
            "execution_metrics": {
                "search_time": search_result.execution_time,
                "writing_time": marketing_copy.execution_time,
                "editing_time": edited_copy.execution_time,
                "total_time": total_execution_time
            },
            "intermediate_results": {
                "initial_tweet": marketing_copy.tweet,
                "target_audience": search_result.target_audience
            }
        }
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("Marketing Copywriter Workflow Complete")
            print(f"Total execution time: {total_execution_time:.2f} seconds")
            print("=" * 50)
            print("\nFinal Marketing Copy:")
            print("-" * 50)
            print(edited_copy.tweet)
            print("-" * 50)
        
        return result


def main():
    """Main function to run the marketing copywriter workflow."""
    
    # Initialize the system
    system = MarketingCopywriterSystem(verbose=True)
    
    # Run the workflow
    print("\n" + "=" * 50)
    print("Marketing Copywriter System")
    print("=" * 50)
    
    result = system.run(COURSE_STRUCTURE_FILE)
    
    # Option to save results
    save_option = input("\nDo you want to save the results to a file? (y/n): ")
    if save_option.lower() == 'y':
        filename = input("Enter filename (default: marketing_results.json): ") or "marketing_results.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
