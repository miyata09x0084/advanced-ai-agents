import warnings
warnings.filterwarnings("ignore")
import sys
import os
from utils import *

"""
Workflow: 
Input ("In") → The input is handled by the create_document() function which takes a topic, style guide, and criteria
LLM Call 1 → Implemented as generate_outline() which creates the initial document outline
Gate → Implemented as check_outline_criteria() which validates if the outline meets requirements
Pass Path:
LLM Call 2 → expand_outline_sections() which expands each section into detailed content
LLM Call 3 → write_final_document() which creates the final polished document
Fail Path → Returns None with feedback if criteria aren't met
Output ("Out") → The final document text

Important Considerations:
- Separation of Concerns
- Structure your inputs and use delimiters
- Use consistent formatting and spacing
- Criteria check to verify outputs
- Structured outputs (not applied in this example)
"""

# Delimiters for our prompts
outline_delimiter = "<outline>"
outline_delimiter_end = "</outline>"

criteria_delimiter = "<criteria>"
criteria_delimiter_end = "</criteria>"

document_delimiter = "<document>"
document_delimiter_end = "</document>"

def generate_outline(topic, style_guide):
    """First LLM call to generate an initial document outline"""
    system_message = f"""
    Your task is to create a well-structured outline for a document about the given topic.
    The outline should follow the style guide provided and include main sections and subsections.
    
    Format the outline with:
    - Main sections marked with numbers (1., 2., 3., etc.)
    - Subsections marked with letters (a., b., c., etc.)
    - Brief descriptions for each section
    
    Output the outline between {outline_delimiter} and {outline_delimiter_end} tags.
    """

    prompt = f"""
    Topic: {topic}
    Style Guide: {style_guide}
    
    Please generate a detailed outline.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    outline = get_chat_completion(messages)
    return outline

def check_outline_criteria(outline, criteria):
    """Gate function to validate if the outline meets specified criteria"""
    system_message = f"""
    Your task is to evaluate if the outline meets all specified criteria.
    You must return ONLY "PASS" or "FAIL" followed by a brief explanation.
    
    Review the outline (delimited by {outline_delimiter}{outline_delimiter_end})
    against the criteria (delimited by {criteria_delimiter}{criteria_delimiter_end}).
    """

    prompt = f"""
    {outline_delimiter}
    {outline}
    {outline_delimiter_end}
    
    {criteria_delimiter}
    {criteria}
    {criteria_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    result = get_chat_completion(messages)
    return result.strip().startswith("PASS"), result

def expand_outline_sections(outline):
    """Second LLM call to expand the outline into detailed sections"""
    system_message = f"""
    Your task is to expand each section of the approved outline into detailed paragraphs.
    For each section:
    1. Maintain the original structure
    2. Add 2-3 paragraphs of content
    3. Include relevant examples or supporting points
    
    Keep the expanded content between {document_delimiter} and {document_delimiter_end} tags.
    """

    prompt = f"""
    Please expand this outline into detailed sections:
    
    {outline_delimiter}
    {outline}
    {outline_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    expanded_sections = get_chat_completion(messages)
    return expanded_sections

def write_final_document(expanded_sections, style_guide):
    """Third LLM call to write the final polished document"""
    system_message = f"""
    Your task is to transform the expanded sections into a polished, cohesive document.
    
    1. Add smooth transitions between sections
    2. Ensure consistent tone and style
    3. Include an introduction and conclusion
    4. Follow the provided style guide
    
    Output the final document between {document_delimiter} and {document_delimiter_end} tags.
    """

    prompt = f"""
    Style Guide: {style_guide}
    
    Please convert these expanded sections into a polished document:
    
    {document_delimiter}
    {expanded_sections}
    {document_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    final_document = get_chat_completion(messages)
    return final_document

def create_document(topic, style_guide, criteria):
    """Main function to orchestrate the document creation workflow"""
    # Step 1: Generate initial outline
    outline = generate_outline(topic, style_guide)
    print("✓ Generated initial outline")
    
    # Step 2: Check if outline meets criteria (Gate)
    passes_criteria, feedback = check_outline_criteria(outline, criteria)
    print(f"→ Criteria check: {feedback}")
    
    if not passes_criteria:
        print("✗ Document creation stopped: Outline did not meet criteria")
        return None
    
    # Step 3: Expand outline into sections
    expanded_sections = expand_outline_sections(outline)
    print("✓ Expanded outline into detailed sections")
    
    # Step 4: Write final document
    final_document = write_final_document(expanded_sections, style_guide)
    print("✓ Completed final document")
    
    return final_document

if __name__ == "__main__":
    # Example usage
    topic = "The Impact of Artificial Intelligence on Modern Healthcare"
    
    style_guide = """
    - Use professional, academic tone
    - Include real-world examples
    - Keep paragraphs concise (3-4 sentences)
    - Use active voice
    - Target audience: Healthcare professionals
    """
    
    criteria = """
    1. Must have at least 3 main sections
    2. Must include both benefits and challenges
    3. Must address ethical considerations
    4. Must include future perspectives
    5. Must have clear logical flow between sections
    """
    
    final_document = create_document(topic, style_guide, criteria)
    
    if final_document:
        print("\nFinal Document:")
        print("=" * 50)
        print(final_document)
