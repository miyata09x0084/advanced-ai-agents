import warnings
warnings.filterwarnings("ignore")
import sys
import os
from utils import *
from pydantic import BaseModel

"""
Workflow:
1. Input ("In") → Customer service queries handled by process_customer_query()
2. LLM Router → Implemented as route_query() which classifies queries into three types:
- GENERAL: Product/service questions
- REFUND: Billing and refund requests
- TECHNICAL: Technical support issues
3. Route-specific LLM Calls:
- LLM Call 1 → handle_general_query() for product/service information
- LLM Call 2 → handle_refund_query() for refund requests
- LLM Call 3 → handle_technical_query() for technical support
"""

# Delimiters for our prompts
query_delimiter = "<query>"
query_delimiter_end = "</query>"

context_delimiter = "<context>"
context_delimiter_end = "</context>"

response_delimiter = "<response>"
response_delimiter_end = "</response>"

# Knowledge base for different query types
PRODUCT_CATALOG = """
Product A: Premium Software Suite
- Price: $99.99/month
- Features: Advanced analytics, cloud storage, 24/7 support
- Trial period: 14 days

Product B: Basic Software Package
- Price: $29.99/month
- Features: Basic analytics, local storage, email support
- Trial period: 7 days
"""

REFUND_POLICY = """
Refund Policy:
1. Full refund available within 30 days of purchase
2. Partial refund (50%) available between 31-60 days
3. No refunds after 60 days
4. Must provide order number and reason for refund
5. Processing time: 5-7 business days
"""

TECHNICAL_DOCS = """
Common Technical Issues:
1. Login Problems
   - Check email/password
   - Clear browser cache
   - Enable cookies

2. Performance Issues
   - Check internet connection
   - Update to latest version
   - Clear temporary files

3. Data Sync
   - Verify account settings
   - Check storage limits
   - Restart application
"""

# Pydantic models for structured outputs
class QueryResponse(BaseModel):
    query_type: str

def route_query(user_query):
    """Router function to classify the type of customer service query"""

    system_message = f"""
    You are a customer service query router. Your task is to classify the user query into one of these categories:
    1. GENERAL: General questions about products, services, or policies
    2. REFUND: Refund requests or billing issues
    3. TECHNICAL: Technical support or troubleshooting
    
    Respond ONLY with the category name in capital letters.
    
    Analyze the query delimited by {query_delimiter}{query_delimiter_end}.
    """

    prompt = f"""
    {query_delimiter}
    {user_query}
    {query_delimiter_end}
    """

    query_type = get_structured_output(prompt, system_message, QueryResponse)
    return query_type.query_type

def handle_general_query(user_query):
    """LLM Call 1: Handle general questions about products and services"""
    
    system_message = f"""
    You are a general customer service representative. Use the product catalog below to answer questions:
    
    {context_delimiter}
    {PRODUCT_CATALOG}
    {context_delimiter_end}
    
    Provide clear, concise responses focusing on product features, pricing, and availability.
    Include relevant product details but keep responses friendly and not too technical.
    """

    prompt = f"""
    {query_delimiter}
    {user_query}
    {query_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    return get_chat_completion(messages)

def handle_refund_query(user_query):
    """LLM Call 2: Handle refund requests and billing issues"""
    
    system_message = f"""
    You are a billing specialist. Use the refund policy below to handle requests:
    
    {context_delimiter}
    {REFUND_POLICY}
    {context_delimiter_end}
    
    Be empathetic but clear about policy requirements. Always ask for:
    1. Order number (if not provided)
    2. Purchase date
    3. Reason for refund
    """

    prompt = f"""
    {query_delimiter}
    {user_query}
    {query_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    return get_chat_completion(messages)

def handle_technical_query(user_query):
    """LLM Call 3: Handle technical support issues"""
    
    system_message = f"""
    You are a technical support specialist. Use the technical documentation below to assist users:
    
    {context_delimiter}
    {TECHNICAL_DOCS}
    {context_delimiter_end}
    
    Provide step-by-step solutions. If the issue isn't covered in the docs:
    1. Ask for specific error messages
    2. Request system/version information
    3. Suggest general troubleshooting steps
    """

    prompt = f"""
    {query_delimiter}
    {user_query}
    {query_delimiter_end}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    return get_chat_completion(messages)

def process_customer_query(user_query):
    """Main function to process and route customer queries"""
    print(f"Processing query: '{user_query}'")
    
    # Route the query
    query_type = route_query(user_query)
    print(f"Query classified as: {query_type}")
    
    # Handle based on type
    if query_type == "GENERAL":
        response = handle_general_query(user_query)
    elif query_type == "REFUND":
        response = handle_refund_query(user_query)
    elif query_type == "TECHNICAL":
        response = handle_technical_query(user_query)
    else:
        response = "Sorry, I couldn't properly classify your query. Please try rephrasing your question."
    
    return response

if __name__ == "__main__":
    # Example queries to test the routing system
    test_queries = [
        "What features are included in the Premium Software Suite?",  # General
        "I want to request a refund for my recent purchase",  # Refund
        "I'm having trouble logging into my account",  # Technical
    ]
    
    print("\nTesting Customer Service Query Router")
    print("=" * 50)
    
    for query in test_queries:
        print("\nTest Query:", query)
        print("-" * 30)
        response = process_customer_query(query)
        print("\nResponse:", response)
        print("=" * 50)
