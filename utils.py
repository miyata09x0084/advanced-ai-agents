from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

# get chat completion from standard chat LLMs
def get_chat_completion(messages, model="gpt-4o", temperature=0, tools=None, tool_choice=None, parallel_tool_calls=False):
    """
    Simple get chat completion function
    """
   
    if tools:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls
    )
        return response.choices[0].message
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

# get reasoning response
def get_reasoning_response(messages, model="o1-mini", reasoning_effort="low"):
    """
    Simple reasoning response
    """

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort
    )
    return response.choices[0].message.content

def get_structured_output(query, system_message, response_schema, description="structured output"):
    """Generic function for getting structured outputs from OpenAI
    
    Args:
        query: The user query to process
        system_message: The system prompt to guide the model's response
        response_schema: The Pydantic model to use for structured output
        description: Description of the response type for error messages
        
    Returns:
        The parsed response object or None if an error occurred
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    try:
        # Using OpenAI's structured output API
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=response_schema,
        )
        
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error with {description} response: {e}")
        return None


def get_structured_output_with_cache_info(query, system_message, response_schema, description="structured output"):
    """Enhanced function for getting structured outputs from OpenAI that also returns cache hit information
    
    Args:
        query: The user query to process
        system_message: The system prompt to guide the model's response
        response_schema: The Pydantic model to use for structured output
        description: Description of the response type for error messages
        
    Returns:
        A tuple containing:
        - The parsed response object or None if an error occurred
        - A dictionary with cache hit information
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    try:
        # Using OpenAI's structured output API
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=response_schema,
        )
        
        # Extract cache information from the response
        cache_info = {}
        if hasattr(completion, 'usage') and completion.usage:
            usage = completion.usage
            
            # Get total prompt tokens (corresponds to total prompt input tokens and doesn't include completion tokens)
            cache_info['total_prompt_tokens'] = usage.prompt_tokens
            
            # Get cached tokens information if available
            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                if hasattr(usage.prompt_tokens_details, 'cached_tokens'):
                    cache_info['cached_tokens'] = usage.prompt_tokens_details.cached_tokens
                    cache_info['cache_hit_ratio'] = cache_info['cached_tokens'] / cache_info['total_prompt_tokens'] if cache_info['total_prompt_tokens'] > 0 else 0
                    cache_info['is_cache_hit'] = cache_info['cached_tokens'] > 0
        
        return completion.choices[0].message.parsed, cache_info
    except Exception as e:
        print(f"Error with {description} response: {e}")
        return None, {}


def get_structured_output_with_tools(messages: list, response_schema: type, tools: list = None, available_functions: dict = None, model: str = "gpt-4o", description: str = "structured output"):
    """Gets structured output using Pydantic, handling potential tool calls.
    
    Args:
        messages: List of message objects for the conversation history.
        response_schema: The Pydantic model class for the final expected response.
        tools: Optional list of tool definitions for the model.
        available_functions: Optional dict mapping tool names to Python functions.
        model: The OpenAI model to use.
        description: Description for error messages.
    
    Returns:
        The parsed Pydantic object or None if an error occurred.
    """
    try:
        print(f"--- Util: Initial call for {description} (tools available: {bool(tools)}) ---")
        # Initial API call
        response_message = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
        ).choices[0].message

        tool_calls = response_message.tool_calls

        # Handle tool calls if requested
        if tool_calls and available_functions:
            print(f"--- Orchestrator: Tool call(s) requested for {description} ---")
            messages.append(response_message)  # Add assistant's tool call message

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name not in available_functions:
                    print(f"--- Orchestrator: Error - Unknown function {function_name} requested ---")
                    continue # Or raise an error

                function_to_call = available_functions[function_name]
                # Assuming tools currently don't need arguments based on get_current_datetime
                # Future enhancement: parse tool_call.function.arguments if needed
                try:
                    function_response = function_to_call()
                    print(f"--- Orchestrator: Executed {function_name}, result: {function_response} ---")
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_response), # Ensure content is string
                        }
                    )
                except Exception as func_exc:
                     print(f"--- Orchestrator: Error executing function {function_name}: {func_exc} ---")
                     # Append an error message for the tool call
                     messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing function: {func_exc}",
                        }
                    )

            # Second API call after executing tools, expecting the structured response
            print(f"--- Orchestrator: Second call for {description} after tool execution ---")
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_schema,
            )
            parsed_response = completion.choices[0].message.parsed

        elif not tool_calls:
             # No tool calls, attempt to parse directly from the first response
             # This assumes the model *might* return the structure directly if no tools needed
             print(f"--- Orchestrator: No tool calls for {description}, attempting direct parse ---")
             # We need to re-call using the .parse endpoint for structured output
             completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages, # Use original messages
                response_format=response_schema,
             )
             parsed_response = completion.choices[0].message.parsed
        else:
            # Tool calls were made, but no available_functions provided
            print(f"--- Orchestrator: Tool calls requested but no functions provided for {description} ---")
            return None # Cannot proceed

        print(f"--- Orchestrator: Successfully parsed {description} ---")
        return parsed_response

    except Exception as e:
        import traceback
        print(f"--- Orchestrator: Error getting/parsing {description}: {e} ---")
        print(traceback.format_exc()) # Print stack trace for debugging
        return None


def get_current_datetime_tool_func():
    """Returns the current date and time as an ISO 8601 formatted string."""
    from datetime import datetime
    return datetime.now().isoformat()


def get_menu_items():
    """Returns a comprehensive list of menu items with detailed descriptions.
    This function centralizes all menu data to make it easier to maintain and extend.
    
    Returns:
        str: A formatted string containing all menu items with their details
    """
    return """
Menu: Kids Menu    
Food Item: Mini Cheeseburger
Price: $6.99
Vegan: N
Popularity: 4/5
Included: Mini beef patty, cheese, lettuce, tomato, and fries.
Description: A perfectly sized burger for young diners, made with high-quality beef and fresh toppings.
Allergens: Contains dairy, gluten
Preparation: Grilled to medium-well for food safety

Menu: Kids Menu
Food Item: Chicken Nuggets
Price: $5.99
Vegan: N
Popularity: 5/5
Included: Six chicken nuggets, fries, and choice of dipping sauce
Description: White meat chicken nuggets, lightly breaded and fried to a golden crisp
Allergens: Contains gluten
Preparation: Fried in dedicated oil to avoid cross-contamination

Menu: Kids Menu
Food Item: Mac & Cheese Bowl
Price: $4.99
Vegan: N
Popularity: 4/5
Included: Creamy macaroni and cheese with breadcrumb topping
Description: House-made cheese sauce with elbow macaroni, perfect for picky eaters
Allergens: Contains dairy, gluten
Preparation: Baked fresh with a three-cheese blend

Menu: Appetizers
Food Item: Loaded Potato Skins
Price: $8.99
Vegan: N
Popularity: 3/5
Included: Crispy potato skins filled with cheese, bacon bits, and served with sour cream.
Description: Double-baked potato skins with melted cheddar and crispy bacon pieces.
Allergens: Contains dairy
Preparation: Baked until golden brown

Menu: Appetizers
Food Item: Spinach Artichoke Dip
Price: $9.99
Vegan: N
Popularity: 4/5
Included: Creamy spinach and artichoke dip with tortilla chips
Description: Blend of cheeses, fresh spinach, and marinated artichoke hearts
Allergens: Contains dairy
Preparation: Baked to order and served bubbling hot

Menu: Appetizers
Food Item: Calamari
Price: $12.99
Vegan: N
Popularity: 3/5
Included: Lightly breaded calamari rings with marinara and lemon aioli
Description: Tender squid rings in a seasoned coating, flash-fried for perfect texture
Allergens: Contains shellfish, gluten
Preparation: Prepared to order to ensure freshness

Menu: Main Course
Food Item: Grilled Salmon Fillet
Price: $24.99
Vegan: N
Popularity: 5/5
Included: Fresh Atlantic salmon, seasonal vegetables, lemon herb sauce
Description: Premium salmon fillet grilled to perfection with a crispy skin and tender center
Allergens: Fish
Preparation: Grilled with herbs and finished with house-made sauce

Menu: Main Course
Food Item: New York Strip Steak
Price: $29.99
Vegan: N
Popularity: 5/5
Included: 12oz USDA Choice steak, garlic mashed potatoes, grilled asparagus
Description: Hand-cut steak with perfect marbling, aged for tenderness and flavor
Allergens: None
Preparation: Grilled to your preferred temperature with house seasoning

Menu: Main Course
Food Item: Chicken Parmesan
Price: $18.99
Vegan: N
Popularity: 4/5
Included: Breaded chicken breast, marinara, mozzarella, spaghetti
Description: Hand-breaded chicken topped with house marinara and melted cheese
Allergens: Contains dairy, gluten
Preparation: Baked until cheese is golden and bubbly

Menu: Vegetarian
Food Item: Mushroom Risotto
Price: $18.99
Vegan: N
Popularity: 4/5
Included: Arborio rice, wild mushrooms, parmesan, truffle oil
Description: Creamy Italian risotto with a medley of seasonal mushrooms and aged parmesan
Allergens: Contains dairy
Preparation: Slow-cooked to achieve perfect consistency

Menu: Vegetarian
Food Item: Eggplant Parmesan
Price: $16.99
Vegan: N
Popularity: 3/5
Included: Breaded eggplant, marinara sauce, mozzarella, side salad
Description: Thinly sliced eggplant layered with sauce and cheese, baked to perfection
Allergens: Contains dairy, gluten
Preparation: Eggplant is salted to remove bitterness before cooking

Menu: Vegetarian
Food Item: Stuffed Bell Peppers
Price: $15.99
Vegan: N
Popularity: 3/5
Included: Bell peppers stuffed with rice, beans, corn, cheese
Description: Colorful bell peppers filled with a flavorful mixture of grains and vegetables
Allergens: Contains dairy
Preparation: Roasted to enhance natural sweetness of peppers

Menu: Desserts
Food Item: Tiramisu
Price: $8.99
Vegan: N
Popularity: 5/5
Included: Ladyfingers, mascarpone cream, coffee, cocoa powder
Description: Classic Italian dessert with layers of coffee-soaked cookies and creamy filling
Allergens: Contains dairy, eggs
Preparation: House-made daily and chilled for optimal texture

Menu: Desserts
Food Item: Chocolate Lava Cake
Price: $7.99
Vegan: N
Popularity: 5/5
Included: Warm chocolate cake with molten center, vanilla ice cream
Description: Rich chocolate cake with a decadent flowing center, served warm
Allergens: Contains dairy, eggs, gluten
Preparation: Baked to order, requires 15 minutes preparation time

Menu: Desserts
Food Item: New York Cheesecake
Price: $8.99
Vegan: N
Popularity: 4/5
Included: Classic cheesecake with graham cracker crust, berry compote
Description: Dense, rich cheesecake with a buttery crust and seasonal fruit topping
Allergens: Contains dairy, gluten
Preparation: Baked slowly in a water bath for perfect texture

Menu: Vegan Special
Food Item: Buddha Bowl
Price: $16.99
Vegan: Y
Popularity: 4/5
Included: Quinoa, roasted vegetables, chickpeas, tahini dressing
Description: Nutrient-rich bowl featuring seasonal vegetables and protein-packed legumes
Allergens: Contains sesame
Preparation: Components prepared fresh daily

Menu: Vegan Special
Food Item: Impossible Burger
Price: $15.99
Vegan: Y
Popularity: 4/5
Included: Plant-based patty, lettuce, tomato, vegan cheese, sweet potato fries
Description: Revolutionary plant-based burger that looks and tastes like beef
Allergens: Contains soy
Preparation: Grilled on dedicated surface to avoid cross-contamination

Menu: Vegan Special
Food Item: Coconut Curry Vegetables
Price: $14.99
Vegan: Y
Popularity: 3/5
Included: Mixed vegetables in coconut curry sauce, jasmine rice
Description: Aromatic curry with seasonal vegetables in a rich coconut milk base
Allergens: Contains tree nuts
Preparation: Made to order with adjustable spice levels

Menu: Breakfast
Food Item: Eggs Benedict
Price: $14.99
Vegan: N
Popularity: 5/5
Included: English muffin, poached eggs, Canadian bacon, hollandaise sauce
Description: Classic breakfast dish with perfectly poached eggs and creamy sauce
Allergens: Contains dairy, eggs, gluten
Preparation: Made to order with house-made hollandaise

Menu: Breakfast
Food Item: Belgian Waffle
Price: $10.99
Vegan: N
Popularity: 4/5
Included: Large Belgian waffle, maple syrup, whipped butter, berries
Description: Light and crispy waffle with deep pockets for holding syrup
Allergens: Contains dairy, eggs, gluten
Preparation: Made fresh on traditional Belgian waffle irons

Menu: Breakfast
Food Item: Avocado Toast
Price: $12.99
Vegan: Y
Popularity: 4/5
Included: Artisan bread, smashed avocado, microgreens, cherry tomatoes
Description: Simple yet elegant breakfast featuring ripe avocados and quality ingredients
Allergens: Contains gluten
Preparation: Avocados smashed to order for maximum freshness

Menu: Lunch Special
Food Item: Asian Chicken Salad
Price: $15.99
Vegan: N
Popularity: 4/5
Included: Grilled chicken, mixed greens, mandarin oranges, crispy wontons
Description: Fresh and light salad with Asian-inspired flavors and textures
Allergens: Contains gluten, soy
Preparation: Chicken marinated for 24 hours before grilling

Menu: Pizza
Food Item: Margherita Pizza
Price: $16.99
Vegan: N
Popularity: 5/5
Included: San Marzano tomatoes, fresh mozzarella, basil, olive oil
Description: Traditional Neapolitan pizza baked in our wood-fired oven
Allergens: Contains dairy, gluten
Preparation: Cooked at 900°F for perfect crust

Menu: Seafood
Food Item: Seafood Paella
Price: $28.99
Vegan: N
Popularity: 4/5
Included: Saffron rice, shrimp, mussels, calamari, chorizo
Description: Spanish rice dish loaded with fresh seafood and authentic spices
Allergens: Shellfish, fish
Preparation: Traditional preparation in paella pan

Menu: Gluten-Free
Food Item: Quinoa Power Bowl
Price: $17.99
Vegan: Y
Popularity: 4/5
Included: Quinoa, roasted sweet potato, black beans, avocado
Description: Protein-rich bowl perfect for health-conscious diners
Allergens: None
Preparation: Components prepared separately to prevent cross-contamination

Menu: Signature Dish
Food Item: Braised Short Ribs
Price: $26.99
Vegan: N
Popularity: 5/5
Included: Beef short ribs, mashed potatoes, roasted vegetables, red wine sauce
Description: Tender beef slow-cooked for 12 hours in rich wine sauce
Allergens: Contains dairy
Preparation: Braised overnight for maximum tenderness
"""