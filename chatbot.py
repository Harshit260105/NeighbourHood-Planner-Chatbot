import os
import google.generativeai as genai
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Successfully configured Gemini API")
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    raise

SECRET_KEY = os.getenv("SECRET_KEY")

class TravelBudgetBot:
    def __init__(self):
        try:
            # Initialize the Gemini model
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
        
        self.user_profile = {}
        self.conversation_history = []
        self.travel_budget_info = {
            "total_budget": None,
            "trip_details": {
                "destination": None,
                "duration": None,
                "travel_dates": None,
                "travelers": None
            },
            "expense_categories": {
                "accommodation": 0,
                "transportation": 0,
                "food": 0,
                "activities": 0,
                "shopping": 0,
                "miscellaneous": 0
            },
            "savings_goal": None,
            "currency": "USD"
        }
    
    def generate_content_safely(self, prompt):
        """Safely generate content with error handling"""
        try:
            logger.debug(f"Sending prompt to Gemini API: {prompt[:100]}...")
            response = self.model.generate_content(prompt)
            logger.debug("Successfully received response from Gemini API")
            return response.text
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."
    
    def process_message(self, user_message):
        """Process user messages and generate appropriate responses"""
        try:
            # Add message to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Extract information from user message
            if "budget" in user_message.lower():
                budget_match = re.search(r"(\d+)\s*(?:k|thousand|k\/trip|for trip)", user_message.lower())
                if budget_match:
                    self.travel_budget_info["total_budget"] = int(budget_match.group(1)) * 1000
            
            # Generate response based on message content
            if "budget" in user_message.lower() or "breakdown" in user_message.lower():
                response = self.generate_travel_budget_breakdown()
            elif "save" in user_message.lower() or "savings" in user_message.lower():
                response = self.suggest_travel_savings_strategies()
            elif "local" in user_message.lower() or "tips" in user_message.lower():
                response = self.provide_local_budget_tips()
            else:
                # Default response for general queries
                prompt = f"""
                You are a helpful travel budget assistant. The user has provided the following information:
                Destination: {self.travel_budget_info['trip_details']['destination']}
                Total budget: {self.travel_budget_info['total_budget']} {self.travel_budget_info['currency']}
                Duration: {self.travel_budget_info['trip_details']['duration']} days
                Current expenses: {json.dumps(self.travel_budget_info['expense_categories'])}
                
                Please provide a helpful response to: {user_message}
                
                Focus on:
                1. Travel budget management
                2. Destination-specific tips
                3. Money-saving travel strategies
                4. Local cost considerations
                5. Travel planning advice
                """
                
                response = self.generate_content_safely(prompt)
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."
    
    def generate_travel_budget_breakdown(self):
        """Generate a travel budget breakdown based on the current information"""
        if not self.travel_budget_info["total_budget"]:
            return "Please provide your total travel budget first."
        
        prompt = f"""
        Create a detailed travel budget breakdown for a trip to {self.travel_budget_info['trip_details']['destination']} 
        with a total budget of {self.travel_budget_info['total_budget']} {self.travel_budget_info['currency']}.
        
        Trip duration: {self.travel_budget_info['trip_details']['duration']} days
        Number of travelers: {self.travel_budget_info['trip_details']['travelers']}
        
        Current expenses:
        {json.dumps(self.travel_budget_info['expense_categories'], indent=2)}
        
        Provide a breakdown of:
        1. Recommended allocation for each expense category
        2. Daily budget per person
        3. Tips for saving money in each category
        4. Must-have experiences within budget
        5. Emergency fund recommendation
        6. Currency exchange tips
        
        Include specific recommendations for:
        - Finding affordable accommodation
        - Transportation options
        - Local food experiences
        - Free or low-cost activities
        - Shopping budget
        """
        
        return self.generate_content_safely(prompt)
    
    def suggest_travel_savings_strategies(self):
        """Suggest travel savings strategies"""
        prompt = f"""
        Suggest travel savings strategies based on:
        Destination: {self.travel_budget_info['trip_details']['destination']}
        Total budget: {self.travel_budget_info['total_budget']} {self.travel_budget_info['currency']}
        Duration: {self.travel_budget_info['trip_details']['duration']} days
        
        Include:
        1. Best time to book flights and accommodation
        2. Money-saving travel hacks for {self.travel_budget_info['trip_details']['destination']}
        3. Local transportation tips
        4. Free activities and attractions
        5. Budget-friendly dining options
        6. Currency exchange tips
        7. Travel insurance considerations
        """
        
        return self.generate_content_safely(prompt)
    
    def provide_local_budget_tips(self):
        """Provide destination-specific budget tips"""
        if not self.travel_budget_info["trip_details"]["destination"]:
            return "Please specify your travel destination first."
        
        prompt = f"""
        Provide budget tips specific to {self.travel_budget_info['trip_details']['destination']}:
        
        Include:
        1. Local cost of living
        2. Typical prices for common items/services
        3. Best value areas to stay
        4. Local transportation options and costs
        5. Budget-friendly local experiences
        6. Money-saving cultural tips
        7. Common tourist traps to avoid
        8. Best value local food options
        """
        
        return self.generate_content_safely(prompt)

def main():
    bot = TravelBudgetBot()
    print("Travel Budget Assistant initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = bot.process_message(user_input)
        print("Assistant:", response)

if __name__ == "__main__":
    main()