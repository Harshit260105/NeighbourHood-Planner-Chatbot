import os
import google.generativeai as genai
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import logging
import random

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

class NeighborhoodEventPlanner:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
        
        self.events = []
        self.conversation_history = []
        self.last_question_asked = None
        self.question_count = 0
        self.event_categories = [
            "Community Gathering",
            "Sports & Recreation",
            "Cultural Celebration",
            "Educational Workshop",
            "Fundraiser",
            "Clean-up Drive",
            "Food Festival",
            "Arts & Crafts",
            "Children's Activities",
            "Senior Citizens' Meet"
        ]
        
        # Varied questions to avoid repetition
        self.varied_questions = [
            "Would you like to create a new event?",
            "Should we plan something for next month?",
            "Would you like to see more event ideas?",
            "Do you need help with event planning?",
            "Would you like to list all current events?",
            "Should we focus on a specific category?",
            "Would you like to see budget options?",
            "Do you want to explore venue suggestions?",
            "Would you like to see seasonal event ideas?",
            "Should we plan a community gathering?"
        ]
        
        # Track previously suggested ideas to avoid repetition
        self.previous_suggestions = set()
        self.creative_themes = [
            "Under the Stars",
            "Around the World",
            "Retro Revival",
            "Future Forward",
            "Nature's Bounty",
            "Urban Adventure",
            "Cultural Fusion",
            "Seasonal Spectacular",
            "Community Heroes",
            "Local Legends"
        ]
        
        # Unique event formats to suggest
        self.event_formats = [
            "pop-up",
            "flash mob",
            "guerrilla",
            "roaming",
            "interactive",
            "immersive",
            "collaborative",
            "participatory",
            "experiential",
            "transformative"
        ]
    
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

    def get_varied_question(self):
        """Get a varied question that hasn't been asked recently"""
        # If we've asked the same question twice, force a different one
        if self.question_count >= 2:
            available_questions = [q for q in self.varied_questions if q != self.last_question_asked]
            if available_questions:
                question = random.choice(available_questions)
            else:
                # If we've used all questions, generate a new one
                question = self.generate_new_question()
            self.question_count = 0
        else:
            # Otherwise, just pick a random question
            question = random.choice(self.varied_questions)
            
        # If it's the same as last time, increment counter
        if question == self.last_question_asked:
            self.question_count += 1
        else:
            self.question_count = 0
            
        self.last_question_asked = question
        return question

    def generate_new_question(self):
        """Generate a new, unique question to avoid repetition"""
        themes = random.sample(self.creative_themes, 1)[0]
        formats = random.sample(self.event_formats, 1)[0]
        
        new_questions = [
            f"Would you like to plan a {themes} themed event?",
            f"Should we organize a {formats} event for the community?",
            f"Would you like to explore {random.choice(self.event_categories)} ideas?",
            f"Should we plan something special for {random.choice(['families', 'seniors', 'youth', 'everyone'])}?",
            f"Would you like to see budget-friendly event options?",
            f"Should we focus on {random.choice(['indoor', 'outdoor', 'hybrid'])} events?",
            f"Would you like to plan a {random.choice(['day', 'evening', 'weekend'])} event?",
            f"Should we organize something with {random.choice(['local businesses', 'schools', 'artists', 'musicians'])}?"
        ]
        
        return random.choice(new_questions)

    def process_message(self, user_message):
        """Process user messages and generate appropriate responses"""
        try:
            # Add message to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Check if the query is related to neighborhood planning
            neighborhood_keywords = [
                "event", "community", "neighborhood", "gathering", "plan", "organize",
                "venue", "budget", "activities", "participants", "schedule", "location",
                "celebration", "festival", "workshop", "meeting", "social", "local"
            ]
            
            is_neighborhood_related = any(keyword in user_message.lower() for keyword in neighborhood_keywords)
            
            if not is_neighborhood_related:
                return "I apologize, but I can only provide information related to neighborhood event planning. Please ask questions about community events, gatherings, or neighborhood activities."
            
            # Check for follow-up responses to previous suggestions
            if len(self.conversation_history) >= 2:
                last_bot_message = self.conversation_history[-2]["content"]
                if "Would you like to" in last_bot_message or "Should we" in last_bot_message:
                    # Handle follow-up to a specific question
                    if "create" in last_bot_message.lower() and "event" in last_bot_message.lower():
                        response = self.handle_event_creation(user_message)
                    elif "list" in last_bot_message.lower() and "events" in last_bot_message.lower():
                        response = self.list_events()
                    elif "suggest" in last_bot_message.lower() or "ideas" in last_bot_message.lower():
                        response = self.suggest_events()
                    elif "plan" in last_bot_message.lower():
                        response = self.handle_event_creation(user_message)
                    elif "budget" in last_bot_message.lower():
                        response = self.handle_budget_query(user_message)
                    elif "venue" in last_bot_message.lower():
                        response = self.handle_venue_query(user_message)
                    else:
                        response = self.handle_general_followup(user_message, last_bot_message)
                    self.conversation_history.append({"role": "assistant", "content": response})
                    return response
            
            # Check for specific event-related queries
            if any(keyword in user_message.lower() for keyword in ["create event", "plan event", "organize"]):
                response = self.handle_event_creation(user_message)
            elif "list events" in user_message.lower() or "show events" in user_message.lower():
                response = self.list_events()
            elif "suggest" in user_message.lower() or "recommend" in user_message.lower():
                response = self.suggest_events()
            else:
                prompt = f"""
                You are a helpful neighborhood event planning assistant. Help the user with their query about community events.
                Current events planned: {json.dumps(self.events, indent=2)}
                Available event categories: {', '.join(self.event_categories)}
                
                User query: {user_message}
                
                Provide a creative and structured response that:
                1. Starts with a brief, engaging introduction (1 sentence)
                2. Uses emojis to make the response more visually appealing
                3. Organizes information in clear subpoints with bullet points
                4. Includes specific details when relevant
                5. Mentions budgets in Indian Rupees (₹)
                6. Uses creative language and metaphors
                7. Does not include a question at the end
                
                Format your response as:
                - Main response with emoji (1 sentence)
                - Subpoint 1: [emoji] Detail
                - Subpoint 2: [emoji] Detail
                - Subpoint 3: [emoji] Detail
                - No paragraphs or long text blocks
                """
                response = self.generate_content_safely(prompt)
                
                # Add a varied question at the end
                response += "\n\n" + self.get_varied_question()
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."

    def create_event(self, event_data):
        """Create a new neighborhood event"""
        try:
            required_fields = ["title", "date", "location", "description", "category"]
            if not all(field in event_data for field in required_fields):
                return {"error": "Missing required fields"}
            
            event = {
                "id": len(self.events) + 1,
                "title": event_data["title"],
                "date": event_data["date"],
                "location": event_data["location"],
                "description": event_data["description"],
                "category": event_data["category"],
                "attendees": [],
                "created_at": datetime.now().isoformat()
            }
            
            self.events.append(event)
            return {"success": True, "event": event}
            
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            return {"error": str(e)}

    def get_all_events(self):
        """Return all planned events"""
        return self.events

    def handle_event_creation(self, user_message):
        """Handle event creation requests"""
        # Generate a unique seed for this suggestion to ensure variety
        seed = random.randint(1, 1000)
        
        prompt = f"""
        Help create a unique and creative neighborhood event based on the user's request: {user_message}
        
        Consider these creative elements:
        - Creative themes: {', '.join(random.sample(self.creative_themes, 3))}
        - Event formats: {', '.join(random.sample(self.event_formats, 3))}
        
        Provide a creative and structured response with:
        1. Event title and category with emoji
        2. Date, time, and venue with emoji
        3. Budget estimate (in ₹) with emoji
        4. Key activities (max 3) with emojis
        
        Format your response as:
        - Main event details with emoji (1-2 sentences)
        - 🗓️ Date & Time: [details]
        - 📍 Venue: [details]
        - 💰 Budget: [details]
        - 🎯 Activities: [details]
        - No paragraphs or long text blocks
        
        Available categories: {', '.join(self.event_categories)}
        
        IMPORTANT: 
        - Keep response very concise
        - Focus on essential details only
        - Use emojis for visual appeal
        - Always mention budgets in Indian Rupees (₹)
        - Do not include a question at the end
        - Be creative and suggest unique ideas
        """
        response = self.generate_content_safely(prompt)
        
        # Add a varied question at the end
        response += "\n\n" + self.get_varied_question()
        return response

    def suggest_events(self):
        """Suggest event ideas based on current events and categories"""
        # Generate a unique seed for this suggestion to ensure variety
        seed = random.randint(1, 1000)
        
        prompt = f"""
        Suggest 2-3 creative and unique neighborhood event ideas considering:
        
        Current events: {json.dumps(self.events, indent=2)}
        Available categories: {', '.join(self.event_categories)}
        Creative themes: {', '.join(self.creative_themes)}
        Event formats: {', '.join(self.event_formats)}
        
        IMPORTANT: Generate completely unique ideas that haven't been suggested before.
        Avoid common or generic event ideas. Be creative and think outside the box.
        
        For each suggestion, provide:
        1. Event title and category with emoji
        2. Brief description (1 sentence) with emoji
        3. Budget estimate (in ₹) with emoji
        4. Key activities (max 2) with emojis
        
        Format your response as:
        - 🎉 Event 1: [title]
          📝 Description: [brief description]
          💰 Budget: [amount]
          🎯 Activities: [activities]
        
        - 🎪 Event 2: [title]
          📝 Description: [brief description]
          💰 Budget: [amount]
          🎯 Activities: [activities]
        
        - 🎨 Event 3: [title] (if applicable)
          📝 Description: [brief description]
          💰 Budget: [amount]
          🎯 Activities: [activities]
        
        IMPORTANT: 
        - Keep each event suggestion very concise
        - Focus on unique and engaging ideas
        - Use emojis for visual appeal
        - Always mention budgets in Indian Rupees (₹)
        - Do not include a question at the end
        - Avoid suggesting ideas that are too similar to existing events
        """
        response = self.generate_content_safely(prompt)
        
        # Add a varied question at the end
        response += "\n\n" + self.get_varied_question()
        return response

    def list_events(self):
        """Format and return list of all events"""
        if not self.events:
            return "📅 No events planned yet. Would you like to create one? Try clicking 'Create Event' or ask for suggestions!"
        
        events_text = "📋 Current Events:\n"
        for event in self.events:
            events_text += f"• 🎉 {event['title']} - 📅 {event['date']} at 📍 {event['location']}\n"
        
        # Add a varied question at the end
        events_text += "\n" + self.get_varied_question()
        return events_text

    def handle_budget_query(self, user_message):
        """Handle budget-related follow-up questions"""
        prompt = f"""
        Provide budget guidance for the user's query: {user_message}
        
        Consider:
        1. Typical budget ranges for similar events
        2. Cost-saving suggestions
        3. Essential expenses to consider
        
        Format your response as:
        - 💰 Main budget guidance with emoji (1-2 sentences)
        - 📊 Budget Breakdown:
          • [expense category]: [amount]
          • [expense category]: [amount]
          • [expense category]: [amount]
        - 💡 Cost-saving tips:
          • [tip 1]
          • [tip 2]
          • [tip 3]
        - No paragraphs or long text blocks
        
        IMPORTANT:
        - Keep response very concise
        - Focus on practical budget advice
        - Use emojis for visual appeal
        - Always mention amounts in Indian Rupees (₹)
        - Do not include a question at the end
        """
        response = self.generate_content_safely(prompt)
        response += "\n\n" + self.get_varied_question()
        return response

    def handle_venue_query(self, user_message):
        """Handle venue-related follow-up questions"""
        prompt = f"""
        Provide venue suggestions for the user's query: {user_message}
        
        Consider:
        1. Suitable venue types
        2. Location accessibility
        3. Capacity requirements
        4. Cost considerations
        
        Format your response as:
        - 📍 Main venue recommendation with emoji (1-2 sentences)
        - 🏢 Venue Options:
          • [venue type]: [description]
          • [venue type]: [description]
          • [venue type]: [description]
        - 🚗 Accessibility:
          • [detail 1]
          • [detail 2]
        - 💰 Cost Range: [amount]
        - No paragraphs or long text blocks
        
        IMPORTANT:
        - Keep response very concise
        - Focus on practical venue options
        - Use emojis for visual appeal
        - Include approximate costs in Indian Rupees (₹)
        - Do not include a question at the end
        """
        response = self.generate_content_safely(prompt)
        response += "\n\n" + self.get_varied_question()
        return response

    def handle_general_followup(self, user_message, last_bot_message):
        """Handle general follow-up responses"""
        prompt = f"""
        Respond to the user's follow-up: {user_message}
        
        Previous bot message: {last_bot_message}
        
        Provide a creative and structured response that:
        1. Addresses the specific follow-up question
        2. Maintains context from previous message
        3. Offers relevant next steps
        4. Uses emojis for visual appeal
        
        Format your response as:
        - Main response with emoji (1-2 sentences)
        - 📌 Key Points:
          • [point 1]
          • [point 2]
          • [point 3]
        - 🚀 Next Steps:
          • [step 1]
          • [step 2]
        - No paragraphs or long text blocks
        
        IMPORTANT:
        - Keep response very concise
        - Focus on actionable information
        - Use emojis for visual appeal
        - Include specific details when relevant
        - Do not include a question at the end
        """
        response = self.generate_content_safely(prompt)
        response += "\n\n" + self.get_varied_question()
        return response 