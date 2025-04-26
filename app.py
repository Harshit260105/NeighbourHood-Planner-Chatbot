from flask import Flask, render_template, request, jsonify
from neighborhood_planner import NeighborhoodEventPlanner
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize event planner chatbot
planner = NeighborhoodEventPlanner()

@app.route('/')
def index():
    registration_numbers = "12303807, 12301350 ,12317980"
    return render_template('chat.html', registration_numbers=registration_numbers)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = planner.process_message(user_message)
    return jsonify({'response': response})

@app.route('/events', methods=['GET'])
def get_events():
    events = planner.get_all_events()
    return jsonify({'events': events})

@app.route('/events', methods=['POST'])
def create_event():
    event_data = request.json
    result = planner.create_event(event_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 