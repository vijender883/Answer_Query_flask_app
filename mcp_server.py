# mcp_server.py
import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# External API base URL
API_BASE_URL = "https://api.practicalsystemdesign.com"

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Tool definitions
TOOLS = [
    {
        "name": "GetEventRegisteredUsers",
        "description": "Get a list of all users registered for the event",
        "version": "1.0",
        "parameters": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of users to return",
                "required": False
            }
        }
    },
    {
        "name": "GetEventCheckedInUsers",
        "description": "Get a list of all users who have checked in to the event",
        "version": "1.0",
        "parameters": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of users to return",
                "required": False
            }
        }
    }
]

# API endpoints
@app.route('/api/tools', methods=['GET'])
def get_tools():
    """Get a list of all available tools"""
    return jsonify({"tools": TOOLS})

@app.route('/api/tools/<tool_name>', methods=['GET'])
def get_tool(tool_name):
    """Get details of a specific tool"""
    for tool in TOOLS:
        if tool["name"] == tool_name:
            return jsonify(tool)
    
    return jsonify({"error": f"Tool '{tool_name}' not found"}), 404

@app.route('/api/tools/<tool_name>/execute', methods=['POST'])
def execute_tool(tool_name):
    """Execute a tool by calling external API endpoints"""
    try:
        # GetEventRegisteredUsers
        if tool_name == "GetEventRegisteredUsers":
            # Call the external API endpoint
            response = requests.get(f"{API_BASE_URL}/api/eventbot/getRegisteredUserCount")
            response.raise_for_status()
            
            # Return the data directly
            return jsonify(response.json())
        
        # GetEventCheckedInUsers
        elif tool_name == "GetEventCheckedInUsers":
            # Call the external API endpoint
            response = requests.get(f"{API_BASE_URL}/api/eventbot/getCheckedInUserCount")
            response.raise_for_status()
            
            # Return the data directly
            return jsonify(response.json())
        
        # Tool not found
        else:
            return jsonify({"error": f"Tool '{tool_name}' not found"}), 404
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling external API: {e}")
        return jsonify({"error": f"Failed to call external API: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Get port from environment variable or use default (8000)
    port = int(os.environ.get("MCP_PORT", 8000))
    
    # Run the server
    app.run(host="0.0.0.0", port=port, debug=False)