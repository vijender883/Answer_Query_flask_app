# agent_system.py
import json
from typing import Dict, Any, List
import google.generativeai as genai

class EventAgent:
    """Agent system for analyzing queries and routing to appropriate tools or RAG"""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        
        # Initialize Gemini for query understanding
        try:
            self.gemini = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            print(f"Gemini initialization error: {e}")
            self.gemini = None
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and required tools"""
        tools = self.mcp_client.discover_tools()
        tool_descriptions = "\n".join([f"- {tool['name']}: {tool.get('description', 'No description')}" for tool in tools])
        
        analysis_prompt = f"""
        You are an Event Bot agent that needs to understand user queries and determine how to handle them.
        
        Available tools:
        {tool_descriptions}
        
        When a query is about:
        - Registered users/participants/attendees lists or counts, use the GetEventRegisteredUsers tool
        - Checked-in users/participants/attendees lists or counts, use the GetEventCheckedInUsers tool
        - Any other event information or participant details should use RAG database query
        
        Analyze this user query: "{query}"
        
        Return your analysis as a JSON object with these fields:
        - tool_needed: string (name of the tool to use, or "rag" for RAG database)
        - confidence: number (0-1, your confidence in this decision)
        - reasoning: string (brief explanation of your decision)
        """
        
        try:
            if self.gemini:
                response = self.gemini.generate_content(analysis_prompt)
                # Parse JSON from response
                result_text = response.text
                # Extract JSON from possible markdown code blocks
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                    
                return json.loads(result_text)
            else:
                # Fallback to simple keyword matching if Gemini is not available
                return self._simple_query_analysis(query)
        except Exception as e:
            print(f"Query analysis error: {e}")
            # Fallback to simple analysis
            return self._simple_query_analysis(query)
    
    def _simple_query_analysis(self, query: str) -> Dict[str, Any]:
        """Simple keyword-based query analysis fallback"""
        query_lower = query.lower()
        
        # Check for registered users queries
        registered_keywords = ["registered", "registration", "sign up", "signed up", "attending", "will attend"]
        if any(keyword in query_lower for keyword in registered_keywords) or "fetch registered users" in query_lower:
            return {
                "tool_needed": "GetEventRegisteredUsers",
                "confidence": 0.8,
                "reasoning": "Query contains keywords related to registered users"
            }
            
        # Check for checked-in users queries
        checkin_keywords = ["check in", "checked in", "arrived", "here now", "present", "attendance"]
        if any(keyword in query_lower for keyword in checkin_keywords) or "fetch checkin users" in query_lower:
            return {
                "tool_needed": "GetEventCheckedInUsers",
                "confidence": 0.8,
                "reasoning": "Query contains keywords related to checked-in users"
            }
            
        # Default to RAG
        return {
            "tool_needed": "rag",
            "confidence": 0.6,
            "reasoning": "No specific tool keywords found, using RAG as fallback"
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query using the appropriate tool or RAG"""
        # Analyze the query
        analysis = self.analyze_query(query)
        tool_needed = analysis.get("tool_needed")
        
        # If RAG is needed, return analysis only
        if tool_needed == "rag":
            return {
                "source": "rag",
                "analysis": analysis,
                "result": None  # No result, will be handled by RAG system
            }
            
        # Execute the specified tool
        tool_result = self.mcp_client.execute_tool(tool_needed)
        
        if "error" in tool_result:
            # If tool execution fails, fall back to RAG
            return {
                "source": "rag_fallback",
                "analysis": analysis,
                "result": None,  # No result, will be handled by RAG system
                "tool_error": tool_result["error"]
            }
        
        # Format the tool result for user response
        formatted_result = self._format_tool_result(tool_needed, tool_result, query)
        
        return {
            "source": "tool",
            "tool": tool_needed,
            "analysis": analysis,
            "result": formatted_result
        }
    
    def _format_tool_result(self, tool_name: str, result: Dict[str, Any], query: str) -> str:
        """Format tool result for user-friendly response"""
        if tool_name == "GetEventRegisteredUsers":
            count = result.get("count", 0)
            
            # Check if query is asking for count or list
            if any(word in query.lower() for word in ["count", "how many", "number of"]):
                return f"There are {count} registered users for this event."
            else:
                # Just return the count in a nicely formatted way
                return f"Currently {count} users are registered for this event."
                
        elif tool_name == "GetEventCheckedInUsers":
            count = result.get("count", 0)
            
            # Check if query is asking for count or list
            if any(word in query.lower() for word in ["count", "how many", "number of"]):
                return f"There are {count} checked-in users at this event."
            else:
                # Just return the count in a nicely formatted way
                return f"Currently {count} users have checked in to this event."
        
        # Generic formatting for other tools
        return str(result)