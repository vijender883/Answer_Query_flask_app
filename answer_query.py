# answer_query.py
import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Import our custom modules
from mcp_client import MCPClient
from agent_system import EventAgent

# Load environment variables
load_dotenv()

# Get API keys and configurations from environment variables
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_index = os.getenv("PINECONE_INDEX")
mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

class AnswerQuery:
    def __init__(self, api_key, pinecone_api_key, pinecone_cloud, pinecone_region, index_name, agent):
        self.api_key = api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_cloud = pinecone_cloud
        self.pinecone_region = pinecone_region
        self.index_name = index_name
        self.agent = agent

        # Initialize Pinecone with new SDK
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            print("Pinecone initialized successfully with new SDK")
        except Exception as e:
            print(f"Pinecone initialization error: {e}")
            self.pc = None
            
        # Initialize Gemini client
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel("gemini-2.0-flash")
            print("Gemini initialized successfully")
        except Exception as e:
            print(f"Gemini client initialization error: {e}")
            self.client = None

        # Initialize Embeddings
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            print("Embeddings initialized successfully")
        except Exception as e:
            print(f"Embeddings initialization error: {e}")
            self.embeddings = None

        self.prompt_template = """
        You are a friendly Event Information Assistant named "Xylo". Your ONLY purpose is to answer questions based STRICTLY on the provided context information. You MUST NOT provide any information that is not explicitly stated in the context.

        CRITICAL RULES - NEVER VIOLATE THESE:
        1. ONLY use information that is explicitly stated in the provided context
        2. NEVER make assumptions or add details not in the context
        3. NEVER mention features, benefits, or details that are not specifically mentioned in the source content
        4. If information is not in the context, always say "I don't have that specific information in the course materials provided"
        5. DO NOT hallucinate or invent course details like mock interviews, specific timelines, or features unless explicitly mentioned
        6. DO NOT assume standard course features - only mention what's documented in the context
        7. If asked about something not covered in the context, politely redirect to contact information if available, or state the limitation

        RESPONSE FORMAT - You MUST respond in this exact JSON format:
        {{
            "answer": "Your response based ONLY on the provided context",
            "show_enroll": true or false,
            "suggested_questions": ["Question 1", "Question 2", "Question 3"]
        }}

        Guidelines for enrollment interest detection:
        - Set "show_enroll": true ONLY if the user:
          * Asks about course details that indicate genuine interest (pricing, schedule, enrollment process)
          * Shows enthusiasm about course content that IS mentioned in the context
          * Asks about certificates, outcomes, or benefits that ARE documented in the context
          * Asks follow-up questions showing continued interest in documented features
        
        - Set "show_enroll": false if the user:
          * Asks basic greetings without course interest
          * Shows disinterest or negative sentiment
          * Asks about things not covered in the context
          * Is just browsing without commitment signals

        Guidelines for suggested questions:
        - ONLY suggest questions about topics that ARE covered in the provided context
        - DO NOT suggest questions about features or topics not mentioned in the source material
        - Base suggestions on what information IS actually available in the context
        - Keep questions relevant to the documented course content only
        - Examples ONLY if context supports them: pricing (if mentioned), schedule (if mentioned), certificates (if mentioned)

        Response guidelines:
        1. Be warm and helpful while staying strictly within the provided information
        2. If context has limited information, acknowledge the limitation honestly
        3. NEVER fill in gaps with assumed or typical course information
        4. If asked about specific features (like mock interviews, career services, etc.), only confirm if explicitly mentioned in context
        5. Use phrases like "Based on the course information provided..." or "According to the course details..."
        6. If context is empty or very limited, be honest about the limited information available
        7. For greetings, be friendly but immediately guide toward topics covered in the context
        8. ALWAYS prioritize accuracy over completeness - better to say "I don't know" than to hallucinate

        CONTEXT INFORMATION (this is the ONLY source of truth):
        {context}

        PREVIOUS CONVERSATION:
        {previous_chats}

        CURRENT QUESTION: {question}

        REMEMBER: Your credibility depends on accuracy. Only use information explicitly stated in the context above. When in doubt, acknowledge the limitation rather than guess or assume.
        """

    def answer_question(self, query, previous_chats=""):
        """Process query using agent system first, then RAG if needed"""
        try:
            # First, let the agent analyze the query
            agent_response = self.agent.process_query(query)
            
            # If agent has a result from a tool, format it as JSON
            if agent_response["source"] == "tool" and agent_response["result"]:
                # Analyze interest based on query and previous chats
                show_enroll = self._analyze_enrollment_interest(query, previous_chats, agent_response["result"])
                # Generate suggested questions based on the query and response
                suggested_questions = self._generate_suggested_questions(query, agent_response["result"])
                
                return {
                    "answer": agent_response["result"],
                    "show_enroll": show_enroll,
                    "suggested_questions": suggested_questions,
                    "source": agent_response["source"],
                    "error": False
                }
                
            # If we need to use RAG, continue with normal processing
            # Check if required components are initialized
            if self.pc is None:
                return {"answer": "Pinecone database is not available.", "show_enroll": False, "error": True}
            if self.client is None:
                return {"answer": "Gemini language model is not available.", "show_enroll": False, "error": True}
            if self.embeddings is None:
                return {"answer": "Embedding model is not available.", "show_enroll": False, "error": True}

            # Get vector store
            try:
                index = self.pc.Index(self.index_name)
                vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=self.embeddings,
                    text_key="text"
                )
            except Exception as e:
                return {"answer": f"Error accessing vector store: {str(e)}", "show_enroll": False, "error": True}
                
            # Search for similar content
            results = vectorstore.similarity_search_with_score(query, k=5)

            # Process search results
            context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])
            if not context_text and results:
                context_text = "No specific details found in the documents for your query."
            elif not results:
                context_text = "No information found in the knowledge base for your query."

            # Generate prompt
            prompt_template_obj = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template_obj.format(
                context=context_text, 
                question=query, 
                previous_chats=previous_chats or "No previous conversation history."
            )

            # Generate response from LLM
            response_genai = self.client.generate_content(prompt)

            # Process LLM response
            raw_response = response_genai.text
            
            # Try to parse JSON response from LLM
            try:
                # Clean the response to extract JSON
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = raw_response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                    
                    answer = parsed_response.get("answer", raw_response)
                    show_enroll = parsed_response.get("show_enroll", False)
                    suggested_questions = parsed_response.get("suggested_questions", self._get_default_suggestions())
                else:
                    # Fallback if JSON parsing fails
                    answer = raw_response
                    show_enroll = self._analyze_enrollment_interest(query, previous_chats, raw_response)
                    suggested_questions = self._generate_suggested_questions(query, raw_response)
                    
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                answer = raw_response
                show_enroll = self._analyze_enrollment_interest(query, previous_chats, raw_response)
                suggested_questions = self._generate_suggested_questions(query, raw_response)
            
            # If we had a tool error, mention it
            if agent_response["source"] == "rag_fallback" and "tool_error" in agent_response:
                answer = f"I tried to use a specialized tool for this query but encountered an issue. Here's what I found in my general knowledge:\n\n{answer}"

            return {
                "answer": answer,
                "show_enroll": show_enroll,
                "suggested_questions": suggested_questions,
                "source": "rag",
                "error": False
            }
            
        except Exception as e:
            error_message = f"An error occurred during question answering: {str(e)}"
            if "permission" in str(e).lower() and "model" in str(e).lower():
                error_message += "\nPlease ensure the API key has permissions for the 'gemini-2.0-flash' model."

            # Print detailed error for server logs
            print(f"Detailed Answering Error: {e}")
            import traceback
            traceback.print_exc()

            return {
                "answer": error_message,
                "show_enroll": False,
                "suggested_questions": self._get_default_suggestions(),
                "error": True
            }

    def _generate_suggested_questions(self, query, response_text=""):
        """Generate contextual suggested questions based on the query and response, avoiding hallucinations"""
        query_lower = query.lower()
        
        # Define conservative question categories that don't assume specific features
        basic_info_questions = [
            "What topics are covered in this course?",
            "What will I learn from this course?",
            "Tell me more about the course content"
        ]
        
        logistics_questions = [
            "What are the course fees?",
            "When does the course start?",
            "How long is the course?"
        ]
        
        completion_questions = [
            "What do I get after completing the course?",
            "Are there any certificates provided?",
            "What are the course outcomes?"
        ]
        
        enrollment_questions = [
            "How can I enroll in this course?",
            "What are the requirements to join?",
            "How do I register for this course?"
        ]
        
        # Choose questions based on query context, but keep them general
        if any(word in query_lower for word in ['price', 'cost', 'fee', 'payment', 'money']):
            return logistics_questions
        elif any(word in query_lower for word in ['schedule', 'timing', 'start', 'batch', 'duration', 'long']):
            return completion_questions
        elif any(word in query_lower for word in ['certificate', 'certification', 'outcome', 'completion', 'finish']):
            return enrollment_questions
        elif any(word in query_lower for word in ['enroll', 'join', 'register', 'apply', 'signup']):
            return basic_info_questions
        elif any(word in query_lower for word in ['about', 'topic', 'curriculum', 'content', 'learn', 'cover']):
            return logistics_questions
        else:
            return basic_info_questions

    def _get_default_suggestions(self):
        """Return conservative default suggested questions that don't assume course features"""
        return [
            "What is this course about?",
            "What are the course details?",
            "How can I learn more about this course?"
        ]

    def _analyze_enrollment_interest(self, query, previous_chats, response_text=""):
        """Analyze if user shows interest in enrollment based on query and chat history"""
        # Convert to lowercase for analysis
        query_lower = query.lower()
        previous_lower = previous_chats.lower() if previous_chats else ""
        
        # Keywords that indicate enrollment interest
        interest_keywords = [
            'enroll', 'register', 'sign up', 'join', 'apply', 'admission',
            'price', 'cost', 'fee', 'payment', 'schedule', 'timing',
            'certificate', 'certification', 'duration', 'requirement',
            'prerequisite', 'benefit', 'outcome', 'career', 'job',
            'deadline', 'start date', 'when does', 'how to join',
            'interested', 'want to', 'would like', 'tell me more',
            'details about', 'more information'
        ]
        
        # Check current query
        query_has_interest = any(keyword in query_lower for keyword in interest_keywords)
        
        # Check previous chats
        history_has_interest = any(keyword in previous_lower for keyword in interest_keywords)
        
        # Additional context checks
        asking_followup = len(previous_chats.strip()) > 0 and ('?' in query or 'how' in query_lower or 'what' in query_lower)
        
        return query_has_interest or history_has_interest or asking_followup

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Check if required environment variables are set
if not api_key:
    print("WARNING: GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
if not pinecone_api_key or not pinecone_index:
    print("WARNING: PINECONE_API_KEY or PINECONE_INDEX not found. Please set them in your environment or .env file.")

# Initialize MCP client
mcp_client = MCPClient(mcp_server_url)

# Initialize the Agent System
event_agent = EventAgent(mcp_client)

# Initialize the answering system if environment variables are available
answer_system = None
if api_key and pinecone_api_key and pinecone_index:
    print("Initializing answer system...")
    answer_system = AnswerQuery(
        api_key=api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_cloud=pinecone_cloud,
        pinecone_region=pinecone_region,
        index_name=pinecone_index,
        agent=event_agent
    )
    print("Answer system initialization complete.")
else:
    print("MISSING ENV VARS. Answer system not initialized.")

@app.route('/api/answer', methods=['POST'])
def answer():
    # Validate request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    query = data.get('query')
    previous_chats = data.get('previous_chats', '')  # New field for chat history

    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    # Check if answer system is initialized
    if answer_system is None:
        return jsonify({"error": "Service unavailable due to configuration issues."}), 503

    # Get answer with previous chats context
    result = answer_system.answer_question(query, previous_chats)

    # Return response
    if result.get("error", False):
        return jsonify({
            "answer": result["answer"],
            "show_enroll": result.get("show_enroll", False)
        }), 500
    else:
        # Return JSON response with answer, show_enroll, and suggested_questions
        return jsonify({
            "answer": result["answer"],
            "show_enroll": result.get("show_enroll", False),
            "suggested_questions": result.get("suggested_questions", [])
        })

@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        "status": "ok",
        "gemini_api": api_key is not None,
        "pinecone_api": pinecone_api_key is not None,
        "pinecone_index": pinecone_index is not None,
        "answer_system_ready": answer_system is not None,
        "mcp_server": mcp_server_url,
        "mcp_tools_discovered": len(mcp_client.discover_tools()) > 0
    }

    if all([health_status["gemini_api"], health_status["pinecone_api"], health_status["pinecone_index"], health_status["answer_system_ready"]]):
        return jsonify(health_status), 200
    else:
        return jsonify(health_status), 503

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))

    # Start Flask app
    app.run(host='0.0.0.0', port=port, debug=False)