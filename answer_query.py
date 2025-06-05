# answer_query.py
import os
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
        Here`s the optimized and clearly formatted version of your prompt without losing any context:
        ---
        You are **Event Bot** named "Xylo", a friendly assistant designed to answer questions about the event described in the provided context. You may also respond to questions based on user-submitted resumes, if available.

        **Guidelines to follow:**

        1. Respond warmly to greetings like "hi", "hello", or "how are you".
        2. Do not initiate greetings unless the user does i.e., don't greet the user unless they greet you.
        3. Only share information present in the context (event or resume).
        4. If something isn`t in the context:
            * For event questions, say: *“I'm sorry, I don't have that specific information.”*
            * For resume questions, say: *“I'm sorry, I don't have that information from the resume.”*
        5. Keep responses concise, friendly, and conversational.
        6. Do not assume or infer beyond the given context.
        7. Prioritize factual accuracy while being helpful.
        8. Never introduce information not in the context.
        9. If uncertain, acknowledge it—do not guess.
        10. Feel free to suggest general event-related questions the user might ask.
        11. Maintain a warm, approachable tone in every response.
        12. Always refer to yourself as **Event Bot**.
        13. Format and structure your answers properly for clarity.

        ---

        **Note:** Your role is to be helpful and conversational, but your primary focus is delivering accurate, context-based information.

        **Context (event and/or resume details):**
        {context}
        ---------

        **User Question:**
        {question}

        """

    def answer_question(self, query):
        """Process query using agent system first, then RAG if needed"""
        try:
            # First, let the agent analyze the query
            agent_response = self.agent.process_query(query)
            
            # If agent has a result from a tool, return it
            if agent_response["source"] == "tool" and agent_response["result"]:
                return {
                    "text": agent_response["result"],
                    "source": agent_response["source"],
                    "error": False
                }
                
            # If we need to use RAG, continue with normal processing
            # Check if required components are initialized
            if self.pc is None:
                return {"text": "Pinecone database is not available.", "error": True}
            if self.client is None:
                return {"text": "Gemini language model is not available.", "error": True}
            if self.embeddings is None:
                return {"text": "Embedding model is not available.", "error": True}

            # Get vector store
            try:
                index = self.pc.Index(self.index_name)
                vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=self.embeddings,
                    text_key="text"
                )
            except Exception as e:
                return {"text": f"Error accessing vector store: {str(e)}", "error": True}
                
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
            prompt = prompt_template_obj.format(context=context_text, question=query)

            # Generate response from LLM
            response_genai = self.client.generate_content(prompt)

            # Process LLM response
            raw_response = response_genai.text
            
            # If we had a tool error, mention it
            if agent_response["source"] == "rag_fallback" and "tool_error" in agent_response:
                raw_response = f"I tried to use a specialized tool for this query but encountered an issue. Here's what I found in my general knowledge:\n\n{raw_response}"

            return {
                "text": raw_response,
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
                "text": error_message,
                "error": True
            }

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

    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    # Check if answer system is initialized
    if answer_system is None:
        return jsonify("Service unavailable due to configuration issues."), 503

    # Get answer
    result = answer_system.answer_question(query)

    # Return response
    if result.get("error", False):
        return jsonify(result["text"]), 500
    else:
        # Only return the text without any metadata
        return jsonify(result["text"])

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