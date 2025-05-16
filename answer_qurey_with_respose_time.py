import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Load environment variables
load_dotenv()

# Get API keys and configurations from environment variables
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")  # Default to aws if not set
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")  # Default region if not set
pinecone_index = os.getenv("PINECONE_INDEX")

class AnswerQuery:
    def __init__(self, api_key, pinecone_api_key, pinecone_cloud, pinecone_region, index_name):
        self.api_key = api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_cloud = pinecone_cloud
        self.pinecone_region = pinecone_region
        self.index_name = index_name

        # Initialize Pinecone with new SDK
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
        except Exception as e:
            print(f"Pinecone initialization error: {e}")
            self.pc = None

        # Initialize Gemini client
        try:
            self.client = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            print(f"Gemini client initialization error: {e}")
            self.client = None

        # Initialize Embeddings
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
        except Exception as e:
            print(f"Embeddings initialization error: {e}")
            self.embeddings = None

        self.prompt_template = """
        You are a friendly Event Information Assistant. Your primary purpose is to answer questions about the event described in the provided context. You can also answer questions based on user-submitted resumes if they have been provided. Follow these guidelines:

        1. You can respond to basic greetings like "hi", "hello", or "how are you" in a warm, welcoming manner
        2. For event information or resume content, only provide details that are present in the context
        3. If information is not in the context, politely say "I'm sorry, I don't have that specific information" (for event) or "I'm sorry, I don't have that information from the resume" (for resume).
        4. Keep responses concise but conversational
        5. Do not make assumptions beyond what's explicitly stated in the context
        6. Always prioritize factual accuracy while maintaining a helpful tone
        7. Do not introduce information that isn't in the context
        8. If unsure about any information, acknowledge uncertainty rather than guess
        9. You may suggest a few general questions users might want to ask about the event
        10. Remember to maintain a warm, friendly tone in all interactions
        11. You should refer to yourself as "Event Bot"
        12. You should not greet if the user has not greeted to you

        Remember: While you can be conversational, your primary role is providing accurate information based on the context provided (event details and/or resume content).

        Context information (event details and/or resume content):
        {context}
        --------

        Now, please answer this question: {question}
        """

    def answer_question(self, query):
        """Use RAG with Google Gemini to answer a question based on retrieved context,
        and measure component response times."""
        vector_db_time = 0
        llm_time = 0
        raw_response = ""
        processed_response_text = ""

        try:
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
            start_time = time.time()
            results = vectorstore.similarity_search_with_score(query, k=5)  # k=5, can be tuned
            end_time = time.time()
            vector_db_time = end_time - start_time

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
            start_time = time.time()
            response_genai = self.client.generate_content(prompt)
            end_time = time.time()
            llm_time = end_time - start_time

            # Process LLM response
            raw_response = response_genai.text
            
            return {
                "text": raw_response,
                "vector_db_time": vector_db_time,
                "llm_time": llm_time,
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
                "vector_db_time": vector_db_time,
                "llm_time": llm_time,
                "error": True
            }

# Initialize Flask app
app = Flask(__name__)

# Check if required environment variables are set
if not api_key:
    print("WARNING: GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
if not pinecone_api_key or not pinecone_index:
    print("WARNING: PINECONE_API_KEY or PINECONE_INDEX not found. Please set them in your environment or .env file.")

# Initialize the answering system if environment variables are available
answer_system = None
if api_key and pinecone_api_key and pinecone_index:
    answer_system = AnswerQuery(
        api_key=api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_cloud=pinecone_cloud,
        pinecone_region=pinecone_region,
        index_name=pinecone_index
    )

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
        return jsonify({
            "error": "LLM answering system is not initialized. Check environment variables.",
            "text": "Service unavailable due to configuration issues."
        }), 503
    
    # Get answer
    result = answer_system.answer_question(query)
    
    # Return response
    if result.get("error", False):
        return jsonify({
            "error": True,
            "text": result["text"]
        }), 500
    else:
        return jsonify({
            "text": result["text"],
            "metadata": {
                "vector_db_time_ms": round(result["vector_db_time"] * 1000, 2),
                "llm_time_ms": round(result["llm_time"] * 1000, 2)
            }
        })

@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        "status": "ok",
        "gemini_api": api_key is not None,
        "pinecone_api": pinecone_api_key is not None,
        "pinecone_index": pinecone_index is not None,
        "answer_system_ready": answer_system is not None
    }
    
    if all(health_status.values()):
        return jsonify(health_status), 200
    else:
        return jsonify(health_status), 503

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Start Flask app
    app.run(host='0.0.0.0', port=port, debug=False)