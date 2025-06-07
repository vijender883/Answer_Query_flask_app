#!/usr/bin/env python3
"""
Standalone Event Bot Question Answering System
A simplified RAG system using Gemini LLM and Pinecone vector database
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EventBot:
    """
    Standalone Event Bot for question answering using RAG with Gemini and Pinecone
    """
    
    def __init__(self, gemini_api_key=None, pinecone_api_key=None, pinecone_index=None):
        """
        Initialize the Event Bot
        
        Args:
            gemini_api_key (str): Google Gemini API key
            pinecone_api_key (str): Pinecone API key  
            pinecone_index (str): Pinecone index name
        """
        # Load environment variables if not provided
        load_dotenv()
        
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_index = pinecone_index or os.getenv("PINECONE_INDEX")
        
        # Validate required credentials
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required")
        if not self.pinecone_index:
            raise ValueError("PINECONE_INDEX is required")
        
        # Initialize components
        self._initialize_pinecone()
        self._initialize_gemini()
        self._initialize_embeddings()
        self._setup_prompt_template()
        
    def _initialize_pinecone(self):
        """Initialize Pinecone client and vector store"""
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.pinecone_index)
            print("✓ Pinecone initialized successfully")
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone: {e}")
    
    def _initialize_gemini(self):
        """Initialize Gemini LLM client"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.llm = genai.GenerativeModel("gemini-2.0-flash")
            print("✓ Gemini LLM initialized successfully")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {e}")
    
    def _initialize_embeddings(self):
        """Initialize Google embeddings for vector search"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key
            )
            
            # Create vector store
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="text"
            )
            print("✓ Embeddings and vector store initialized successfully")
        except Exception as e:
            raise Exception(f"Failed to initialize embeddings: {e}")
    
    def _setup_prompt_template(self):
        """Setup the prompt template for the event bot"""
        self.prompt_template = """
You are a friendly Event Information Assistant named "Event Bot". Your primary purpose is to answer questions about the event described in the provided context. You can also answer questions based on user-submitted resumes if they have been provided. Follow these guidelines:

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
13. Format and structure the answer properly.

Remember: While you can be conversational, your primary role is providing accurate information based on the context provided (event details and/or resume content).

Context information (event details and/or resume content):
{context}
--------

Now, please answer this question: {question}
"""

    def answer_question(self, question, top_k=5):
        """
        Answer a question using RAG (Retrieval-Augmented Generation)
        
        Args:
            question (str): The user's question
            top_k (int): Number of similar documents to retrieve
            
        Returns:
            dict: Response containing answer text and metadata
        """
        try:
            # Retrieve relevant documents from vector store
            results = self.vectorstore.similarity_search_with_score(question, k=top_k)
            
            # Process search results to create context
            if results:
                context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])
                if not context_text:
                    context_text = "No specific details found in the documents for your query."
            else:
                context_text = "No information found in the knowledge base for your query."
            
            # Create prompt using template
            prompt_template_obj = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template_obj.format(context=context_text, question=question)
            
            # Generate response using Gemini
            response = self.llm.generate_content(prompt)
            answer_text = response.text
            
            return {
                "answer": answer_text,
                "context_found": len(results) > 0,
                "num_sources": len(results),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            error_message = f"Error answering question: {str(e)}"
            print(f"Error details: {e}")
            
            return {
                "answer": error_message,
                "context_found": False,
                "num_sources": 0,
                "success": False,
                "error": str(e)
            }
    
    def get_similar_documents(self, query, top_k=5):
        """
        Get similar documents for a query (useful for debugging)
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            list: List of similar documents with scores
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            return [
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def health_check(self):
        """
        Check if all components are working properly
        
        Returns:
            dict: Health status of all components
        """
        status = {
            "gemini_api": False,
            "pinecone_connection": False,
            "embeddings": False,
            "vector_store": False,
            "overall_health": False
        }
        
        # Test Gemini
        try:
            test_response = self.llm.generate_content("Say 'OK' if you can respond")
            status["gemini_api"] = "OK" in test_response.text
        except:
            status["gemini_api"] = False
        
        # Test Pinecone
        try:
            index_stats = self.index.describe_index_stats()
            status["pinecone_connection"] = True
        except:
            status["pinecone_connection"] = False
        
        # Test Embeddings
        try:
            test_embedding = self.embeddings.embed_query("test")
            status["embeddings"] = len(test_embedding) > 0
        except:
            status["embeddings"] = False
        
        # Test Vector Store
        try:
            test_search = self.vectorstore.similarity_search("test", k=1)
            status["vector_store"] = True
        except:
            status["vector_store"] = False
        
        # Overall health
        status["overall_health"] = all([
            status["gemini_api"],
            status["pinecone_connection"], 
            status["embeddings"],
            status["vector_store"]
        ])
        
        return status


def main():
    """
    Example usage of the EventBot
    """
    try:
        # Initialize the bot
        print("Initializing Event Bot...")
        bot = EventBot()
        
        # Health check
        print("\nPerforming health check...")
        health = bot.health_check()
        print(f"Health Status: {health}")
        
        if not health["overall_health"]:
            print("❌ System not healthy. Please check your configuration.")
            return
        
        print("\n✅ Event Bot is ready!")
        
        # Interactive loop
        print("\n" + "="*50)
        print("Event Bot - Ask me anything about the event!")
        print("Type 'quit' or 'exit' to stop")
        print("="*50)
        
        while True:
            try:
                question = input("\n🤖 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🔍 Searching for relevant information...")
                result = bot.answer_question(question)
                
                if result["success"]:
                    print(f"\n📝 Event Bot: {result['answer']}")
                    print(f"\n📊 Sources found: {result['num_sources']}")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
        
    except Exception as e:
        print(f"❌ Failed to initialize Event Bot: {e}")
        print("\nPlease check that you have:")
        print("1. Set GEMINI_API_KEY in your environment")
        print("2. Set PINECONE_API_KEY in your environment") 
        print("3. Set PINECONE_INDEX in your environment")
        print("4. Installed required packages: pip install google-generativeai pinecone langchain langchain-pinecone langchain-google-genai python-dotenv")


if __name__ == "__main__":
    main()