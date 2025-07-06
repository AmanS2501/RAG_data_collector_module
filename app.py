# import streamlit as st
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_community.llms import HuggingFacePipeline
# import os
# import sys
# import torch
# from huggingface_hub import login
# import re
# import time
# from datetime import datetime
# import logging

# # Configuration
# PROJECT_ROOT = "/content/drive/MyDrive"
# FINETUNED_MODEL_PATH = "/content/drive/MyDrive/lora_llama_finetuned"
# LOG_FILE = os.path.join(PROJECT_ROOT, "chatbot.log")

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(LOG_FILE),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize session state with more comprehensive tracking
# def initialize_session_state():
#     """Initialize all session state variables"""
#     if 'rag_chain' not in st.session_state:
#         st.session_state.rag_chain = None
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'model_loaded' not in st.session_state:
#         st.session_state.model_loaded = False
#     if 'initialization_time' not in st.session_state:
#         st.session_state.initialization_time = None
#     if 'total_queries' not in st.session_state:
#         st.session_state.total_queries = 0
#     if 'model_type' not in st.session_state:
#         st.session_state.model_type = None

# initialize_session_state()

# def validate_environment():
#     """Validate required directories and files exist"""
#     required_paths = [
#         PROJECT_ROOT,
#         os.path.join(PROJECT_ROOT, "vector_store")
#     ]
    
#     for path in required_paths:
#         if not os.path.exists(path):
#             st.error(f"❌ Required path does not exist: {path}")
#             return False
#     return True

# @st.cache_resource
# def initialize_rag_system():
#     """Initialize the RAG system with enhanced error handling and logging"""
    
#     start_time = time.time()
#     logger.info("Starting RAG system initialization")
    
#     try:
#         # Validate environment first
#         if not validate_environment():
#             raise Exception("Environment validation failed")
        
#         # Enhanced HuggingFace authentication
#         hf_token = None
#         try:
#             # Try multiple environment variable names
#             hf_token = (os.getenv("HUGGINGFACE_TOKEN") or 
#                        os.getenv("HF_TOKEN") or 
#                        os.getenv("HUGGING_FACE_HUB_TOKEN"))
            
#             if hf_token:
#                 login(token=hf_token)
#                 st.success("✅ HuggingFace authentication successful")
#                 logger.info("HuggingFace authentication successful")
#             else:
#                 st.warning("⚠️ No HuggingFace token found. Using public models only.")
#                 logger.warning("No HuggingFace token found")
                
#         except Exception as e:
#             st.warning(f"⚠️ HuggingFace login failed: {str(e)}")
#             logger.warning(f"HuggingFace login failed: {str(e)}")
        
#         # Load environment variables
#         env_path = os.path.join(PROJECT_ROOT, ".env")
#         if os.path.exists(env_path):
#             load_dotenv(dotenv_path=env_path)
#             logger.info("Environment variables loaded")
        
#         # Initialize embeddings with error handling
#         try:
#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
#             )
#             logger.info("Embeddings model loaded successfully")
#         except Exception as e:
#             logger.error(f"Error loading embeddings: {str(e)}")
#             raise
        
#         # Load vector database with validation
#         vector_store_path = os.path.join(PROJECT_ROOT, "vector_store")
#         if not os.path.exists(vector_store_path):
#             raise Exception(f"Vector store not found at {vector_store_path}")
        
#         vectorstore = Chroma(
#             persist_directory=vector_store_path,
#             embedding_function=embeddings
#         )
        
#         # Validate vector store has documents
#         collection_count = len(vectorstore.get()['ids'])
#         if collection_count == 0:
#             st.warning("⚠️ Vector store appears to be empty")
#             logger.warning("Vector store is empty")
#         else:
#             st.info(f"📚 Vector store loaded with {collection_count} documents")
#             logger.info(f"Vector store loaded with {collection_count} documents")
        
#         retriever = vectorstore.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 5}
#         )
        
#         # Enhanced model loading with fallback strategy
#         model_loaded = False
#         model_type = None
        
#         # Try loading fine-tuned model first
#         if os.path.exists(FINETUNED_MODEL_PATH):
#             try:
#                 logger.info("Attempting to load fine-tuned model")
#                 tokenizer = AutoTokenizer.from_pretrained(
#                     FINETUNED_MODEL_PATH,
#                     trust_remote_code=True,
#                     token=hf_token
#                 )
                
#                 model = AutoModelForCausalLM.from_pretrained(
#                     FINETUNED_MODEL_PATH,
#                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                     trust_remote_code=True,
#                     token=hf_token,
#                     low_cpu_mem_usage=True
#                 )
                
#                 model_loaded = True
#                 model_type = "Fine-tuned Llama"
#                 st.success("✅ Fine-tuned model loaded successfully!")
#                 logger.info("Fine-tuned model loaded successfully")
                
#             except Exception as e:
#                 st.error(f"❌ Error loading fine-tuned model: {str(e)}")
#                 logger.error(f"Error loading fine-tuned model: {str(e)}")
        
#         # Fallback to base model if fine-tuned model fails
#         if not model_loaded:
#             try:
#                 st.info("🔄 Loading base Llama model...")
#                 logger.info("Loading base Llama model")
                
#                 base_model_name = "meta-llama/Llama-3.2-1B"
#                 tokenizer = AutoTokenizer.from_pretrained(
#                     base_model_name,
#                     trust_remote_code=True,
#                     token=hf_token
#                 )
                
#                 model = AutoModelForCausalLM.from_pretrained(
#                     FINETUNED_MODEL_PATH,
#                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                     trust_remote_code=True,
#                     token=hf_token,
#                     low_cpu_mem_usage=True
#                 )
                
#                 model_loaded = True
#                 model_type = "Base Llama-3.2-1B"
#                 st.warning("⚠️ Using base Llama model")
#                 logger.info("Base Llama model loaded successfully")
                
#             except Exception as e:
#                 st.error(f"❌ Error loading base model: {str(e)}")
#                 logger.error(f"Error loading base model: {str(e)}")
#                 raise e
        
#         if not model_loaded:
#             raise Exception("Failed to load any model")
        
#         # Setup tokenizer
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#             tokenizer.pad_token_id = tokenizer.eos_token_id
        
#         # Create optimized pipeline
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=500,
#             temperature=1.0,
#             repetition_penalty=1.15,
#             do_sample=True,
#             top_p=0.85,
#             top_k=40,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             return_full_text=True
#         )

        
#         llm = HuggingFacePipeline(pipeline=pipe)
        
#         # Enhanced prompt template
#         template = """You are a helpful and knowledgeable assistant. Use the provided context to answer the user's question accurately and comprehensively.

# Context Information:
# {context}

# User Question: {question}

# Instructions:
# - Answer based primarily on the provided context
# - If the context doesn't contain sufficient information, clearly state this
# - Provide specific details and examples when available
# - Keep your response focused and relevant
# - Be helpful and informative

# Response:"""

#         prompt = PromptTemplate.from_template(template)
        
#         # Create RAG chain
#         rag_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff",
#             chain_type_kwargs={"prompt": prompt},
#             return_source_documents=True,
#             verbose=False
#         )
        
#         # Update session state
#         st.session_state.model_type = model_type
#         st.session_state.model_loaded = True
#         st.session_state.initialization_time = time.time() - start_time
        
#         logger.info(f"RAG system initialized successfully in {st.session_state.initialization_time:.2f} seconds")
#         return rag_chain
        
#     except Exception as e:
#         logger.error(f"RAG system initialization failed: {str(e)}")
#         raise e

# def format_response(response_text):
#     """Enhanced response formatting with better cleaning"""
#     try:
#         if not response_text or not isinstance(response_text, str):
#             return "I apologize, but I couldn't generate a proper response."

#         # Remove common prefixes
#         prefixes_to_remove = ["Answer:", "Response:", "Assistant:", "Bot:"]
#         for prefix in prefixes_to_remove:
#             if response_text.startswith(prefix):
#                 response_text = response_text[len(prefix):].strip()

#         # Clean up the text
#         cleaned = response_text.replace("\n", " ").strip()

#         # Remove excessive whitespace
#         cleaned = re.sub(r'\s+', ' ', cleaned)

#         # Remove repetitive patterns
#         words = cleaned.split()
#         if len(words) > 10:
#             for i in range(len(words) - 5):
#                 phrase = ' '.join(words[i:i+3])
#                 remaining_text = ' '.join(words[i+3:])
#                 if phrase in remaining_text:
#                     cleaned = ' '.join(words[:i+3])
#                     break

#         return cleaned if cleaned else "I apologize, but I couldn't generate a proper response."
    
#     except Exception as e:
#         return f"I apologize, but I encountered an error while formatting the response: {str(e)}"



# def log_interaction(question, answer, sources_count):
#     """Log user interactions for analytics"""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     logger.info(f"Query: {question[:100]}... | Response length: {len(answer)} | Sources: {sources_count}")

# def main():
#     # Enhanced page configuration
#     st.set_page_config(
#         page_title="Advanced RAG LLM Chatbot",
#         page_icon="🤖",
#         layout="wide",
#         initial_sidebar_state="expanded",
#         menu_items={
#             'Get Help': None,
#             'Report a bug': None,
#             'About': "Advanced RAG Chatbot powered by Llama and LangChain"
#         }
#     )
    
#     # Custom CSS for better styling
#     st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .status-good { color: #28a745; }
#     .status-warning { color: #ffc107; }
#     .status-error { color: #dc3545; }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Header
#     st.markdown('<h1 class="main-header">🤖 Advanced RAG LLM Chatbot</h1>', unsafe_allow_html=True)
#     st.markdown("---")
    
#     # Sidebar with enhanced controls
#     with st.sidebar:
#         st.header("⚙️ System Controls")
        
#         # Authentication section
#         st.subheader("🔑 Authentication")
#         hf_token_input = st.text_input(
#             "HuggingFace Token:",
#             type="password",
#             help="Optional: Enter your HuggingFace token for accessing gated models",
#             placeholder="hf_xxxxxxxxxxxxxxxxx"
#         )
        
#         if hf_token_input:
#             os.environ["HUGGINGFACE_TOKEN"] = hf_token_input
#             st.success("✅ Token configured!")
        
#         # System initialization
#         col1, col2 = st.columns(2)
#         with col1:
#             init_button = st.button("🚀 Initialize", type="primary", use_container_width=True)
#         with col2:
#             clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
        
#         if init_button:
#             with st.spinner("🔄 Initializing system..."):
#                 try:
#                     st.session_state.rag_chain = initialize_rag_system()
#                     st.success("✅ System ready!")
#                     st.balloons()
#                 except Exception as e:
#                     st.error(f"❌ Initialization failed: {str(e)}")
        
#         if clear_button:
#             st.session_state.chat_history = []
#             st.success("🧹 Chat cleared!")
#             st.rerun()
        
#         # System status
#         st.subheader("📊 System Status")
        
#         if st.session_state.rag_chain and st.session_state.model_loaded:
#             st.markdown('<p class="status-good">🟢 System: Active</p>', unsafe_allow_html=True)
#             st.info(f"**Model:** {st.session_state.model_type}")
#             if st.session_state.initialization_time:
#                 st.info(f"**Init Time:** {st.session_state.initialization_time:.1f}s")
#         else:
#             st.markdown('<p class="status-warning">🟡 System: Not Ready</p>', unsafe_allow_html=True)
        
#         # Usage statistics
#         st.subheader("📈 Usage Stats")
#         st.metric("Total Queries", st.session_state.total_queries)
#         st.metric("Chat Messages", len(st.session_state.chat_history))
        
#         # System information
#         # st.subheader("ℹ️ System Info")
#         # gpu_available = torch.cuda.is_available()
#         # gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        
#         # st.text(f"GPU Available: {'Yes' if gpu_available else 'No'}")
#         # if gpu_available:
#         #     st.text(f"GPU: {gpu_name}")
#         # st.text(f"PyTorch: {torch.__version__}")
        
#         # Quick actions
#         st.subheader("🚀 Quick Start")
#         sample_questions = [
#             "What books are available in the collection?",
#             "Recommend science fiction novels",
#             "What are the main topics covered?",
#             "Tell me about classic literature",
#             "What fantasy books do you recommend?"
#         ]
        
#         for i, question in enumerate(sample_questions):
#             if st.button(question, key=f"sample_{i}", use_container_width=True):
#                 st.session_state.quick_question = question
#                 st.rerun()
    
#     # Main interface
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         st.subheader("💬 Chat Interface")
        
#         # Check system status
#         if not st.session_state.rag_chain:
#             st.warning("⚠️ Please initialize the system using the sidebar controls.")
#             st.info("💡 Click the **Initialize** button to load the RAG system.")
#         else:
#             # Question input with better UX
#             question = st.text_input(
#                 "Ask your question:",
#                 placeholder="What would you like to know?",
#                 key="question_input",
#                 value=st.session_state.get("quick_question", "")
#             )
            
#             # Clear quick question after setting
#             if "quick_question" in st.session_state:
#                 del st.session_state.quick_question
            
#             # Enhanced submit handling
#             col_submit, col_clear_input = st.columns([3, 1])
#             with col_submit:
#                 submit_clicked = st.button("🔍 Get Answer", type="primary", use_container_width=True)
#             with col_clear_input:
#                 if st.button("❌ Clear", use_container_width=True):
#                     st.session_state.question_input = ""
#                     st.rerun()
            
#             # Process question
#             if submit_clicked and question and question.strip():
#                 process_question(question.strip())
        
#         # Enhanced chat history display
#         display_chat_history()
    
#     with col2:
#         st.subheader("📊 Analytics")
#         display_analytics()

# def process_question(question):
#     """Process user question with enhanced error handling"""
#     # Add to chat history
#     st.session_state.chat_history.append({
#         "role": "user", 
#         "content": question,
#         "timestamp": datetime.now().strftime("%H:%M:%S")
#     })
    
#     # Generate response
#     with st.spinner("🤔 Generating response..."):
#         try:
#             start_time = time.time()
#             result = st.session_state.rag_chain.invoke({"query": question})
#             response_time = time.time() - start_time
            
#             raw_answer = result.get('result', '')
#             sources = result.get("source_documents", [])
            
#             # Format response
#             cleaned_answer = format_response(raw_answer)
            
#             # Add response to chat history
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "content": cleaned_answer,
#                 "sources": sources,
#                 "timestamp": datetime.now().strftime("%H:%M:%S"),
#                 "response_time": response_time
#             })
            
#             # Update statistics
#             st.session_state.total_queries += 1
            
#             # Log interaction
#             log_interaction(question, cleaned_answer, len(sources))
            
#         except Exception as e:
#             error_msg = f"I apologize, but I encountered an error: {str(e)}"
#             st.error(f"❌ {error_msg}")
            
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "content": error_msg,
#                 "sources": [],
#                 "timestamp": datetime.now().strftime("%H:%M:%S"),
#                 "response_time": 0
#             })
            
#             logger.error(f"Error processing question: {str(e)}")
    
#     st.rerun()

# def display_chat_history():
#     """Display chat history with enhanced formatting"""
#     if not st.session_state.chat_history:
#         st.info("💭 Start a conversation by asking a question!")
#         return
    
#     st.subheader("📝 Conversation History")
    
#     for i, message in enumerate(st.session_state.chat_history):
#         timestamp = message.get("timestamp", "")
        
#         if message["role"] == "user":
#             st.markdown(f"### 👤 You ({timestamp}):")
#             st.info(message["content"])
        
#         elif message["role"] == "assistant":
#             response_time = message.get("response_time", 0)
#             st.markdown(f"### 🤖 Assistant ({timestamp}):")
#             if response_time > 0:
#                 st.caption(f"⏱️ Response time: {response_time:.2f}s")
            
#             st.success(message["content"])
            
#             # Enhanced source display
#             sources = message.get("sources", [])
#             if sources:
#                 with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
#                     for j, doc in enumerate(sources):
#                         source = doc.metadata.get('source', 'Unknown')
#                         content = doc.page_content
                        
#                         st.markdown(f"**📄 Source {j+1}:** `{source}`")
                        
#                         # Show content preview with better formatting
#                         preview = content[:400] + "..." if len(content) > 400 else content
#                         st.text_area(
#                             f"Preview:",
#                             preview,
#                             height=100,
#                             key=f"source_{i}_{j}",
#                             disabled=True
#                         )
                        
#                         if j < len(sources) - 1:
#                             st.divider()
        
#         if i < len(st.session_state.chat_history) - 1:
#             st.markdown("---")

# def display_analytics():
#     """Display analytics and statistics"""
#     if not st.session_state.chat_history:
#         st.info("📊 Analytics will appear after conversations")
#         return
    
#     # Basic metrics
#     total_messages = len(st.session_state.chat_history)
#     user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
#     bot_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("👤 Questions", user_messages)
#     with col2:
#         st.metric("🤖 Responses", bot_messages)
    
#     # Response time analytics
#     response_times = [
#         m.get("response_time", 0) 
#         for m in st.session_state.chat_history 
#         if m["role"] == "assistant" and m.get("response_time", 0) > 0
#     ]
    
#     if response_times:
#         avg_response_time = sum(response_times) / len(response_times)
#         st.metric("⏱️ Avg Response", f"{avg_response_time:.1f}s")
        
#         # Response time chart
#         if len(response_times) > 1:
#             st.line_chart(response_times)
    
#     # Source usage
#     all_sources = []
#     for message in st.session_state.chat_history:
#         if message["role"] == "assistant" and message.get("sources"):
#             all_sources.extend([s.metadata.get('source', 'Unknown') for s in message["sources"]])
    
#     if all_sources:
#         st.subheader("📚 Source Usage")
#         source_counts = {}
#         for source in all_sources:
#             source_counts[source] = source_counts.get(source, 0) + 1
        
#         # Show top sources
#         sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
#         for source, count in sorted_sources:
#             st.text(f"{count}× {os.path.basename(source)}")

# if __name__ == "__main__":
#     main()


import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import os
import torch
from huggingface_hub import login
from datetime import datetime
import logging
import re

# Configuration
PROJECT_ROOT = "/content/drive/MyDrive"
LOG_FILE = os.path.join(PROJECT_ROOT, "chatbot.log")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Session state setup
def initialize_session_state():
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'initialization_time' not in st.session_state:
        st.session_state.initialization_time = None
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None

initialize_session_state()


# Cleaning function
def format_response(response_text):
    try:
        if not response_text or not isinstance(response_text, str):
            return "I apologize, but I couldn't generate a proper response."

        prefixes_to_remove = ["Answer:", "Response:", "Assistant:", "Bot:"]
        for prefix in prefixes_to_remove:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()

        cleaned = response_text.replace("\n", " ").strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)

        words = cleaned.split()
        if len(words) > 10:
            for i in range(len(words) - 5):
                phrase = ' '.join(words[i:i+3])
                remaining_text = ' '.join(words[i+3:])
                if phrase in remaining_text:
                    cleaned = ' '.join(words[:i+3])
                    break

        return cleaned if cleaned else "I apologize, but I couldn't generate a proper response."

    except Exception as e:
        return f"I apologize, but I encountered an error while formatting the response: {str(e)}"


def initialize_rag_system():
    start_time = time.time()
    logger.info("Starting RAG system initialization")

    try:
        # Hugging Face Token
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or ""

        if hf_token:
            login(token=hf_token)
            st.success("✅ HuggingFace authentication successful")
            logger.info("HuggingFace login successful")

        load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

        # Load vector store
        vector_store_path = os.path.join(PROJECT_ROOT, "vector_store")
        if not os.path.exists(vector_store_path):
            raise Exception(f"Vector store not found at {vector_store_path}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        vectorstore = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Load base model
        base_model_name = "google/gemma-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            token=hf_token
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.2,
            repetition_penalty=1.15,
            do_sample=True,
            top_p=0.85,
            top_k=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        template = """
You are a precise and trustworthy assistant. Use only the provided context to answer the user's question.

If the answer is not found in the context, say "I couldn't find the answer in the provided sources."

Context:
{context}

Question:
{question}

Answer:
"""

        prompt = PromptTemplate.from_template(template)

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )

        st.session_state.model_type = "google/gemma-2b-it"
        st.session_state.model_loaded = True
        st.session_state.initialization_time = time.time() - start_time

        logger.info(f"RAG system initialized in {st.session_state.initialization_time:.2f} seconds")
        return rag_chain

    except Exception as e:
        logger.error(f"RAG system initialization failed: {str(e)}")
        raise e


def log_interaction(question, answer, sources_count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Query: {question[:100]}... | Response length: {len(answer)} | Sources: {sources_count}")


def process_question(question):
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

    with st.spinner("🤔 Generating response..."):
        try:
            start_time = time.time()
            result = st.session_state.rag_chain.invoke({"query": question})
            response_time = time.time() - start_time

            raw_answer = result.get('result', '')
            sources = result.get("source_documents", [])
            cleaned_answer = format_response(raw_answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": cleaned_answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "response_time": response_time
            })

            st.session_state.total_queries += 1
            log_interaction(question, cleaned_answer, len(sources))

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            st.error(f"❌ {error_msg}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "response_time": 0
            })

    st.rerun()


def display_chat_history():
    st.subheader("📝 Conversation History")

    if not st.session_state.chat_history:
        st.info("💭 Start a conversation by asking a question!")
        return

    for i, message in enumerate(st.session_state.chat_history):
        timestamp = message.get("timestamp", "")
        if message["role"] == "user":
            st.markdown(f"**👤 You ({timestamp}):**")
            st.info(message["content"])
        elif message["role"] == "assistant":
            response_time = message.get("response_time", 0)
            st.markdown(f"**🤖 Assistant ({timestamp}):**")
            if response_time > 0:
                st.caption(f"⏱️ Response time: {response_time:.2f}s")
            st.success(message["content"])

            sources = message.get("sources", [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                    for j, doc in enumerate(sources):
                        source = doc.metadata.get('source', 'Unknown')
                        content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                        st.markdown(f"**Source {j+1}:** `{source}`")
                        st.text_area("Preview:", content, height=100, key=f"source_{i}_{j}", disabled=True)


def main():
    st.set_page_config(page_title="Base Llama RAG Chatbot", layout="wide")

    st.title("🤖 RAG Chatbot")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ System Controls")

        hf_token_input = st.text_input(
            "HuggingFace Token:",
            type="password",
            help="Optional: Enter token if model is gated",
            placeholder="hf_xxx..."
        )

        if hf_token_input:
            os.environ["HUGGINGFACE_TOKEN"] = hf_token_input
            st.success("✅ Token configured!")

        if st.button("🚀 Initialize"):
            with st.spinner("Initializing system..."):
                try:
                    st.session_state.rag_chain = initialize_rag_system()
                    st.success("✅ System ready!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.success("🧹 Chat cleared!")
            st.rerun()

        st.markdown("### 📊 Status")
        if st.session_state.model_loaded:
            st.info(f"**Model:** {st.session_state.model_type}")
            st.info(f"**Init Time:** {st.session_state.initialization_time:.1f}s")
            st.markdown("🟢 **System: Active**")
        else:
            st.markdown("🟡 **System: Not Ready**")

        st.metric("Total Queries", st.session_state.total_queries)
        st.metric("Chat Messages", len(st.session_state.chat_history))

    st.subheader("💬 Ask a Question")
    if not st.session_state.rag_chain:
        st.warning("⚠️ Please initialize the system from the sidebar.")
    else:
        question = st.text_input("What would you like to know?")
        if st.button("🔍 Get Answer") and question.strip():
            process_question(question.strip())

    display_chat_history()


if __name__ == "__main__":
    import time
    main()
