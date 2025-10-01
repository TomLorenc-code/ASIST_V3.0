"""
AUTONOMOUS DOCUMENT PROCESSING AGENT
====================================

This is a complete autonomous AI agent that:
1. Takes documents from a folder
2. Breaks them into smaller chunks
3. Converts chunks into numerical vectors (embeddings)
4. Stores everything in a vector database
5. Makes all decisions automatically using OLLAMA LLaMA

The agent acts like a smart assistant that can process documents without human intervention.

Author: AI Assistant
Version: 1.0
Date: 2025
"""

# ==============================================================================
# STEP 1: IMPORT ALL REQUIRED LIBRARIES AND PACKAGES
# ==============================================================================

import os                    # For file and folder operations
import json                  # For handling JSON data
import time                  # For adding delays and timestamps
import uuid                  # For generating unique IDs
from typing import Dict, List, Any, Optional, Union  # For type hints (better code documentation)
from dataclasses import dataclass  # For creating data structures
from enum import Enum        # For creating status categories
import numpy as np           # For numerical operations
import requests             # For making HTTP requests to OLLAMA
from pathlib import Path    # For better file path handling
import glob                 # For finding files with patterns
import logging              # For detailed logging

# Vector and embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers imported successfully")
except ImportError:
    print("❌ Please install: pip install sentence-transformers")
    exit(1)

try:
    import chromadb
    print("✅ ChromaDB imported successfully")
except ImportError:
    print("❌ Please install: pip install chromadb")
    exit(1)

print("📦 All required packages imported successfully!")
print("="*80)

# ==============================================================================
# STEP 2: CONFIGURATION SETTINGS
# ==============================================================================

class Config:
    """
    Configuration settings for our autonomous agent
    Think of this as the 'settings panel' for our agent
    """
    
    # OLLAMA Settings (This is where we configure our AI brain)
    OLLAMA_BASE_URL = "http://localhost:11434"  # Where OLLAMA is running
    OLLAMA_MODEL = "llama3.2"  # Which LLaMA model to use
    
    # Document Processing Settings
    INPUT_FOLDER = "./documents"          # Where to find documents to process
    SUPPORTED_FORMATS = [".txt", ".md", ".html"]   # What file types we can handle
    
    # Chunking Settings (How we break documents into pieces)
    CHUNK_SIZE = 1000        # Maximum characters per chunk
    CHUNK_OVERLAP = 200      # How much chunks should overlap
    
    # Vector Database Settings
    VECTOR_DB_PATH = "./vector_db"       # Where to store our vector database
    EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Which model to convert text to numbers
    
    # Agent Settings
    MAX_ITERATIONS = 20      # Maximum decision loops to prevent infinite loops
    DELAY_BETWEEN_STEPS = 1  # Seconds to wait between steps (for demonstration)

print(f"⚙️ Agent configured with:")
print(f"   - OLLAMA URL: {Config.OLLAMA_BASE_URL}")
print(f"   - Model: {Config.OLLAMA_MODEL}")
print(f"   - Input folder: {Config.INPUT_FOLDER}")
print(f"   - Vector DB path: {Config.VECTOR_DB_PATH}")

# ==============================================================================
# STEP 3: DEFINE DATA STRUCTURES AND STATUS TRACKING
# ==============================================================================

class TaskStatus(Enum):
    """
    Different states our agent can be in
    Think of this like a traffic light system for tracking progress
    """
    PENDING = "pending"           # Task is waiting to start
    LOADING_DOCUMENTS = "loading_documents"    # Finding and reading documents
    CHUNKING = "chunking"         # Breaking documents into pieces
    GENERATING_EMBEDDINGS = "generating_embeddings"  # Converting to numbers
    STORING_VECTORS = "storing_vectors"        # Saving to database
    COMPLETED = "completed"       # All done successfully
    FAILED = "failed"            # Something went wrong

@dataclass
class DocumentInfo:
    """
    Information about each document we're processing
    Like a filing card that tracks details about each document
    """
    filename: str                # Name of the file
    filepath: str               # Where the file is located
    content: str                # The actual text inside
    size_bytes: int             # How big the file is
    chunks: List[str] = None    # The pieces we break it into
    embeddings: List[List[float]] = None  # The numerical representations

@dataclass
class AgentState:
    """
    The 'memory' of our agent - tracks everything it's doing
    Like a detailed logbook of all activities
    """
    task_id: str                                    # Unique identifier for this task
    status: TaskStatus                             # Current state of processing
    documents: List[DocumentInfo]                  # All documents being processed
    total_chunks: int = 0                          # Total number of chunks created
    total_embeddings: int = 0                      # Total number of embeddings generated
    vector_collection_name: str = None             # Where vectors are stored
    start_time: float = None                       # When processing started
    end_time: float = None                         # When processing finished
    reasoning_log: List[Dict] = None               # All decisions made by the agent
    errors: List[str] = None                       # Any problems encountered
    
    def __post_init__(self):
        """Initialize empty lists if not provided"""
        if self.reasoning_log is None:
            self.reasoning_log = []
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = time.time()

# ==============================================================================
# STEP 4: SETUP LOGGING SYSTEM
# ==============================================================================

def setup_logging():
    """
    Setup detailed logging so we can track everything the agent does
    Like having a security camera that records all activities
    """
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/agent_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger("AutonomousAgent")
    logger.info("🔍 Logging system initialized")
    return logger

# Initialize logger
logger = setup_logging()

# ==============================================================================
# STEP 5: DOCUMENT LOADER TOOL
# ==============================================================================

class DocumentLoader:
    """
    TOOL 1: Document Loader
    This tool finds and reads documents from a specified folder
    Like a librarian that finds and opens books for you
    """
    
    def __init__(self, input_folder: str = Config.INPUT_FOLDER):
        self.input_folder = Path(input_folder)
        logger.info(f"📁 Document Loader initialized for folder: {input_folder}")
    
    def load_documents(self) -> List[DocumentInfo]:
        """
        Find and load all supported documents from the input folder
        
        Returns:
            List of DocumentInfo objects containing document details
        """
        
        print(f"\n🔍 STARTING DOCUMENT LOADING")
        print(f"Looking for documents in: {self.input_folder}")
        
        # Create input folder if it doesn't exist
        self.input_folder.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        # Look for each supported file type
        for file_extension in Config.SUPPORTED_FORMATS:
            pattern = f"*{file_extension}"
            file_paths = list(self.input_folder.glob(pattern))
            
            print(f"   Searching for {pattern} files: found {len(file_paths)} files")
            
            for file_path in file_paths:
                try:
                    # Read the file content
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    # Get file size
                    size_bytes = file_path.stat().st_size
                    
                    # Create document info
                    doc_info = DocumentInfo(
                        filename=file_path.name,
                        filepath=str(file_path),
                        content=content,
                        size_bytes=size_bytes
                    )
                    
                    documents.append(doc_info)
                    
                    print(f"   ✅ Loaded: {file_path.name} ({size_bytes} bytes, {len(content)} characters)")
                    logger.info(f"Loaded document: {file_path.name}")
                    
                except Exception as e:
                    error_msg = f"Failed to load {file_path.name}: {str(e)}"
                    print(f"   ❌ {error_msg}")
                    logger.error(error_msg)
        
        print(f"\n📊 DOCUMENT LOADING SUMMARY:")
        print(f"   Total documents loaded: {len(documents)}")
        print(f"   Total characters: {sum(len(doc.content) for doc in documents)}")
        
        if len(documents) == 0:
            print(f"\n⚠️  No documents found in {self.input_folder}")
            print(f"   Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}")
            print(f"   Please add some documents to process!")
        
        return documents

# ==============================================================================
# STEP 6: DOCUMENT CHUNKING TOOL
# ==============================================================================

class DocumentChunker:
    """
    TOOL 2: Document Chunker
    This tool breaks large documents into smaller, manageable pieces
    Like cutting a big pizza into slices so everyone can have a piece
    """
    
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"✂️ Document Chunker initialized (size: {chunk_size}, overlap: {overlap})")
    
    def chunk_documents(self, documents: List[DocumentInfo]) -> int:
        """
        Break all documents into smaller chunks
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            Total number of chunks created
        """
        
        print(f"\n✂️ STARTING DOCUMENT CHUNKING")
        print(f"Chunk size: {self.chunk_size} characters")
        print(f"Overlap: {self.overlap} characters")
        
        total_chunks = 0
        
        for doc in documents:
            print(f"\n   Processing: {doc.filename}")
            print(f"   Document length: {len(doc.content)} characters")
            
            # If document is smaller than chunk size, use as single chunk
            if len(doc.content) <= self.chunk_size:
                doc.chunks = [doc.content]
                print(f"   Document fits in single chunk")
            else:
                # Break document into overlapping chunks
                chunks = []
                start = 0
                
                while start < len(doc.content):
                    # Calculate end position
                    end = start + self.chunk_size
                    
                    # If we're not at the end, try to break at a sentence boundary
                    if end < len(doc.content):
                        # Look for sentence endings within the last 100 characters
                        sentence_boundaries = ['.', '!', '?', '\n']
                        best_break = end
                        
                        for boundary in sentence_boundaries:
                            boundary_pos = doc.content.rfind(boundary, start + self.chunk_size - 100, end)
                            if boundary_pos > start:
                                best_break = boundary_pos + 1
                                break
                        
                        end = best_break
                    
                    # Extract chunk and clean it up
                    chunk = doc.content[start:end].strip()
                    
                    if chunk:  # Only add non-empty chunks
                        chunks.append(chunk)
                    
                    # Move to next position with overlap
                    start = end - self.overlap
                    if start >= len(doc.content):
                        break
                
                doc.chunks = chunks
                print(f"   Created {len(chunks)} chunks")
            
            total_chunks += len(doc.chunks)
            
            # Log some example chunks for verification
            print(f"   Example chunk: '{doc.chunks[0][:100]}...'")
        
        print(f"\n📊 CHUNKING SUMMARY:")
        print(f"   Total documents processed: {len(documents)}")
        print(f"   Total chunks created: {total_chunks}")
        print(f"   Average chunks per document: {total_chunks/len(documents):.1f}")
        
        logger.info(f"Chunking completed: {total_chunks} chunks from {len(documents)} documents")
        
        return total_chunks

# ==============================================================================
# STEP 7: EMBEDDING GENERATION TOOL
# ==============================================================================

class EmbeddingGenerator:
    """
    TOOL 3: Embedding Generator
    This tool converts text chunks into numerical vectors (embeddings)
    Like translating words into a language that computers understand (numbers)
    """
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        print(f"\n🤖 Loading embedding model: {model_name}")
        print("   This converts text into numerical vectors...")
        
        try:
            self.model = SentenceTransformer(model_name)
            print("   ✅ Embedding model loaded successfully!")
            logger.info(f"Embedding model loaded: {model_name}")
        except Exception as e:
            error_msg = f"Failed to load embedding model: {str(e)}"
            print(f"   ❌ {error_msg}")
            logger.error(error_msg)
            raise
    
    def generate_embeddings(self, documents: List[DocumentInfo]) -> int:
        """
        Generate embeddings for all chunks in all documents
        
        Args:
            documents: List of documents with chunks to embed
            
        Returns:
            Total number of embeddings generated
        """
        
        print(f"\n🔢 STARTING EMBEDDING GENERATION")
        print("   Converting text chunks into numerical vectors...")
        
        total_embeddings = 0
        
        for doc in documents:
            if not doc.chunks:
                print(f"   ⚠️ Skipping {doc.filename}: No chunks available")
                continue
            
            print(f"\n   Processing: {doc.filename}")
            print(f"   Number of chunks: {len(doc.chunks)}")
            
            try:
                # Generate embeddings for all chunks at once (more efficient)
                print("   Generating embeddings... (this may take a moment)")
                embeddings = self.model.encode(doc.chunks, convert_to_numpy=True, show_progress_bar=True)
                
                # Convert to list format for JSON serialization
                doc.embeddings = [embedding.tolist() for embedding in embeddings]
                
                print(f"   ✅ Generated {len(doc.embeddings)} embeddings")
                print(f"   Embedding dimensions: {len(doc.embeddings[0])}")
                
                total_embeddings += len(doc.embeddings)
                
            except Exception as e:
                error_msg = f"Failed to generate embeddings for {doc.filename}: {str(e)}"
                print(f"   ❌ {error_msg}")
                logger.error(error_msg)
        
        print(f"\n📊 EMBEDDING GENERATION SUMMARY:")
        print(f"   Total embeddings generated: {total_embeddings}")
        print(f"   Vector dimensions: {len(documents[0].embeddings[0]) if documents and documents[0].embeddings else 'N/A'}")
        
        logger.info(f"Embedding generation completed: {total_embeddings} embeddings")
        
        return total_embeddings

# ==============================================================================
# STEP 8: VECTOR DATABASE STORAGE TOOL
# ==============================================================================

class VectorDatabase:
    """
    TOOL 4: Vector Database
    This tool stores and manages our embeddings in a searchable database
    Like a high-tech filing cabinet that can find similar documents instantly
    """
    
    def __init__(self, db_path: str = Config.VECTOR_DB_PATH):
        self.db_path = Path(db_path)
        print(f"\n🗄️ Initializing Vector Database")
        print(f"   Database path: {db_path}")
        
        try:
            # Create database directory if needed
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            print("   ✅ Vector database initialized successfully!")
            logger.info(f"Vector database initialized at {db_path}")
            
        except Exception as e:
            error_msg = f"Failed to initialize vector database: {str(e)}"
            print(f"   ❌ {error_msg}")
            logger.error(error_msg)
            raise
    
    def store_embeddings(self, documents: List[DocumentInfo], collection_name: str = None) -> str:
        """
        Store all embeddings in the vector database
        
        Args:
            documents: List of documents with embeddings to store
            collection_name: Name for the collection (auto-generated if None)
            
        Returns:
            Name of the collection where embeddings are stored
        """
        
        if collection_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            collection_name = f"doc_collection_{timestamp}"
        
        print(f"\n💾 STARTING VECTOR STORAGE")
        print(f"   Collection name: {collection_name}")
        
        try:
            # Delete collection if it already exists
            try:
                self.client.delete_collection(collection_name)
                print("   Removed existing collection")
            except:
                pass  # Collection didn't exist, which is fine
            
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Autonomous agent document collection"}
            )
            
            # Prepare data for batch insertion
            all_embeddings = []
            all_documents = []
            all_ids = []
            all_metadatas = []
            
            for doc in documents:
                if not doc.embeddings or not doc.chunks:
                    print(f"   ⚠️ Skipping {doc.filename}: No embeddings or chunks")
                    continue
                
                print(f"   Processing: {doc.filename}")
                print(f"   Storing {len(doc.embeddings)} vectors...")
                
                for i, (embedding, chunk) in enumerate(zip(doc.embeddings, doc.chunks)):
                    all_embeddings.append(embedding)
                    all_documents.append(chunk)
                    all_ids.append(f"{doc.filename}_chunk_{i}")
                    all_metadatas.append({
                        "source_file": doc.filename,
                        "chunk_index": i,
                        "file_size": doc.size_bytes,
                        "chunk_length": len(chunk)
                    })
            
            # Store everything in batch (more efficient)
            print(f"   Storing {len(all_embeddings)} vectors in database...")
            collection.add(
                embeddings=all_embeddings,
                documents=all_documents,
                ids=all_ids,
                metadatas=all_metadatas
            )
            
            # Verify storage
            collection_info = collection.count()
            print(f"   ✅ Successfully stored {collection_info} vectors")
            
            print(f"\n📊 VECTOR STORAGE SUMMARY:")
            print(f"   Collection: {collection_name}")
            print(f"   Total vectors: {collection_info}")
            print(f"   Documents processed: {len([d for d in documents if d.embeddings])}")
            
            logger.info(f"Vector storage completed: {collection_info} vectors in {collection_name}")
            
            return collection_name
            
        except Exception as e:
            error_msg = f"Failed to store embeddings: {str(e)}"
            print(f"   ❌ {error_msg}")
            logger.error(error_msg)
            raise

# ==============================================================================
# STEP 9: OLLAMA INTERFACE (AI BRAIN)
# ==============================================================================

class OllamaInterface:
    """
    OLLAMA Interface - The AI Brain of our agent
    This connects to OLLAMA to use LLaMA for reasoning and decision making
    Like having a smart advisor that helps make decisions
    """
    
    def __init__(self, base_url: str = Config.OLLAMA_BASE_URL, model: str = Config.OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        print(f"\n🧠 Initializing OLLAMA Interface")
        print(f"   URL: {base_url}")
        print(f"   Model: {model}")
        
        # Test connection to OLLAMA
        if self._test_connection():
            print("   ✅ OLLAMA connection successful!")
            logger.info(f"OLLAMA interface initialized: {base_url}/{model}")
        else:
            error_msg = "❌ Cannot connect to OLLAMA. Please ensure OLLAMA is running."
            print(f"   {error_msg}")
            print(f"   Try running: ollama serve")
            print(f"   Then: ollama pull {model}")
            logger.error(error_msg)
            # Don't raise error, let agent continue with simulated responses
    
    def _test_connection(self) -> bool:
        """Test if OLLAMA is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def reason_and_decide(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Use OLLAMA/LLaMA to analyze current state and decide next action
        This is where the AI brain makes decisions about what to do next
        
        Args:
            agent_state: Current state of the agent
            
        Returns:
            Dictionary with decision and reasoning
        """
        
        print(f"\n🧠 OLLAMA REASONING SESSION")
        print("   Analyzing current state and deciding next action...")
        
        # Create a detailed prompt for LLaMA
        prompt = self._create_reasoning_prompt(agent_state)
        
        # Try to get response from OLLAMA
        try:
            decision = self._query_ollama(prompt)
            print(f"   🎯 OLLAMA Decision: {decision['action']}")
            print(f"   💭 Reasoning: {decision['reasoning']}")
            
        except Exception as e:
            print(f"   ⚠️ OLLAMA unavailable, using fallback logic: {str(e)}")
            decision = self._fallback_decision_logic(agent_state)
            print(f"   🎯 Fallback Decision: {decision['action']}")
        
        # Log the decision
        reasoning_entry = {
            "timestamp": time.time(),
            "status": agent_state.status.value,
            "action": decision["action"],
            "reasoning": decision["reasoning"]
        }
        agent_state.reasoning_log.append(reasoning_entry)
        
        logger.info(f"Decision made: {decision['action']} - {decision['reasoning']}")
        
        return decision
    
    def _create_reasoning_prompt(self, agent_state: AgentState) -> str:
        """Create a detailed prompt for LLaMA reasoning"""
        
        # Count current progress
        docs_loaded = len(agent_state.documents)
        docs_with_chunks = len([d for d in agent_state.documents if d.chunks])
        docs_with_embeddings = len([d for d in agent_state.documents if d.embeddings])
        
        prompt = f"""
You are an autonomous document processing agent. Your task is to process documents through these stages:
1. Load documents from folder
2. Chunk documents into smaller pieces
3. Generate embeddings (convert text to vectors)
4. Store vectors in database

Current Status Analysis:
- Current Stage: {agent_state.status.value}
- Documents loaded: {docs_loaded}
- Documents chunked: {docs_with_chunks}/{docs_loaded}
- Documents with embeddings: {docs_with_embeddings}/{docs_loaded}
- Vector collection: {agent_state.vector_collection_name or 'Not created'}
- Total chunks: {agent_state.total_chunks}
- Total embeddings: {agent_state.total_embeddings}

Available Actions:
- load_documents: Find and load documents from input folder
- chunk_documents: Break documents into smaller pieces
- generate_embeddings: Convert chunks to numerical vectors
- store_vectors: Save vectors to database
- task_complete: Mark processing as finished

Based on the current status, what should be the next action? 
Respond in JSON format with 'action' and 'reasoning' fields.

Example: {{"action": "load_documents", "reasoning": "No documents loaded yet, need to start by loading documents"}}
"""
        
        return prompt
    
    def _query_ollama(self, prompt: str) -> Dict[str, Any]:
        """Send query to OLLAMA and parse response"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent responses
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Try to parse JSON response
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        decision_json = json.loads(json_match.group())
                        return {
                            "action": decision_json.get("action", "error"),
                            "reasoning": decision_json.get("reasoning", "No reasoning provided"),
                            "source": "ollama"
                        }
                except:
                    pass
                
                # If JSON parsing fails, extract action from text
                action = self._extract_action_from_text(response_text)
                return {
                    "action": action,
                    "reasoning": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "source": "ollama_text"
                }
            else:
                raise Exception(f"OLLAMA returned status code: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"OLLAMA query failed: {str(e)}")
    
    def _extract_action_from_text(self, text: str) -> str:
        """Extract action from text response if JSON parsing fails"""
        text_lower = text.lower()
        
        if "load" in text_lower and "document" in text_lower:
            return "load_documents"
        elif "chunk" in text_lower:
            return "chunk_documents"
        elif "embed" in text_lower or "vector" in text_lower:
            return "generate_embeddings"
        elif "store" in text_lower or "save" in text_lower:
            return "store_vectors"
        elif "complete" in text_lower or "finish" in text_lower:
            return "task_complete"
        else:
            return "error"
    
    def _fallback_decision_logic(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Fallback decision logic when OLLAMA is not available
        Simple rule-based decision making
        """
        
        # Simple state machine logic
        if agent_state.status == TaskStatus.PENDING:
            return {
                "action": "load_documents",
                "reasoning": "Starting fresh - need to load documents first",
                "source": "fallback"
            }
        
        elif agent_state.status == TaskStatus.LOADING_DOCUMENTS:
            if len(agent_state.documents) == 0:
                return {
                    "action": "error",
                    "reasoning": "No documents found to process",
                    "source": "fallback"
                }
            return {
                "action": "chunk_documents", 
                "reasoning": "Documents loaded, now need to chunk them",
                "source": "fallback"
            }
        
        elif agent_state.status == TaskStatus.CHUNKING:
            return {
                "action": "generate_embeddings",
                "reasoning": "Chunking done, now generate embeddings",
                "source": "fallback"
            }
        
        elif agent_state.status == TaskStatus.GENERATING_EMBEDDINGS:
            return {
                "action": "store_vectors",
                "reasoning": "Embeddings ready, now store in vector database",
                "source": "fallback"
            }
        
        elif agent_state.status == TaskStatus.STORING_VECTORS:
            return {
                "action": "task_complete",
                "reasoning": "All processing steps completed successfully",
                "source": "fallback"
            }
        
        else:
            return {
                "action": "error",
                "reasoning": f"Unknown status: {agent_state.status}",
                "source": "fallback"
            }

# ==============================================================================
# STEP 10: MAIN AUTONOMOUS AGENT CONTROLLER
# ==============================================================================

class AutonomousAgent:
    """
    MAIN AUTONOMOUS AGENT CONTROLLER
    
    This is the 'brain' of our system that coordinates all the tools
    It acts like a project manager that:
    1. Analyzes the current situation
    2. Decides what to do next
    3. Calls the right tool to do the work
    4. Checks the results
    5. Repeats until everything is done
    
    Think of it as a smart robot that can manage a complex workflow automatically
    """
    
    def __init__(self):
        print(f"\n🤖 INITIALIZING AUTONOMOUS AGENT")
        print("="*80)
        print("   Setting up all tools and components...")
        
        # Initialize all the tools our agent will use
        try:
            print("   📁 Setting up Document Loader...")
            self.document_loader = DocumentLoader()
            
            print("   ✂️ Setting up Document Chunker...")
            self.document_chunker = DocumentChunker()
            
            print("   🔢 Setting up Embedding Generator...")
            self.embedding_generator = EmbeddingGenerator()
            
            print("   🗄️ Setting up Vector Database...")
            self.vector_database = VectorDatabase()
            
            print("   🧠 Setting up OLLAMA Interface...")
            self.ollama_interface = OllamaInterface()
            
            print("   ✅ All tools initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to initialize agent tools: {str(e)}"
            print(f"   ❌ {error_msg}")
            logger.error(error_msg)
            raise
        
        # Map actions to their corresponding methods
        # This is like a phone book - when LLaMA says "do X", we know which tool to call
        self.action_map = {
            "load_documents": self._execute_document_loading,
            "chunk_documents": self._execute_document_chunking,
            "generate_embeddings": self._execute_embedding_generation,
            "store_vectors": self._execute_vector_storage,
            "task_complete": self._complete_task,
            "error": self._handle_error
        }
        
        logger.info("Autonomous Agent initialized successfully")
        print(f"\n🎯 AGENT READY TO PROCESS DOCUMENTS!")
        print("="*80)
    
    def process_documents_autonomously(self, task_id: str = None) -> AgentState:
        """
        MAIN ENTRY POINT - This is where the magic happens!
        
        This method starts the autonomous processing of documents
        The agent will:
        1. Look at the current situation
        2. Ask OLLAMA what to do next
        3. Execute that action
        4. Repeat until everything is done
        
        Args:
            task_id: Optional unique identifier for this task
            
        Returns:
            AgentState object with complete processing results
        """
        
        # Generate unique task ID if not provided
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        print(f"\n🚀 STARTING AUTONOMOUS DOCUMENT PROCESSING")
        print("="*80)
        print(f"   Task ID: {task_id}")
        print(f"   Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Input folder: {Config.INPUT_FOLDER}")
        print("="*80)
        
        # Initialize agent state (the agent's memory)
        agent_state = AgentState(
            task_id=task_id,
            status=TaskStatus.PENDING,
            documents=[]
        )
        
        logger.info(f"Starting autonomous processing - Task ID: {task_id}")
        
        # MAIN AGENT LOOP - This is the heart of our autonomous system
        iteration = 0
        max_iterations = Config.MAX_ITERATIONS
        
        print(f"\n🔄 ENTERING AGENT DECISION LOOP")
        print(f"   Maximum iterations allowed: {max_iterations}")
        print("-" * 80)
        
        while (agent_state.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
               and iteration < max_iterations):
            
            iteration += 1
            print(f"\n🔄 AGENT LOOP - ITERATION {iteration}")
            print(f"   Current Status: {agent_state.status.value}")
            print("-" * 40)
            
            try:
                # STEP 1: OLLAMA REASONING
                # Ask our AI brain what to do next based on current situation
                print("   🧠 Asking OLLAMA for next action...")
                decision = self.ollama_interface.reason_and_decide(agent_state)
                
                # STEP 2: ACTION EXECUTION
                # Execute the action that OLLAMA decided on
                action = decision["action"]
                print(f"   🎯 Executing action: {action}")
                
                if action in self.action_map:
                    # Call the appropriate method to handle this action
                    success = self.action_map[action](agent_state, decision)
                    
                    if not success and action != "error":
                        print(f"   ❌ Action failed: {action}")
                        agent_state.status = TaskStatus.FAILED
                        agent_state.errors.append(f"Action failed: {action}")
                        break
                        
                else:
                    error_msg = f"Unknown action: {action}"
                    print(f"   ❌ {error_msg}")
                    agent_state.status = TaskStatus.FAILED
                    agent_state.errors.append(error_msg)
                    break
                
                # STEP 3: PROGRESS CHECK
                # Small delay to make the process observable
                time.sleep(Config.DELAY_BETWEEN_STEPS)
                
                # Show progress
                self._show_progress_summary(agent_state, iteration)
                
            except Exception as e:
                error_msg = f"Error in agent loop iteration {iteration}: {str(e)}"
                print(f"   ❌ {error_msg}")
                logger.error(error_msg)
                agent_state.status = TaskStatus.FAILED
                agent_state.errors.append(error_msg)
                break
        
        # FINALIZATION
        agent_state.end_time = time.time()
        
        # Check if we hit max iterations
        if iteration >= max_iterations:
            print(f"\n⚠️  MAXIMUM ITERATIONS REACHED ({max_iterations})")
            agent_state.status = TaskStatus.FAILED
            agent_state.errors.append("Maximum iterations exceeded")
        
        # Show final results
        self._show_final_results(agent_state)
        
        return agent_state
    
    def _execute_document_loading(self, agent_state: AgentState, decision: Dict) -> bool:
        """
        Execute document loading action
        This tells our Document Loader tool to find and read all documents
        """
        
        print("   📁 EXECUTING: Document Loading")
        agent_state.status = TaskStatus.LOADING_DOCUMENTS
        
        try:
            documents = self.document_loader.load_documents()
            agent_state.documents = documents
            
            if len(documents) == 0:
                print("   ⚠️ No documents found to process")
                return False
            
            print(f"   ✅ Successfully loaded {len(documents)} documents")
            return True
            
        except Exception as e:
            error_msg = f"Document loading failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            agent_state.errors.append(error_msg)
            return False
    
    def _execute_document_chunking(self, agent_state: AgentState, decision: Dict) -> bool:
        """
        Execute document chunking action
        This tells our Chunker tool to break documents into smaller pieces
        """
        
        print("   ✂️ EXECUTING: Document Chunking")
        agent_state.status = TaskStatus.CHUNKING
        
        try:
            if not agent_state.documents:
                print("   ❌ No documents available for chunking")
                return False
            
            total_chunks = self.document_chunker.chunk_documents(agent_state.documents)
            agent_state.total_chunks = total_chunks
            
            if total_chunks == 0:
                print("   ❌ No chunks were created")
                return False
            
            print(f"   ✅ Successfully created {total_chunks} chunks")
            return True
            
        except Exception as e:
            error_msg = f"Document chunking failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            agent_state.errors.append(error_msg)
            return False
    
    def _execute_embedding_generation(self, agent_state: AgentState, decision: Dict) -> bool:
        """
        Execute embedding generation action
        This tells our Embedding Generator to convert chunks into numerical vectors
        """
        
        print("   🔢 EXECUTING: Embedding Generation")
        agent_state.status = TaskStatus.GENERATING_EMBEDDINGS
        
        try:
            # Check if we have chunks to process
            chunks_available = sum(1 for doc in agent_state.documents if doc.chunks)
            if chunks_available == 0:
                print("   ❌ No chunks available for embedding generation")
                return False
            
            total_embeddings = self.embedding_generator.generate_embeddings(agent_state.documents)
            agent_state.total_embeddings = total_embeddings
            
            if total_embeddings == 0:
                print("   ❌ No embeddings were generated")
                return False
            
            print(f"   ✅ Successfully generated {total_embeddings} embeddings")
            return True
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            agent_state.errors.append(error_msg)
            return False
    
    def _execute_vector_storage(self, agent_state: AgentState, decision: Dict) -> bool:
        """
        Execute vector storage action
        This tells our Vector Database to save all the embeddings
        """
        
        print("   💾 EXECUTING: Vector Storage")
        agent_state.status = TaskStatus.STORING_VECTORS
        
        try:
            # Check if we have embeddings to store
            embeddings_available = sum(1 for doc in agent_state.documents if doc.embeddings)
            if embeddings_available == 0:
                print("   ❌ No embeddings available for storage")
                return False
            
            collection_name = self.vector_database.store_embeddings(agent_state.documents)
            agent_state.vector_collection_name = collection_name
            
            print(f"   ✅ Successfully stored vectors in collection: {collection_name}")
            return True
            
        except Exception as e:
            error_msg = f"Vector storage failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            agent_state.errors.append(error_msg)
            return False
    
    def _complete_task(self, agent_state: AgentState, decision: Dict) -> bool:
        """
        Mark the task as completed
        This is called when all processing steps are done
        """
        
        print("   🎉 EXECUTING: Task Completion")
        agent_state.status = TaskStatus.COMPLETED
        
        print("   ✅ All processing steps completed successfully!")
        print("   📊 Final validation:")
        print(f"      - Documents processed: {len(agent_state.documents)}")
        print(f"      - Total chunks: {agent_state.total_chunks}")
        print(f"      - Total embeddings: {agent_state.total_embeddings}")
        print(f"      - Vector collection: {agent_state.vector_collection_name}")
        
        return True
    
    def _handle_error(self, agent_state: AgentState, decision: Dict) -> bool:
        """
        Handle error situations
        This is called when something goes wrong or OLLAMA can't decide what to do
        """
        
        print("   ❌ EXECUTING: Error Handling")
        agent_state.status = TaskStatus.FAILED
        
        error_msg = decision.get("reasoning", "Unknown error occurred")
        print(f"   Error details: {error_msg}")
        agent_state.errors.append(error_msg)
        
        return False
    
    def _show_progress_summary(self, agent_state: AgentState, iteration: int):
        """
        Show a quick progress summary after each iteration
        This helps us see what's happening during processing
        """
        
        print(f"\n   📊 PROGRESS SUMMARY (Iteration {iteration}):")
        print(f"      Status: {agent_state.status.value}")
        print(f"      Documents: {len(agent_state.documents)}")
        print(f"      Chunks: {agent_state.total_chunks}")
        print(f"      Embeddings: {agent_state.total_embeddings}")
        print(f"      Vector Collection: {agent_state.vector_collection_name or 'Not created'}")
        
        # Calculate progress percentage
        progress_steps = {
            TaskStatus.PENDING: 0,
            TaskStatus.LOADING_DOCUMENTS: 25,
            TaskStatus.CHUNKING: 50,
            TaskStatus.GENERATING_EMBEDDINGS: 75,
            TaskStatus.STORING_VECTORS: 90,
            TaskStatus.COMPLETED: 100
        }
        
        progress = progress_steps.get(agent_state.status, 0)
        print(f"      Progress: {progress}%")
        
        # Show progress bar
        bar_length = 20
        filled_length = int(bar_length * progress // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"      [{bar}] {progress}%")
    
    def _show_final_results(self, agent_state: AgentState):
        """
        Show comprehensive final results
        This is like a detailed report of everything that happened
        """
        
        print(f"\n{'='*80}")
        print(f"📋 FINAL PROCESSING RESULTS")
        print(f"{'='*80}")
        
        # Basic information
        print(f"Task ID: {agent_state.task_id}")
        print(f"Final Status: {agent_state.status.value.upper()}")
        
        # Timing information
        if agent_state.start_time and agent_state.end_time:
            duration = agent_state.end_time - agent_state.start_time
            print(f"Processing Time: {duration:.2f} seconds")
        
        # Processing statistics
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   Documents Found: {len(agent_state.documents)}")
        
        if agent_state.documents:
            total_chars = sum(len(doc.content) for doc in agent_state.documents)
            avg_doc_size = total_chars / len(agent_state.documents)
            print(f"   Total Characters: {total_chars:,}")
            print(f"   Average Document Size: {avg_doc_size:.0f} characters")
        
        print(f"   Total Chunks Created: {agent_state.total_chunks}")
        print(f"   Total Embeddings Generated: {agent_state.total_embeddings}")
        
        if agent_state.total_chunks > 0 and agent_state.documents:
            total_chars = sum(len(doc.content) for doc in agent_state.documents)
            print(f"   Average Chunk Size: {total_chars / agent_state.total_chunks:.0f} characters")
        
        # Vector database information
        if agent_state.vector_collection_name:
            print(f"\n🗄️ VECTOR DATABASE:")
            print(f"   Collection Name: {agent_state.vector_collection_name}")
            print(f"   Storage Location: {Config.VECTOR_DB_PATH}")
            print(f"   Vectors Stored: {agent_state.total_embeddings}")
        
        # Document details
        if agent_state.documents:
            print(f"\n📄 DOCUMENT DETAILS:")
            for i, doc in enumerate(agent_state.documents, 1):
                print(f"   {i}. {doc.filename}")
                print(f"      Size: {doc.size_bytes:,} bytes")
                print(f"      Chunks: {len(doc.chunks) if doc.chunks else 0}")
                print(f"      Embeddings: {len(doc.embeddings) if doc.embeddings else 0}")
        
        # Decision log
        if agent_state.reasoning_log:
            print(f"\n🧠 DECISION LOG:")
            for i, decision in enumerate(agent_state.reasoning_log, 1):
                timestamp = time.strftime('%H:%M:%S', time.localtime(decision['timestamp']))
                print(f"   {i}. [{timestamp}] {decision['action']}")
                print(f"      Status: {decision['status']}")
                print(f"      Reasoning: {decision['reasoning'][:100]}...")
        
        # Errors (if any)
        if agent_state.errors:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for i, error in enumerate(agent_state.errors, 1):
                print(f"   {i}. {error}")
        
        # Success message
        if agent_state.status == TaskStatus.COMPLETED:
            print(f"\n🎉 SUCCESS!")
            print(f"   Your documents have been successfully processed!")
            print(f"   Vector database is ready for similarity searches.")
            print(f"   Collection: {agent_state.vector_collection_name}")
        else:
            print(f"\n❌ PROCESSING INCOMPLETE")
            print(f"   Status: {agent_state.status.value}")
            print(f"   Check the error log above for details.")
        
        print(f"{'='*80}")
        
        # Log final results
        logger.info(f"Processing completed - Status: {agent_state.status.value}, "
                   f"Documents: {len(agent_state.documents)}, "
                   f"Chunks: {agent_state.total_chunks}, "
                   f"Embeddings: {agent_state.total_embeddings}")

# ==============================================================================
# STEP 11: DEMO AND USAGE FUNCTIONS
# ==============================================================================

def create_sample_documents():
    """
    Create sample documents for testing
    This function creates example documents so you can test the agent
    """
    
    print(f"\n📝 CREATING SAMPLE DOCUMENTS")
    print("   This will create example documents for testing...")
    
    # Create input directory
    input_dir = Path(Config.INPUT_FOLDER)
    input_dir.mkdir(exist_ok=True)
    
    # Sample documents with different content
    sample_docs = {
        "artificial_intelligence.txt": """
        Artificial Intelligence (AI) represents one of the most transformative technologies of our time. 
        It encompasses machine learning algorithms that can process vast datasets to identify complex patterns 
        and make intelligent predictions. Natural language processing enables computers to understand, 
        interpret, and generate human language with remarkable accuracy.
        
        Computer vision systems allow machines to perceive and analyze visual information from the world, 
        enabling applications like autonomous vehicles, medical image analysis, and facial recognition systems. 
        Deep learning neural networks have revolutionized how we approach problems in image recognition, 
        speech processing, and game playing.
        
        The applications of AI span across numerous industries. In healthcare, AI assists radiologists 
        in detecting diseases, helps pharmaceutical companies discover new drugs, and enables personalized 
        treatment plans. Financial institutions use AI for fraud detection, algorithmic trading, and 
        risk assessment. Manufacturing benefits from predictive maintenance and quality control systems.
        """,
        
        "machine_learning_basics.txt": """
        Machine Learning is a subset of artificial intelligence that focuses on algorithms that can 
        learn and improve from experience without being explicitly programmed. There are three main 
        types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
        
        Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. 
        Common supervised learning tasks include classification (predicting categories) and regression 
        (predicting numerical values). Popular algorithms include linear regression, decision trees, 
        random forests, and support vector machines.
        
        Unsupervised learning finds hidden patterns in data without labeled examples. Clustering algorithms 
        group similar data points together, while dimensionality reduction techniques help visualize 
        high-dimensional data. Principal Component Analysis (PCA) and K-means clustering are widely used 
        unsupervised learning techniques.
        
        Reinforcement learning involves an agent learning to make decisions by interacting with an 
        environment and receiving rewards or penalties. This approach has been particularly successful 
        in game playing, robotics, and autonomous systems.
        """,
        
        "data_science_overview.txt": """
        Data Science combines statistics, mathematics, programming, and domain expertise to extract 
        insights from data. The data science process typically involves data collection, cleaning, 
        exploration, modeling, and interpretation of results.
        
        Data preprocessing is crucial for successful analysis. This includes handling missing values, 
        removing outliers, normalizing data, and feature engineering. Exploratory data analysis helps 
        understand data distributions, correlations, and potential issues before modeling.
        
        Statistical modeling and machine learning algorithms are then applied to discover patterns 
        and make predictions. Model validation techniques like cross-validation ensure that results 
        generalize to new data. Finally, data visualization and storytelling communicate findings 
        to stakeholders effectively.
        
        Popular tools in data science include Python libraries like pandas, numpy, scikit-learn, 
        and matplotlib. R is another powerful language for statistical analysis. SQL is essential 
        for database queries, while tools like Tableau help create interactive visualizations.
        """
    }
    
    # Write sample documents
    for filename, content in sample_docs.items():
        file_path = input_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        
        print(f"   ✅ Created: {filename} ({len(content)} characters)")
    
    print(f"\n✅ Sample documents created in: {Config.INPUT_FOLDER}")
    print(f"   Total documents: {len(sample_docs)}")
    print(f"   Ready for autonomous processing!")
    
    return list(sample_docs.keys())

def run_demo():
    """
    Complete demonstration of the autonomous agent
    This function shows how to use the entire system
    """
    
    print(f"\n🎬 AUTONOMOUS DOCUMENT PROCESSING AGENT DEMO")
    print(f"{'='*80}")
    print(f"   This demo will show the complete autonomous document processing workflow")
    print(f"   The agent will make all decisions automatically using OLLAMA/LLaMA")
    print(f"{'='*80}")
    
    try:
        # Step 1: Create sample documents (if needed)
        if not Path(Config.INPUT_FOLDER).exists() or not list(Path(Config.INPUT_FOLDER).glob("*.txt")):
            print(f"\n📝 No documents found. Creating sample documents...")
            create_sample_documents()
        
        # Step 2: Initialize and run the autonomous agent
        print(f"\n🤖 Initializing Autonomous Agent...")
        agent = AutonomousAgent()
        
        # Step 3: Start autonomous processing
        print(f"\n🚀 Starting autonomous processing...")
        result = agent.process_documents_autonomously()
        
        # Step 4: Show results summary
        if result.status == TaskStatus.COMPLETED:
            print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print(f"   The agent has successfully processed all documents")
            print(f"   Vector database is ready for similarity searches")
            
            # Show how to access the results
            print(f"\n📋 HOW TO ACCESS RESULTS:")
            print(f"   Vector Database Path: {Config.VECTOR_DB_PATH}")
            print(f"   Collection Name: {result.vector_collection_name}")
            print(f"   Total Vectors: {result.total_embeddings}")
            
        else:
            print(f"\n❌ DEMO ENCOUNTERED ISSUES")
            print(f"   Final Status: {result.status.value}")
            if result.errors:
                print(f"   Errors: {', '.join(result.errors)}")
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {str(e)}")
        logger.error(f"Demo failed: {str(e)}")

def check_requirements():
    """
    Check if all required packages and services are available
    This function verifies that everything is properly installed
    """
    
    print(f"\n🔍 CHECKING SYSTEM REQUIREMENTS")
    print(f"{'='*50}")
    
    requirements_met = True
    
    # Check Python packages
    required_packages = [
        ("sentence-transformers", "SentenceTransformer embeddings"),
        ("chromadb", "Vector database"),
        ("numpy", "Numerical operations"),
        ("requests", "HTTP requests to OLLAMA")
    ]
    
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}: {description}")
        except ImportError:
            print(f"   ❌ {package}: {description} - MISSING")
            print(f"      Install with: pip install {package}")
            requirements_met = False
    
    # Check OLLAMA connection
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ OLLAMA: Connection successful")
        else:
            print(f"   ❌ OLLAMA: Server returned status {response.status_code}")
            requirements_met = False
    except Exception as e:
        print(f"   ⚠️  OLLAMA: Not accessible ({str(e)})")
        print(f"      Start with: ollama serve")
        print(f"      Install model: ollama pull {Config.OLLAMA_MODEL}")
        print(f"      Agent will use fallback logic if OLLAMA unavailable")
    
    # Check directories
    print(f"   📁 Input folder: {Config.INPUT_FOLDER}")
    print(f"   🗄️ Vector DB path: {Config.VECTOR_DB_PATH}")
    
    if requirements_met:
        print(f"\n✅ ALL REQUIREMENTS MET - Ready to run!")
    else:
        print(f"\n⚠️  Some requirements missing - Check installation")
    
    return requirements_met

# ==============================================================================
# STEP 12: MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    """
    Main execution point
    This is where the program starts when you run the script
    """
    
    print(f"""
{'='*80}
🤖 AUTONOMOUS DOCUMENT PROCESSING AGENT
{'='*80}
    
This agent will:
1. 📁 Load documents from input folder
2. ✂️ Chunk documents into smaller pieces  
3. 🔢 Generate embeddings (convert text to vectors)
4. 💾 Store vectors in ChromaDB database
5. 🧠 Make all decisions autonomously using OLLAMA/LLaMA

The agent acts as an intelligent coordinator that uses OLLAMA for reasoning
and specialized tools for each processing step.

{'='*80}
""")
    
    # Check system requirements first
    print("Checking system requirements...")
    check_requirements()
    
    # Ask user what they want to do
    print(f"\nWhat would you like to do?")
    print(f"1. Run complete demo with sample documents")
    print(f"2. Process documents from {Config.INPUT_FOLDER}")
    print(f"3. Create sample documents only")
    print(f"4. Check system requirements")
    
    try:
        choice = input(f"\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_demo()
        
        elif choice == "2":
            # Process existing documents
            agent = AutonomousAgent()
            result = agent.process_documents_autonomously()
            
        elif choice == "3":
            create_sample_documents()
        
        elif choice == "4":
            check_requirements()
        
        else:
            print(f"Invalid choice. Running demo...")
            run_demo()
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Processing interrupted by user")
        print(f"Thank you for using the Autonomous Document Processing Agent!")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.error(f"Main execution error: {str(e)}")
    
    finally:
        print(f"\n{'='*80}")
        print(f"🎯 Autonomous Document Processing Agent - Session Complete")
        print(f"{'='*80}")

# ==============================================================================
# END OF AUTONOMOUS DOCUMENT PROCESSING AGENT
# ==============================================================================

"""
USAGE INSTRUCTIONS FOR NON-TECHNICAL STAKEHOLDERS:
================================================================

WHAT THIS AGENT DOES:
- Takes documents from a folder and processes them automatically
- Breaks documents into smaller pieces for better handling
- Converts text into numbers that computers can understand (embeddings)
- Stores everything in a special database for fast searching
- Makes all decisions by itself using AI (OLLAMA/LLaMA)

HOW TO USE:
1. Install required packages: pip install sentence-transformers chromadb numpy requests
2. Start OLLAMA: ollama serve
3. Install LLaMA model: ollama pull llama3.2
4. Put your documents in the './documents' folder
5. Run this script: python autonomous_agent.py
6. Watch the agent work automatically!

WHAT YOU'LL SEE:
- The agent will show you each step it's taking
- It will explain why it made each decision
- You'll see progress updates as it works
- Final summary shows what was accomplished

THE RESULT:
- All your documents will be processed and stored in a vector database
- You can later search for similar content very quickly
- The system is ready for building AI applications that understand your documents

TROUBLESHOOTING:
- If OLLAMA isn't available, the agent uses simple rule-based logic
- Check system requirements if something doesn't work
- All activities are logged for debugging

This agent is designed to work completely autonomously - just start it and watch it work!
"""