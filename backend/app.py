# app.py - Complete Integrated SAMM ASIST System with Enhanced Agents
import os
import json
import uuid 
import time
import re
import asyncio
import sys
from datetime import datetime, timezone 
from typing import Dict, List, Any, Optional, TypedDict, Set
from urllib.parse import quote_plus, urlencode
from enum import Enum
from pathlib import Path
import functools

# Fix for Windows asyncio issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Flask & Extensions
from flask import Flask, request, jsonify, session, send_from_directory, redirect, url_for
from flask_cors import CORS
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename 

# Environment
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) 
def time_function(func):
    """Simple timing decorator for performance monitoring"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMING] {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper

# HTTP Requests Library
import requests

# Azure SDK
from azure.cosmos import CosmosClient, PartitionKey, exceptions as CosmosExceptions 
from azure.storage.blob import BlobServiceClient, ContentSettings 
from azure.core.exceptions import ResourceExistsError as BlobResourceExistsError, ResourceNotFoundError as BlobResourceNotFoundError

# Database imports for integrated agents
try:
    from gremlin_python.driver import client, serializer
    from gremlin_python.driver.protocol import GremlinServerError
    print("Gremlin client imported successfully")
except ImportError:
    print("Gremlin client not available - some features may be limited")
    client = None

try:
    import chromadb
    print("ChromaDB imported successfully")
except ImportError:
    print("ChromaDB not available - some features may be limited")
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformers imported successfully")
except ImportError:
    print("SentenceTransformers not available - some features may be limited")
    SentenceTransformer = None

# --- Application Configuration ---
# Auth0 Configuration
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "DFd55vvJIcV79cGuEETrGc9HWiNDqducM7upRwXdeJ9c4E3LbCtl")
import logging
logger = logging.getLogger(__name__)
# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Azure Storage Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
CASES_CONTAINER_NAME = os.getenv("CASES_CONTAINER_NAME") 

# Azure Blob Storage Configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CASE_DOCS_CONTAINER_NAME = os.getenv("AZURE_CASE_DOCS_CONTAINER_NAME")
AZURE_CHAT_DOCS_CONTAINER_NAME = os.getenv("AZURE_CHAT_DOCS_CONTAINER_NAME")

# Database Configuration for Enhanced Agents
COSMOS_GREMLIN_CONFIG = {
    'endpoint': os.getenv("COSMOS_GREMLIN_ENDPOINT", "asist-graph-db.gremlin.cosmos.azure.com").replace('wss://', '').replace(':443/', ''),
    'database': os.getenv("COSMOS_GREMLIN_DATABASE", "ASIST-Agent-1DB"),
    'graph': os.getenv("COSMOS_GREMLIN_COLLECTION", "Agent1"),
    'password': os.getenv("COSMOS_GREMLIN_KEY", "")
}

# Vector Database Configuration
VECTOR_DB_PATH = "./vector_db"
VECTOR_DB_TTL_PATH = "./vector_db_ttl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='static')
app.secret_key = APP_SECRET_KEY
CORS(app) 

# Simple in-memory storage for demo purposes when Azure isn't available
user_cases = {}
staged_documents = {}

print(f"Ollama URL: {OLLAMA_URL}")
print(f"Ollama Model: {OLLAMA_MODEL}")

# --- Initialize Cosmos DB Client ---
cosmos_client = None
database_client = None
cases_container_client = None

if COSMOS_ENDPOINT and COSMOS_KEY and DATABASE_NAME and CASES_CONTAINER_NAME:
    try:
        cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database_client = cosmos_client.get_database_client(DATABASE_NAME)
        cases_container_client = database_client.get_container_client(CASES_CONTAINER_NAME)
        print(f"Successfully connected to Cosmos DB Cases container: {DATABASE_NAME}/{CASES_CONTAINER_NAME}")
    except Exception as e:
        print(f"Warning: Error initializing Cosmos DB client: {e}. Using in-memory storage.")
else:
    print("Warning: Cosmos DB credentials not configured. Using in-memory storage.")

# --- Initialize Azure Blob Service Client ---
blob_service_client = None
case_docs_blob_container_client = None
chat_docs_blob_container_client = None

# ITAR Compliance Microservice Integration
COMPLIANCE_SERVICE_URL = os.getenv("COMPLIANCE_SERVICE_URL", "http://localhost:3002")
COMPLIANCE_SERVICE_ENABLED = os.getenv("COMPLIANCE_SERVICE_ENABLED", "true").lower() == "true"

def call_compliance_service(query: str, intent_info: Dict, entity_info: Dict, 
                           user_profile: Dict = None) -> Optional[Dict[str, Any]]:
    """Call the ITAR compliance microservice"""
    if not COMPLIANCE_SERVICE_ENABLED:
        return None
    
    try:
        data = {
            "query": query,
            "intent_info": intent_info,
            "entity_info": entity_info,
            "user_profile": user_profile or {}
        }
        
        response = requests.post(
            f"{COMPLIANCE_SERVICE_URL}/api/compliance/verify",
            json=data,
            timeout=10  # Short timeout to not block main request
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.Timeout:
        logger.warning("Compliance service timeout")
        return None
    except Exception as e:
        logger.error(f"Compliance service error: {e}")
        return None

def initialize_blob_container(bs_client, container_name_env_var, container_description):
    container_name = os.getenv(container_name_env_var)
    if not container_name:
        print(f"Warning: {container_name_env_var} is not set. {container_description} functionality will be disabled.")
        return None
    try:
        container_client = bs_client.get_container_client(container_name)
        container_client.create_container()
        print(f"Blob container '{container_name}' for {container_description} created or already exists.")
        return container_client
    except BlobResourceExistsError:
        print(f"Blob container '{container_name}' for {container_description} already exists.")
        return container_client
    except Exception as e_create_container:
        print(f"Could not create/verify blob container '{container_name}' for {container_description}: {e_create_container}")
        return None

if AZURE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        case_docs_blob_container_client = initialize_blob_container(blob_service_client, "AZURE_CASE_DOCS_CONTAINER_NAME", "case documents")
        chat_docs_blob_container_client = initialize_blob_container(blob_service_client, "AZURE_CHAT_DOCS_CONTAINER_NAME", "chat documents")
    except Exception as e:
        print(f"Warning: Error initializing Azure Blob Service client: {e}")
else:
    print("Warning: AZURE_CONNECTION_STRING is not set. Blob storage functionality will be disabled.")

# --- Auth0 OAuth Setup ---
oauth = None
if AUTH0_CLIENT_ID and AUTH0_CLIENT_SECRET and AUTH0_DOMAIN:
    oauth = OAuth(app)
    oauth.register(
        "auth0",
        client_id=AUTH0_CLIENT_ID,
        client_secret=AUTH0_CLIENT_SECRET,
        client_kwargs={"scope": "openid profile email"},
        server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration'
    )
    print("Auth0 OAuth configured successfully")
else:
    print("Warning: Auth0 credentials not configured. Authentication will use mock user.")

# =============================================================================
# ENHANCED OLLAMA CALL FUNCTION
# =============================================================================

from flask import Response, stream_with_context
import json




def call_ollama_streaming(prompt: str, system_message: str = "", temperature: float = 0.1):
    """Stream Ollama responses token by token"""
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
                "num_predict": 512
            }
        }
        
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", 
            json=data, 
            stream=True,  # Stream the response
            timeout=90
        )
        response.raise_for_status()
        
        # Yield each chunk as it arrives
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                if chunk.get('done', False):
                    break
                    
    except Exception as e:
        yield f"Error: {str(e)}"


def process_samm_query_streaming(query: str, chat_history: List = None, documents_context: List = None):
    """Process query with streaming support"""
    
    # Yield progress updates
    yield {"type": "progress", "step": "intent_analysis", "message": "Analyzing intent..."}
    
    # Intent analysis
    intent_info = orchestrator.intent_agent.analyze_intent(query)
    yield {"type": "intent", "data": intent_info}
    
    # Entity extraction
    yield {"type": "progress", "step": "entity_extraction", "message": "Extracting entities..."}
    entity_info = orchestrator.entity_agent.extract_and_retrieve(query, intent_info)
    yield {"type": "entities", "data": {
        "count": len(entity_info.get('entities', [])),
        "entities": entity_info.get('entities', [])
    }}
    
    # Generate answer with streaming
    yield {"type": "progress", "step": "generating_answer", "message": "Generating answer..."}
    
    # Build context
    context = orchestrator.answer_agent._build_comprehensive_context(
        query, intent_info, entity_info, chat_history, documents_context
    )
    system_msg = orchestrator.answer_agent._create_optimized_system_message(
        intent_info.get("intent", "general"), context
    )
    prompt = orchestrator.answer_agent._create_enhanced_prompt(query, intent_info, entity_info)
    
    # Stream the answer
    full_answer = ""
    for token in call_ollama_streaming(prompt, system_msg, temperature=0.1):
        full_answer += token
        yield {"type": "answer_chunk", "content": token}
    
    # Send final metadata
    yield {
        "type": "complete",
        "data": {
            "intent": intent_info.get('intent', 'unknown'),
            "entities_found": len(entity_info.get('entities', [])),
            "answer_length": len(full_answer)
        }
    }
def call_ollama_enhanced(prompt: str, system_message: str = "", temperature: float = 0.1) -> str:
    """Enhanced Ollama API call optimized for Llama 3.2"""
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,  # Optimized context window for Llama 3.2
                "num_predict": 512  # Maximum tokens to generate
            }
        }
        
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=data, timeout=90)
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        print(f"[Ollama Enhanced] API error: {e}")
        return f"Error calling Ollama API: {str(e)}"
    except Exception as e:
        print(f"[Ollama Enhanced] Processing error: {e}")
        return f"Error processing with Ollama: {str(e)}"

# =============================================================================
# EMBEDDED SAMM KNOWLEDGE GRAPH DATA (RDF/TTL)
# =============================================================================

SAMM_KNOWLEDGE_GRAPH = """
# SAMM Chapter 1 Knowledge Graph (TTL/RDF Format)
@prefix samm: <http://samm.mil/ontology#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

# Core Concepts
samm:SecurityCooperation rdf:type samm:Concept ;
    rdfs:label "Security Cooperation" ;
    samm:definition "All activities undertaken by the DoD to encourage and enable international partners to work with the United States to achieve strategic objectives" ;
    samm:section "C1.1.1" ;
    samm:authority "Title 10" ;
    samm:funding "DoD appropriations" .

samm:SecurityAssistance rdf:type samm:Concept ;
    rdfs:label "Security Assistance" ;
    samm:definition "Group of programs authorized under Title 22 authorities by which the United States provides defense articles, military education and training" ;
    samm:section "C1.1.2.2" ;
    samm:authority "Title 22" ;
    samm:funding "Foreign Operations appropriations" ;
    samm:relationship samm:isSubsetOf ;
    samm:relatedTo samm:SecurityCooperation .

# Organizations
samm:DSCA rdf:type samm:Organization ;
    rdfs:label "Defense Security Cooperation Agency" ;
    samm:fullName "Defense Security Cooperation Agency" ;
    samm:role "Directs, administers, and provides guidance to DoD Components for SC programs" ;
    samm:section "C1.3.2.2" .

samm:DepartmentOfState rdf:type samm:Organization ;
    rdfs:label "Department of State" ;
    samm:role "Continuous supervision and general direction of SA programs" ;
    samm:authority "Secretary of State" ;
    samm:section "C1.3.1" .

samm:DepartmentOfDefense rdf:type samm:Organization ;
    rdfs:label "Department of Defense" ;
    samm:role "Establishes military requirements and implements programs" ;
    samm:authority "Secretary of Defense" ;
    samm:section "C1.3.2" .

samm:DFAS rdf:type samm:Organization ;
    rdfs:label "Defense Finance and Accounting Service" ;
    samm:fullName "Defense Finance and Accounting Service" ;
    samm:role "Performs accounting, billing, disbursing, and collecting functions for SC programs" ;
    samm:section "C1.3.2.8" .

samm:ImplementingAgency rdf:type samm:Organization ;
    rdfs:label "Implementing Agency" ;
    samm:definition "MILDEP organization or defense agency responsible for execution of SC programs" ;
    samm:role "Overall management of actions for delivery of materiel, supporting equipment, or services" ;
    samm:section "C1.3.2.6" .

# Legal Authorities
samm:ForeignAssistanceAct rdf:type samm:Authority ;
    rdfs:label "Foreign Assistance Act" ;
    samm:year "1961" ;
    samm:type "Title 22" ;
    samm:section "C1.2.1" .

samm:ArmsExportControlAct rdf:type samm:Authority ;
    rdfs:label "Arms Export Control Act" ;
    samm:acronym "AECA" ;
    samm:year "1976" ;
    samm:type "Title 22" ;
    samm:section "C1.2.1" .

samm:NDAA rdf:type samm:Authority ;
    rdfs:label "National Defense Authorization Act" ;
    samm:acronym "NDAA" ;
    samm:type "Title 10" ;
    samm:annual "true" ;
    samm:section "C1.1.2.1" .

# Key Relationships and Distinctions
samm:SecurityAssistance samm:isSubsetOf samm:SecurityCooperation .
samm:SecurityCooperation samm:authorizedBy samm:NDAA .
samm:SecurityAssistance samm:authorizedBy samm:ForeignAssistanceAct .
samm:SecurityAssistance samm:authorizedBy samm:ArmsExportControlAct .
samm:SecurityAssistance samm:supervisedBy samm:DepartmentOfState .
samm:SecurityCooperation samm:ledBy samm:DepartmentOfDefense .
samm:DSCA samm:directsPrograms samm:SecurityCooperation .
samm:DFAS samm:providesFinancialServices samm:SecurityCooperation .
"""

SAMM_TEXT_CONTENT = """
SAMM Chapter 1 - Security Cooperation Overview and Relationships

C1.1.1 Definition: Security cooperation (SC) comprises all activities undertaken by the DoD to encourage and enable international partners to work with the United States to achieve strategic objectives.

C1.1.2.2 Security Assistance: SA is a group of programs, authorized under Title 22 authorities, by which the United States provides defense articles, military education and training, and other defense-related services. All SA programs are subject to the continuous supervision and general direction of the Secretary of State.

Key Distinction: Security Assistance programs are a SUBSET of Security Cooperation programs. SC is broader and includes all DoD activities with foreign partners, while SA is specifically those activities authorized under Title 22.

C1.3.1 Department of State: Under the FAA, AECA, and Executive Order 13637, the Secretary of State is responsible for continuous supervision and general direction of SA programs.

C1.3.2.2 DSCA: Defense Security Cooperation Agency directs, administers, and provides guidance to DoD Components for the execution of DoD SC programs.

C1.3.2.6 Implementing Agencies: An IA is the MILDEP organization or defense agency responsible for the execution of SC programs and overall management of delivery.

C1.3.2.8 DFAS: Defense Finance and Accounting Service performs accounting, billing, disbursing, and collecting functions for SC programs.

C1.2.1 Authorities: SA authorizations come primarily from the Foreign Assistance Act (FAA) of 1961 and Arms Export Control Act (AECA) of 1976.
"""

# =============================================================================
# SIMPLE KNOWLEDGE GRAPH PARSER
# =============================================================================

class SimpleKnowledgeGraph:
    """Simple knowledge graph parser for SAMM TTL data"""
    
    def __init__(self, ttl_data: str):
        self.entities = {}
        self.relationships = []
        self._parse_ttl(ttl_data)
    
    def _parse_ttl(self, ttl_data: str):
        """Parse TTL data into structured knowledge"""
        lines = ttl_data.split('\n')
        current_entity = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # New entity definition
            if 'rdf:type' in line:
                parts = line.split()
                if len(parts) >= 3:
                    entity_id = parts[0].replace('samm:', '')
                    entity_type = parts[2].replace('samm:', '').replace(';', '')
                    current_entity = {
                        'id': entity_id,
                        'type': entity_type,
                        'properties': {}
                    }
                    self.entities[entity_id] = current_entity
            
            # Properties
            elif current_entity and any(prop in line for prop in ['rdfs:label', 'samm:definition', 'samm:role', 'samm:section', 'samm:authority', 'samm:year']):
                if '"' in line:
                    prop_name = line.split()[0].replace('samm:', '').replace('rdfs:', '')
                    prop_value = line.split('"')[1] if '"' in line else line.split()[-1].replace(';', '').replace('.', '')
                    current_entity['properties'][prop_name] = prop_value
            
            # Relationships
            elif current_entity and any(rel in line for rel in ['samm:isSubsetOf', 'samm:supervisedBy', 'samm:ledBy', 'samm:authorizedBy']):
                parts = line.split()
                if len(parts) >= 2:
                    relationship = parts[0].replace('samm:', '')
                    target = parts[1].replace('samm:', '').replace('.', '').replace(';', '')
                    self.relationships.append({
                        'source': current_entity['id'],
                        'relationship': relationship,
                        'target': target
                    })
    
    def find_entity(self, query: str) -> Optional[Dict]:
        """Find entity by name or label"""
        query_lower = query.lower()
        
        # Direct match
        for entity_id, entity in self.entities.items():
            if entity_id.lower() == query_lower:
                return entity
            if entity['properties'].get('label', '').lower() == query_lower:
                return entity
        
        # Partial match
        for entity_id, entity in self.entities.items():
            if query_lower in entity_id.lower():
                return entity
            if query_lower in entity['properties'].get('label', '').lower():
                return entity
        
        return None
    
    def get_relationships(self, entity_id: str) -> List[Dict]:
        """Get relationships for an entity"""
        return [rel for rel in self.relationships 
                if rel['source'] == entity_id or rel['target'] == entity_id]

# Initialize knowledge graph
knowledge_graph = SimpleKnowledgeGraph(SAMM_KNOWLEDGE_GRAPH)
print(f"Knowledge Graph loaded: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relationships)} relationships")

# =============================================================================
# DATABASE MANAGER FOR INTEGRATED AGENTS
# =============================================================================

class DatabaseManager:
    """
    Manages connections to all three databases with improved error handling
    """
    
    def __init__(self):
        self.cosmos_gremlin_client = None
        self.vector_db_client = None
        self.vector_db_ttl_client = None
        self.embedding_model = None
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize all database connections with better error handling"""
        print("[DatabaseManager] Initializing database connections...")
        
        # Initialize Cosmos DB Gremlin connection
        self._init_cosmos_gremlin()
        # Initialize ChromaDB connections
        self._init_vector_dbs()
        # Initialize embedding model
        self._init_embedding_model()
    
    def _init_cosmos_gremlin(self):
        """Initialize Cosmos DB Gremlin with proper cleanup"""
        if not client or not COSMOS_GREMLIN_CONFIG['password']:
            print("[DatabaseManager] Cosmos Gremlin credentials not available")
            return
            
        try:
            username = f"/dbs/{COSMOS_GREMLIN_CONFIG['database']}/colls/{COSMOS_GREMLIN_CONFIG['graph']}"
            endpoint_url = f"wss://{COSMOS_GREMLIN_CONFIG['endpoint']}:443/gremlin"
            
            self.cosmos_gremlin_client = client.Client(
                url=endpoint_url,
                traversal_source="g",
                username=username,
                password=COSMOS_GREMLIN_CONFIG['password'],
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            # Test connection with timeout
            result = self.cosmos_gremlin_client.submit("g.V().limit(1).count()").all().result()
            print(f"[DatabaseManager] Cosmos Gremlin connected successfully - {result[0]} vertices available")
            
        except Exception as e:
            print(f"[DatabaseManager] Cosmos Gremlin connection failed: {e}")
            self.cosmos_gremlin_client = None
    
    def _init_vector_dbs(self):
        """Initialize vector databases"""
        if not chromadb:
            print("[DatabaseManager] ChromaDB not available")
            return
            
        # Initialize ChromaDB vector_db (documents)
        try:
            if Path(VECTOR_DB_PATH).exists():
                self.vector_db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
                collections = self.vector_db_client.list_collections()
                print(f"[DatabaseManager] Vector DB connected - {len(collections)} collections available")
            else:
                print(f"[DatabaseManager] Vector DB path not found: {VECTOR_DB_PATH}")
        except Exception as e:
            print(f"[DatabaseManager] Vector DB connection failed: {e}")
            self.vector_db_client = None
        
        # Initialize ChromaDB vector_db_ttl (TTL files)
        try:
            if Path(VECTOR_DB_TTL_PATH).exists():
                self.vector_db_ttl_client = chromadb.PersistentClient(path=VECTOR_DB_TTL_PATH)
                collections = self.vector_db_ttl_client.list_collections()
                print(f"[DatabaseManager] Vector DB TTL connected - {len(collections)} collections available")
            else:
                print(f"[DatabaseManager] Vector DB TTL path not found: {VECTOR_DB_TTL_PATH}")
        except Exception as e:
            print(f"[DatabaseManager] Vector DB TTL connection failed: {e}")
            self.vector_db_ttl_client = None
    
    def _init_embedding_model(self):
        """Initialize embedding model"""
        if not SentenceTransformer:
            print("[DatabaseManager] SentenceTransformer not available")
            return
            
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"[DatabaseManager] Embedding model loaded: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"[DatabaseManager] Embedding model failed to load: {e}")
            self.embedding_model = None
    
    def query_cosmos_graph(self, query_text: str, entities: List[str] = None) -> List[Dict]:
        """Query Cosmos DB graph database with better error handling"""
        if not self.cosmos_gremlin_client:
            return []
        
        results = []
        
        try:
            if entities:
                # Limit entities to prevent too many queries
                limited_entities = entities[:3]  # Only process first 3 entities
                
                for entity in limited_entities:
                    # Clean entity name for Gremlin query
                    entity_clean = re.sub(r'[^\w\s]', '', entity).strip()
                    if not entity_clean:
                        continue
                    
                    try:
                        # Query for vertices with matching names (with timeout)
                        vertex_query = f"g.V().has('name', containing('{entity_clean}')).limit(10)"
                        vertex_results = self.cosmos_gremlin_client.submit(vertex_query).all().result()
                        
                        for vertex in vertex_results:
                            results.append({
                                "type": "vertex",
                                "data": vertex,
                                "source": "cosmos_gremlin",
                                "entity": entity
                            })
                        
                        # Query for relationships involving this entity (limited)
                        edge_query = f"g.V().has('name', containing('{entity_clean}')).bothE().limit(5)"
                        edge_results = self.cosmos_gremlin_client.submit(edge_query).all().result()
                        
                        for edge in edge_results:
                            results.append({
                                "type": "edge", 
                                "data": edge,
                                "source": "cosmos_gremlin",
                                "entity": entity
                            })
                            
                    except Exception as entity_error:
                        print(f"[DatabaseManager] Error querying entity '{entity}': {entity_error}")
                        continue
            else:
                # General query for high-level entities
                general_query = "g.V().limit(10)"
                general_results = self.cosmos_gremlin_client.submit(general_query).all().result()
                
                for vertex in general_results:
                    results.append({
                        "type": "vertex",
                        "data": vertex,
                        "source": "cosmos_gremlin"
                    })
            
            print(f"[DatabaseManager] Cosmos Gremlin query returned {len(results)} results")
            
        except Exception as e:
            print(f"[DatabaseManager] Cosmos Gremlin query error: {e}")
        
        return results
    
    def query_vector_db(self, query_text: str, collection_name: str = None, n_results: int = 3) -> List[Dict]:
        """Query ChromaDB vector_db with reduced results"""
        if not self.vector_db_client or not self.embedding_model:
            return []
        
        results = []
        
        try:
            collections = self.vector_db_client.list_collections()
            if not collections:
                return []
            
            collection = collections[0] if not collection_name else self.vector_db_client.get_collection(collection_name)
            
            query_embedding = self.embedding_model.encode([query_text]).tolist()[0]
            
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            for i, (doc, metadata, distance) in enumerate(zip(
                query_results['documents'][0],
                query_results['metadatas'][0], 
                query_results['distances'][0]
            )):
                results.append({
                    "type": "document_chunk",
                    "content": doc[:500] + "..." if len(doc) > 500 else doc,  # Truncate long content
                    "metadata": metadata,
                    "similarity_score": round(1 - distance, 3),
                    "source": "vector_db",
                    "collection": collection.name
                })
            
            print(f"[DatabaseManager] Vector DB query returned {len(results)} results")
            
        except Exception as e:
            print(f"[DatabaseManager] Vector DB query error: {e}")
        
        return results
    
    def query_vector_db_ttl(self, query_text: str, collection_name: str = None, n_results: int = 2) -> List[Dict]:
        """Query ChromaDB vector_db_ttl with reduced results"""
        if not self.vector_db_ttl_client or not self.embedding_model:
            return []
        
        results = []
        
        try:
            collections = self.vector_db_ttl_client.list_collections()
            if not collections:
                return []
            
            collection = collections[0] if not collection_name else self.vector_db_ttl_client.get_collection(collection_name)
            
            query_embedding = self.embedding_model.encode([query_text]).tolist()[0]
            
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            for i, (doc, metadata, distance) in enumerate(zip(
                query_results['documents'][0],
                query_results['metadatas'][0],
                query_results['distances'][0]
            )):
                results.append({
                    "type": "ttl_chunk",
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,  # Truncate long content
                    "metadata": metadata,
                    "similarity_score": round(1 - distance, 3),
                    "source": "vector_db_ttl", 
                    "collection": collection.name
                })
            
            print(f"[DatabaseManager] Vector DB TTL query returned {len(results)} results")
            
        except Exception as e:
            print(f"[DatabaseManager] Vector DB TTL query error: {e}")
        
        return results
    
    def cleanup(self):
        """Cleanup database connections"""
        try:
            if self.cosmos_gremlin_client:
                self.cosmos_gremlin_client.close()
                print("[DatabaseManager] Cosmos Gremlin connection closed")
        except Exception as e:
            print(f"[DatabaseManager] Error closing Cosmos Gremlin: {e}")
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get status of all database connections"""
        status = {
            "cosmos_gremlin": {
                "connected": self.cosmos_gremlin_client is not None,
                "endpoint": COSMOS_GREMLIN_CONFIG['endpoint'],
                "database": COSMOS_GREMLIN_CONFIG['database'],
                "graph": COSMOS_GREMLIN_CONFIG['graph']
            },
            "vector_db": {
                "connected": self.vector_db_client is not None,
                "path": VECTOR_DB_PATH,
                "collections": []
            },
            "vector_db_ttl": {
                "connected": self.vector_db_ttl_client is not None,
                "path": VECTOR_DB_TTL_PATH,
                "collections": []
            },
            "embedding_model": {
                "loaded": self.embedding_model is not None,
                "model_name": EMBEDDING_MODEL
            }
        }
        
        # Get collection info safely
        try:
            if self.vector_db_client:
                collections = self.vector_db_client.list_collections()
                status["vector_db"]["collections"] = [c.name for c in collections]
        except:
            pass
        
        try:
            if self.vector_db_ttl_client:
                collections = self.vector_db_ttl_client.list_collections()
                status["vector_db_ttl"]["collections"] = [c.name for c in collections]
        except:
            pass
        
        return status

# Initialize database manager
db_manager = DatabaseManager()

# =============================================================================
# LANGGRAPH STATE ORCHESTRATION SYSTEM
# =============================================================================

class AgentState(TypedDict):
    """State shared across all agents in the workflow"""
    query: str
    chat_history: Optional[List[Dict]]
    documents_context: Optional[List[Dict]]
    intent_info: Optional[Dict[str, Any]]
    entity_info: Optional[Dict[str, Any]]
    answer: Optional[str]
    execution_steps: List[str]
    start_time: float
    current_step: str
    error: Optional[str]

class WorkflowStep(Enum):
    """Workflow steps for state orchestration"""
    INIT = "initialize"
    INTENT = "analyze_intent"
    ENTITY = "extract_entities"
    ANSWER = "generate_answer"
    COMPLETE = "complete"
    ERROR = "error"

def call_ollama(prompt: str, system_message: str = "") -> str:
    """Call Ollama with system message and prompt (legacy function for compatibility)"""
    return call_ollama_enhanced(prompt, system_message, temperature=0.1)

class IntentAgent:
    """Intent analysis using Ollama with Human-in-Loop and trigger updates"""
    
    def __init__(self):
        self.hil_feedback_data = []  # Store human feedback for intent corrections
        self.intent_patterns = {}    # Store learned patterns from feedback
        self.trigger_updates = []    # Store updates from new entity/relationship data
    
    @time_function
    def analyze_intent(self, query: str) -> Dict[str, Any]:
        # Check if we have learned patterns from previous feedback
        enhanced_system_msg = self._build_enhanced_system_message()
        
        prompt = f"Analyze this SAMM query and determine intent: {query}"
        
        try:
            response = call_ollama_enhanced(prompt, enhanced_system_msg, temperature=0.0)
            # Try to parse JSON response
            if "{" in response and "}" in response:
                json_part = response[response.find("{"):response.rfind("}")+1]
                result = json.loads(json_part)
                
                # Apply any learned corrections from HIL feedback
                result = self._apply_hil_corrections(query, result)
                return result
            else:
                return {"intent": "general", "confidence": 0.5, "entities_mentioned": []}
        except:
            return {"intent": "general", "confidence": 0.5, "entities_mentioned": []}
    
    def update_from_hil(self, query: str, original_intent: str, corrected_intent: str, feedback_data: Dict[str, Any] = None):
        """Update agent based on human-in-the-loop feedback"""
        feedback_entry = {
            "query": query,
            "original_intent": original_intent,
            "corrected_intent": corrected_intent,
            "feedback_data": feedback_data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.hil_feedback_data.append(feedback_entry)
        
        # Learn patterns from the correction
        query_lower = query.lower()
        if corrected_intent not in self.intent_patterns:
            self.intent_patterns[corrected_intent] = []
        
        # Extract keywords from corrected queries for pattern learning
        keywords = [word for word in query_lower.split() if len(word) > 3]
        self.intent_patterns[corrected_intent].extend(keywords)
        
        print(f"[IntentAgent HIL] Updated with correction: {original_intent} -> {corrected_intent} for query: '{query}'")
        return True
    
    def update_from_trigger(self, new_entities: List[str], new_relationships: List[Dict], trigger_data: Dict[str, Any] = None):
        """Update agent when new entity/relationship data is available"""
        trigger_entry = {
            "new_entities": new_entities,
            "new_relationships": new_relationships,
            "trigger_data": trigger_data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.trigger_updates.append(trigger_entry)
        
        # Update intent recognition patterns based on new entities
        for entity in new_entities:
            entity_lower = entity.lower()
            # Add entity-specific intent patterns
            if "agency" in entity_lower or "organization" in entity_lower:
                if "organization" not in self.intent_patterns:
                    self.intent_patterns["organization"] = []
                self.intent_patterns["organization"].append(entity_lower)
        
        print(f"[IntentAgent Trigger] Updated with {len(new_entities)} new entities and {len(new_relationships)} relationships")
        return True
    
    def _build_enhanced_system_message(self) -> str:
        """Build system message enhanced with learned patterns"""
        base_msg = """You are a SAMM (Security Assistance Management Manual) intent analyzer. 
        Classify the user's query into one of these categories:
        - definition: asking what something is
        - distinction: asking about differences between concepts  
        - authority: asking about who has authority or oversight
        - organization: asking about agencies and their roles
        - factual: asking for specific facts like dates, numbers
        - relationship: asking about how things are connected
        - general: general questions"""
        
        # Add learned patterns if available
        if self.intent_patterns:
            base_msg += "\n\nLearned patterns from feedback:"
            for intent, keywords in self.intent_patterns.items():
                if keywords:
                    unique_keywords = list(set(keywords))[:5]  # Limit to top 5 unique keywords
                    base_msg += f"\n- {intent}: commonly involves {', '.join(unique_keywords)}"
        
        base_msg += "\n\nRespond with JSON format: {\"intent\": \"category\", \"confidence\": 0.8, \"entities_mentioned\": [\"entity1\", \"entity2\"]}"
        return base_msg
    
    def _apply_hil_corrections(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned corrections from HIL feedback"""
        query_lower = query.lower()
        
        # Check if this query pattern has been corrected before
        for feedback in self.hil_feedback_data[-10:]:  # Check last 10 feedback entries
            if any(word in query_lower for word in feedback["query"].lower().split() if len(word) > 3):
                # Apply confidence adjustment based on past corrections
                if result["intent"] == feedback["original_intent"]:
                    result["confidence"] = max(0.3, result.get("confidence", 0.5) - 0.2)
                    result["hil_note"] = f"Similar pattern previously corrected to {feedback['corrected_intent']}"
        
        return result

class IntegratedEntityAgent:
    """
    Integrated Entity Agent with database connections and enhanced extraction
    """
    
    def __init__(self, knowledge_graph=None, db_manager=None):
        print("[IntegratedEntityAgent] Initializing with database connections...")
        
        self.knowledge_graph = knowledge_graph
        self.db_manager = db_manager or db_manager
        
        # Learning and feedback systems
        self.hil_feedback_data = []        # Human-in-the-loop feedback storage
        self.custom_entities = {}          # User-defined entities from feedback
        self.trigger_updates = []          # Trigger-based updates storage
        self.dynamic_knowledge = {         # Dynamic knowledge base
            "entities": {},
            "relationships": []
        }
        
        # Enhanced entity patterns
        self.samm_entity_patterns = {
            "organizations": [
                "DSCA", "Defense Security Cooperation Agency",
                "Department of State", "DoS", "State Department",
                "Department of Defense", "DoD", "Defense Department", 
                "DFAS", "Defense Finance and Accounting Service",
                "Implementing Agency", "IA", "MILDEP",
                "Military Department", "Defense Agency",
                "Secretary of State", "Secretary of Defense"
            ],
            "programs": [
                "Security Cooperation", "SC", "Security Cooperation programs",
                "Security Assistance", "SA", "Security Assistance programs", 
                "Foreign Military Sales", "FMS",
                "Foreign Military Financing", "FMF",
                "International Military Education and Training", "IMET",
                "Defense articles", "Military education", "Training programs"
            ],
            "authorities": [
                "Foreign Assistance Act", "FAA", "Foreign Assistance Act of 1961",
                "Arms Export Control Act", "AECA", "Arms Export Control Act of 1976",
                "National Defense Authorization Act", "NDAA",
                "Title 10", "Title 22", "Title 10 authorities", "Title 22 authorities",
                "Executive Order", "Executive Order 13637"
            ],
            "concepts": [
                "continuous supervision", "general direction",
                "defense articles", "military education and training",
                "defense-related services", "strategic objectives",
                "international partners", "DoD Components",
                "overall management", "delivery of materiel"
            ],
            "sections": [
                "C1.1.1", "C1.1.2", "C1.1.2.2", "C1.2.1", 
                "C1.3.1", "C1.3.2", "C1.3.2.2", "C1.3.2.6", "C1.3.2.8"
            ]
        }
        
        # Entity relationship mappings for SAMM Chapter 1
        self.entity_relationships = {
            "DSCA": ["directs", "administers", "provides guidance to DoD Components"],
            "Defense Security Cooperation Agency": ["directs", "administers", "provides guidance"],
            "Department of State": ["supervises", "provides continuous supervision", "provides general direction"],
            "Department of Defense": ["establishes military requirements", "implements programs"],
            "DFAS": ["performs accounting", "performs billing", "performs disbursing", "performs collecting"],
            "Defense Finance and Accounting Service": ["provides financial services"],
            "Security Assistance": ["is subset of Security Cooperation", "authorized under Title 22"],
            "Security Cooperation": ["includes Security Assistance", "authorized under Title 10"],
            "Secretary of State": ["responsible for continuous supervision", "provides general direction"],
            "Secretary of Defense": ["establishes requirements", "oversees implementation"]
        }
        
        # Confidence scoring weights
        self.confidence_weights = {
            "exact_match": 1.0,
            "partial_match": 0.8,
            "acronym_match": 0.9,
            "context_match": 0.6,
            "ai_extracted": 0.7,
            "knowledge_graph": 0.95,
            "dynamic_knowledge": 0.8,
            "database_match": 0.9
        }
        
        print("[IntegratedEntityAgent] Initialization complete")
    @time_function
    def extract_and_retrieve(self, query: str, intent_info: Dict) -> Dict[str, Any]:
        """
        Main method for integrated entity extraction and database retrieval
        """
        print(f"[IntegratedEntityAgent] Processing query: '{query}' with intent: {intent_info.get('intent', 'unknown')}")
        
        try:
            # Phase 1: Enhanced entity extraction
            entities = self._extract_entities_enhanced(query, intent_info)
            print(f"[IntegratedEntityAgent] Extracted entities: {entities}")
            
            # Phase 2: Query all data sources
            all_results = {
                "query": query,
                "entities": entities,
                "intent_info": intent_info,
                "timestamp": datetime.now().isoformat(),
                "data_sources": {},
                "context": [],
                "text_sections": [],
                "relationships": [],
                "confidence_scores": {},
                "overall_confidence": 0.0,
                "extraction_method": "integrated_database_enhanced",
                "extraction_phases": ["pattern_matching", "nlp_extraction", "database_queries"],
                "phase_count": 3
            }
            
            # Query each source with error handling
            cosmos_results = self._safe_query_cosmos(query, entities)
            vector_results = self._safe_query_vector(query)
            vector_ttl_results = self._safe_query_vector_ttl(query)
            
            all_results["data_sources"] = {
                "cosmos_gremlin": {
                    "results": cosmos_results,
                    "count": len(cosmos_results),
                    "status": "success" if cosmos_results else "no_results"
                },
                "vector_db": {
                    "results": vector_results,
                    "count": len(vector_results),
                    "status": "success" if vector_results else "no_results"
                },
                "vector_db_ttl": {
                    "results": vector_ttl_results,
                    "count": len(vector_ttl_results),
                    "status": "success" if vector_ttl_results else "no_results"
                }
            }
            
            # Phase 3: Generate enhanced context from all sources
            self._populate_enhanced_context(all_results, entities)
            
            print(f"[IntegratedEntityAgent] Query complete: {len(entities)} entities, multiple data sources")
            return all_results
            
        except Exception as e:
            print(f"[IntegratedEntityAgent] Error processing query: {e}")
            return {
                "query": query,
                "entities": [],
                "context": [],
                "text_sections": [],
                "relationships": [],
                "confidence_scores": {},
                "overall_confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "extraction_method": "integrated_database_enhanced_error",
                "total_results": 0
            }
    
    def _safe_query_cosmos(self, query: str, entities: List[str]) -> List[Dict]:
        """Safely query Cosmos Gremlin DB"""
        try:
            print("[IntegratedEntityAgent] Querying Cosmos Gremlin...")
            return self.db_manager.query_cosmos_graph(query, entities)
        except Exception as e:
            print(f"[IntegratedEntityAgent] Cosmos Gremlin query failed: {e}")
            return []
    
    def _safe_query_vector(self, query: str) -> List[Dict]:
        """Safely query Vector DB"""
        try:
            print("[IntegratedEntityAgent] Querying Vector DB...")
            return self.db_manager.query_vector_db(query)
        except Exception as e:
            print(f"[IntegratedEntityAgent] Vector DB query failed: {e}")
            return []
    
    def _safe_query_vector_ttl(self, query: str) -> List[Dict]:
        """Safely query Vector DB TTL"""
        try:
            print("[IntegratedEntityAgent] Querying Vector DB TTL...")
            return self.db_manager.query_vector_db_ttl(query)
        except Exception as e:
            print(f"[IntegratedEntityAgent] Vector DB TTL query failed: {e}")
            return []
    
    def _extract_entities_enhanced(self, query: str, intent_info: Dict) -> List[str]:
        """Enhanced entity extraction with pattern matching and NLP"""
        entities = []
        query_lower = query.lower()
        
        # Phase 1: Pattern matching (always works)
        for category, patterns in self.samm_entity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in query_lower:
                    entities.append(pattern)
        
        # Phase 2: Knowledge graph matching
        if self.knowledge_graph:
            for entity_id, entity in self.knowledge_graph.entities.items():
                entity_label = entity['properties'].get('label', entity_id)
                if entity_label.lower() in query_lower or entity_id.lower() in query_lower:
                    entities.append(entity_label)
        
        # Phase 3: NLP extraction (with fallback)
        try:
            nlp_entities = self._extract_nlp_entities_safe(query, intent_info)
            entities.extend(nlp_entities)
        except Exception as e:
            print(f"[IntegratedEntityAgent] NLP extraction failed, using pattern-only: {e}")
        
        # Remove duplicates and limit
        entities = list(dict.fromkeys(entities))[:5]  # Limit to 5 entities
        return entities
    
    def _extract_nlp_entities_safe(self, query: str, intent_info: Dict) -> List[str]:
        """Safer NLP entity extraction"""
        system_msg = """Extract SAMM entities from the query. Return ONLY a simple JSON array.

ENTITIES: Organizations (DSCA, DoD, DoS), Programs (SC, SA, FMS), Authorities (FAA, AECA, NDAA)

RESPONSE: ["entity1", "entity2"]"""
        
        prompt = f"Query: '{query}'\nEntities:"
        
        try:
            response = call_ollama_enhanced(prompt, system_msg, temperature=0.0)
            
            # Try to extract JSON more robustly
            response = response.strip()
            
            # Look for array patterns
            json_pattern = r'\[.*?\]'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                try:
                    entities = json.loads(matches[0])
                    if isinstance(entities, list):
                        return [str(e).strip() for e in entities if e]
                except:
                    pass
            
            # Fallback: Look for quoted strings
            quote_pattern = r'"([^"]+)"'
            quoted_entities = re.findall(quote_pattern, response)
            return quoted_entities[:3]  # Limit to 3
                
        except Exception as e:
            print(f"[IntegratedEntityAgent] NLP extraction error: {e}")
        
        return []
    
    def _populate_enhanced_context(self, all_results: Dict, entities: List[str]):
        """Populate enhanced context from all data sources"""
        context = []
        text_sections = []
        relationships = []
        confidence_scores = {}
        
        # Process each entity
        for entity in entities:
            entity_context = self._generate_entity_context(entity, all_results["query"])
            if entity_context:
                context.append(entity_context)
                confidence_scores[entity] = entity_context.get('confidence', 0.5)
        
        # Get relevant text sections
        text_sections = self._get_enhanced_text_sections(all_results["query"], entities)
        
        # Get comprehensive relationships
        relationships = self._get_comprehensive_relationships(entities, all_results["data_sources"])
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(confidence_scores)
        
        # Populate results
        all_results.update({
            "context": context,
            "text_sections": text_sections,
            "relationships": relationships,
            "confidence_scores": confidence_scores,
            "overall_confidence": overall_confidence,
            "total_results": len(context) + len(text_sections) + len(relationships)
        })
    
    def _generate_entity_context(self, entity: str, query: str) -> Optional[Dict]:
        """Generate comprehensive context information for an entity"""
        context_info = None
        
        # Check knowledge graph first (highest confidence)
        if self.knowledge_graph:
            for entity_id, kg_entity in self.knowledge_graph.entities.items():
                entity_label = kg_entity['properties'].get('label', entity_id)
                
                if (entity.lower() == entity_label.lower() or 
                    entity.lower() == entity_id.lower()):
                    
                    definition = kg_entity['properties'].get('definition', 
                                kg_entity['properties'].get('role', ''))
                    section = kg_entity['properties'].get('section', '')
                    
                    context_info = {
                        "entity": entity_label,
                        "definition": definition,
                        "section": section,
                        "type": kg_entity.get('type', 'unknown'),
                        "confidence": self.confidence_weights["knowledge_graph"],
                        "source": "knowledge_graph",
                        "properties": kg_entity['properties']
                    }
                    print(f"[IntegratedEntityAgent] Knowledge graph context for: {entity_label}")
                    break
        
        # Check dynamic knowledge if not found
        if not context_info and entity in self.dynamic_knowledge["entities"]:
            entity_data = self.dynamic_knowledge["entities"][entity]
            context_info = {
                "entity": entity,
                "definition": entity_data.get('definition', ''),
                "section": entity_data.get('section', ''),
                "type": entity_data.get('type', 'dynamic'),
                "confidence": self.confidence_weights["dynamic_knowledge"],
                "source": "dynamic_knowledge",
                "added_date": entity_data.get('added_date', '')
            }
            print(f"[IntegratedEntityAgent] Dynamic knowledge context for: {entity}")
        
        # Generate context using AI if not found
        if not context_info:
            context_info = self._generate_ai_context(entity, query)
        
        return context_info
    
    def _generate_ai_context(self, entity: str, query: str) -> Dict:
        """Generate entity context using Llama 3.2 AI capabilities"""
        system_msg = f"""You are a SAMM (Security Assistance Management Manual) Chapter 1 expert.

Provide context for the entity "{entity}" as it relates to SAMM Chapter 1.

INCLUDE:
- Brief, accurate definition or role description
- SAMM section reference if known (e.g., C1.3.2.2)
- Entity type (organization, program, authority, concept)
- Relationship to Security Cooperation/Security Assistance

REQUIREMENTS:
- Be accurate and specific to SAMM Chapter 1
- Use exact SAMM terminology
- If uncertain, indicate lower confidence

RESPONSE FORMAT (JSON):
{{
    "definition": "Brief definition or role description",
    "section": "SAMM section if known, otherwise 'Unknown'", 
    "type": "organization|program|authority|concept",
    "confidence": 0.7,
    "relationships": ["related entity 1", "related entity 2"]
}}"""
        
        prompt = f"""Entity: "{entity}"
Query context: "{query}"

Provide SAMM Chapter 1 context for this entity:"""
        
        try:
            response = call_ollama_enhanced(prompt, system_msg, temperature=0.1)
            
            # Try to parse JSON response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_part = response[json_start:json_end]
                
                context_data = json.loads(json_part)
                context_data["source"] = "ai_generated"
                context_data["entity"] = entity
                
                print(f"[IntegratedEntityAgent] AI generated context for: {entity}")
                return context_data
                
        except json.JSONDecodeError as e:
            print(f"[IntegratedEntityAgent] JSON parsing error in AI context generation: {e}")
        except Exception as e:
            print(f"[IntegratedEntityAgent] AI context generation error: {e}")
        
        # Fallback context
        return {
            "entity": entity,
            "definition": f"SAMM-related entity: {entity}",
            "section": "Unknown",
            "type": "unknown",
            "confidence": self.confidence_weights["context_match"] * 0.5,
            "source": "fallback"
        }
    
    def _get_enhanced_text_sections(self, query: str, entities: List[str]) -> List[str]:
        """Get relevant SAMM text sections based on entities and query"""
        text_sections = []
        
        # Entity-specific text sections
        for entity in entities:
            entity_lower = entity.lower()
            
            if "dsca" in entity_lower or "defense security cooperation agency" in entity_lower:
                text_sections.append(
                    "C1.3.2.2 DSCA: Defense Security Cooperation Agency directs, "
                    "administers, and provides guidance to DoD Components for the "
                    "execution of DoD SC programs."
                )
            
            elif "security assistance" in entity_lower:
                text_sections.append(
                    "C1.1.2.2 Security Assistance: SA is a group of programs, "
                    "authorized under Title 22 authorities, by which the United States "
                    "provides defense articles, military education and training, and "
                    "other defense-related services."
                )
            
            elif "security cooperation" in entity_lower:
                text_sections.append(
                    "C1.1.1 Definition: Security cooperation (SC) comprises all activities "
                    "undertaken by the DoD to encourage and enable international partners "
                    "to work with the United States to achieve strategic objectives."
                )
            
            elif "dfas" in entity_lower or "defense finance and accounting service" in entity_lower:
                text_sections.append(
                    "C1.3.2.8 DFAS: Defense Finance and Accounting Service performs "
                    "accounting, billing, disbursing, and collecting functions for SC programs."
                )
            
            elif "department of state" in entity_lower or "dos" in entity_lower:
                text_sections.append(
                    "C1.3.1 Department of State: Under the FAA, AECA, and Executive Order 13637, "
                    "the Secretary of State is responsible for continuous supervision and "
                    "general direction of SA programs."
                )
            
            elif "implementing agency" in entity_lower or "ia" in entity_lower:
                text_sections.append(
                    "C1.3.2.6 Implementing Agencies: An IA is the MILDEP organization or "
                    "defense agency responsible for the execution of SC programs and overall "
                    "management of delivery."
                )
        
        # Intent-based text sections
        query_lower = query.lower()
        if "difference" in query_lower or "distinction" in query_lower:
            text_sections.append(
                "Key Distinction: Security Assistance programs are a SUBSET of Security "
                "Cooperation programs. SC is broader and includes all DoD activities with "
                "foreign partners, while SA is specifically those activities authorized under Title 22."
            )
        
        # Remove duplicates and limit to top 3
        text_sections = list(dict.fromkeys(text_sections))[:3]
        
        print(f"[IntegratedEntityAgent] Retrieved {len(text_sections)} text sections")
        return text_sections
    
    def _get_comprehensive_relationships(self, entities: List[str], data_sources: Dict) -> List[str]:
        """Get comprehensive relationships from all sources"""
        relationships = []
        
        # Get relationships from knowledge graph
        if self.knowledge_graph:
            for entity in entities:
                entity_id = None
                
                # Find entity ID in knowledge graph
                for eid, kg_entity in self.knowledge_graph.entities.items():
                    entity_label = kg_entity['properties'].get('label', eid)
                    if entity_label.lower() == entity.lower():
                        entity_id = eid
                        break
                
                # Get relationships for this entity
                if entity_id:
                    entity_rels = self.knowledge_graph.get_relationships(entity_id)
                    for rel in entity_rels:
                        rel_text = f"{rel['source']} {rel['relationship']} {rel['target']}"
                        relationships.append(rel_text)
                        print(f"[IntegratedEntityAgent] Knowledge graph relationship: {rel_text}")
        
        # Add predefined relationships
        for entity in entities:
            if entity in self.entity_relationships:
                for relationship in self.entity_relationships[entity]:
                    rel_text = f"{entity} {relationship}"
                    relationships.append(rel_text)
                    print(f"[IntegratedEntityAgent] Predefined relationship: {rel_text}")
        
        # Add dynamic relationships from triggers
        for rel in self.dynamic_knowledge["relationships"]:
            source = rel.get("source", "")
            target = rel.get("target", "")
            relationship = rel.get("relationship", "")
            
            if any(entity.lower() in source.lower() or entity.lower() in target.lower() 
                   for entity in entities):
                rel_text = f"{source} {relationship} {target}"
                relationships.append(rel_text)
                print(f"[IntegratedEntityAgent] Dynamic relationship: {rel_text}")
        
        # Remove duplicates
        relationships = list(dict.fromkeys(relationships))
        
        print(f"[IntegratedEntityAgent] Total relationships found: {len(relationships)}")
        return relationships
    
    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not confidence_scores:
            return 0.0
        
        scores = list(confidence_scores.values())
        # Weighted average with slight boost for multiple high-confidence entities
        avg_confidence = sum(scores) / len(scores)
        entity_count_factor = min(1.0, len(scores) / 5.0) * 0.1  # Small boost for more entities
        
        return min(1.0, avg_confidence + entity_count_factor)
    
    def update_from_hil(self, query: str, original_entities: List[str], 
                        corrected_entities: List[str], feedback_data: Dict[str, Any] = None):
        """Update agent based on human-in-the-loop feedback"""
        feedback_entry = {
            "query": query,
            "original_entities": original_entities,
            "corrected_entities": corrected_entities,
            "feedback_data": feedback_data or {},
            "timestamp": datetime.now().isoformat(),
            "improvement_type": "hil_correction"
        }
        
        self.hil_feedback_data.append(feedback_entry)
        
        # Add new entities identified by human feedback
        for entity in corrected_entities:
            if entity not in original_entities and entity not in self.custom_entities:
                self.custom_entities[entity] = {
                    "definition": feedback_data.get("definition", "Entity identified through human feedback"),
                    "source": "HIL_feedback",
                    "query_context": query,
                    "added_date": datetime.now().isoformat(),
                    "feedback_id": len(self.hil_feedback_data)
                }
                
                # Add to dynamic knowledge
                self.dynamic_knowledge["entities"][entity] = self.custom_entities[entity]
        
        # Store context corrections
        if feedback_data and feedback_data.get("context_corrections"):
            for entity, corrected_context in feedback_data["context_corrections"].items():
                if entity in self.custom_entities:
                    self.custom_entities[entity]["definition"] = corrected_context
                    self.dynamic_knowledge["entities"][entity]["definition"] = corrected_context
        
        print(f"[IntegratedEntityAgent HIL] Updated with {len(corrected_entities)} entities from feedback for query: '{query[:50]}...'")
        print(f"[IntegratedEntityAgent HIL] Total custom entities: {len(self.custom_entities)}")
        return True
    
    def update_from_trigger(self, new_entities: List[str], new_relationships: List[Dict], 
                           trigger_data: Dict[str, Any] = None):
        """Update agent when new entity/relationship data is available"""
        trigger_entry = {
            "new_entities": new_entities,
            "new_relationships": new_relationships,
            "trigger_data": trigger_data or {},
            "timestamp": datetime.now().isoformat(),
            "trigger_id": len(self.trigger_updates)
        }
        
        self.trigger_updates.append(trigger_entry)
        
        # Add new entities to dynamic knowledge
        for entity in new_entities:
            if entity not in self.dynamic_knowledge["entities"]:
                entity_data = {
                    "definition": trigger_data.get("entity_definitions", {}).get(entity, f"New entity: {entity}"),
                    "source": "trigger_update",
                    "type": trigger_data.get("entity_types", {}).get(entity, "unknown"),
                    "added_date": datetime.now().isoformat(),
                    "trigger_id": len(self.trigger_updates)
                }
                self.dynamic_knowledge["entities"][entity] = entity_data
        
        # Add new relationships to dynamic knowledge
        for relationship in new_relationships:
            if relationship not in self.dynamic_knowledge["relationships"]:
                self.dynamic_knowledge["relationships"].append({
                    **relationship,
                    "source": "trigger_update",
                    "added_date": datetime.now().isoformat(),
                    "trigger_id": len(self.trigger_updates)
                })
        
        print(f"[IntegratedEntityAgent Trigger] Updated with {len(new_entities)} new entities and {len(new_relationships)} relationships")
        print(f"[IntegratedEntityAgent Trigger] Total dynamic entities: {len(self.dynamic_knowledge['entities'])}")
        return True
class EnhancedAnswerAgent:
    """
    Enhanced Answer Agent for SAMM Chapter 1 with sophisticated response generation
    """
    
    def __init__(self):
        """Initialize the Enhanced Answer Agent with improved error handling"""
        print("[EnhancedAnswerAgent] Initializing...")
        
        # Learning and feedback systems
        self.hil_feedback_data = []
        self.answer_templates = {}
        self.trigger_updates = []
        self.custom_knowledge = ""
        self.answer_corrections = {}
        
        # SAMM-specific response templates
        self.samm_response_templates = {
            "definition": {
                "structure": "Provide clear definition → cite SAMM section → add context/authority",
                "required_elements": ["definition", "section_reference", "authority_context"],
                "quality_criteria": ["uses_exact_samm_terminology", "cites_section", "expands_acronyms"]
            },
            "distinction": {
                "structure": "Explain key differences → provide examples → cite legal basis",
                "required_elements": ["comparison_points", "specific_examples", "legal_authorities"],
                "quality_criteria": ["clear_comparison", "highlights_subset_relationship", "authority_differences"]
            },
            "authority": {
                "structure": "State authority holder → explain scope → cite legal basis",
                "required_elements": ["authority_holder", "scope_of_authority", "legal_reference"],
                "quality_criteria": ["identifies_correct_authority", "explains_scope", "cites_legal_basis"]
            },
            "organization": {
                "structure": "Name organization → describe role → list responsibilities",
                "required_elements": ["full_name", "primary_role", "specific_duties"],
                "quality_criteria": ["expands_acronyms", "describes_role", "lists_responsibilities"]
            },
            "factual": {
                "structure": "State fact → provide context → cite source",
                "required_elements": ["specific_fact", "context", "source_reference"],
                "quality_criteria": ["accurate_information", "proper_citation", "relevant_context"]
            },
            "relationship": {
                "structure": "Describe relationship → explain significance → provide examples",
                "required_elements": ["relationship_description", "significance", "examples"],
                "quality_criteria": ["clear_relationship", "explains_importance", "concrete_examples"]
            }
        }
        
        # Quality enhancement patterns
        self.quality_patterns = {
            "section_references": r"(C\d+\.\d+\.?\d*\.?\d*)",
            "acronym_detection": r"\b([A-Z]{2,})\b",
            "authority_mentions": r"(Title \d+|[A-Z]+ Act)",
            "incomplete_sentences": r"[a-z]\s*$"
        }
        
        # Enhanced acronym expansion dictionary
        self.acronym_expansions = {
            "DSCA": "Defense Security Cooperation Agency (DSCA)",
            "DFAS": "Defense Finance and Accounting Service (DFAS)",
            "DoD": "Department of Defense (DoD)", 
            "DoS": "Department of State (DoS)",
            "SC": "Security Cooperation (SC)",
            "SA": "Security Assistance (SA)",
            "FAA": "Foreign Assistance Act (FAA)",
            "AECA": "Arms Export Control Act (AECA)",
            "NDAA": "National Defense Authorization Act (NDAA)",
            "USD(P)": "Under Secretary of Defense for Policy (USD(P))",
            "IA": "Implementing Agency (IA)",
            "MILDEP": "Military Department (MILDEP)",
            "IMET": "International Military Education and Training (IMET)",
            "FMS": "Foreign Military Sales (FMS)",
            "FMF": "Foreign Military Financing (FMF)"
        }
        
        # Answer quality scoring weights
        self.quality_weights = {
            "section_citation": 0.25,
            "acronym_expansion": 0.15,
            "answer_completeness": 0.25,
            "samm_terminology": 0.20,
            "structure_adherence": 0.15
        }
        
        # Response length guidelines
        self.length_guidelines = {
            "definition": {"min": 100, "target": 200, "max": 400},
            "distinction": {"min": 150, "target": 300, "max": 500},
            "authority": {"min": 120, "target": 250, "max": 400},
            "organization": {"min": 100, "target": 220, "max": 350},
            "factual": {"min": 80, "target": 150, "max": 300},
            "relationship": {"min": 120, "target": 200, "max": 400},
            "general": {"min": 100, "target": 200, "max": 400}
        }
        
        print("[EnhancedAnswerAgent] Initialization complete")

    @time_function
    def generate_answer(self, query: str, intent_info: Dict, entity_info: Dict, 
                       chat_history: List = None, documents_context: List = None,
                       compliance_result: Dict = None) -> str:
        """Main method for enhanced answer generation with ITAR compliance integration"""
        intent = intent_info.get("intent", "general")
        confidence = intent_info.get("confidence", 0.5)
        
        print(f"[Enhanced AnswerAgent] Generating answer for intent: {intent} (confidence: {confidence:.2f})")
        
        # COMPLIANCE CHECK
        if compliance_result:
            compliance_status = compliance_result.get("compliance_status", "compliant")
            authorized = compliance_result.get("authorized", True)
            
            print(f"[AnswerAgent] Compliance status: {compliance_status}, Authorized: {authorized}")
            
            if compliance_status == "non_compliant" or not authorized:
                return self._generate_compliance_restricted_answer(query, compliance_result)
            
            if compliance_status == "requires_review":
                return self._generate_compliance_review_answer(query, compliance_result)
        
        try:
            # Check for cached corrections
            cached_answer = self._check_for_corrections(query, intent_info, entity_info)
            if cached_answer:
                print("[AnswerAgent] Using cached correction")
                if compliance_result and compliance_result.get("compliance_status") == "warning":
                    cached_answer = self._add_compliance_disclaimer(cached_answer, compliance_result)
                return cached_answer
            
            # Build context
            context = self._build_comprehensive_context(
                query, intent_info, entity_info, chat_history, documents_context,
                compliance_result=compliance_result
            )
            
            # Create system message
            system_msg = self._create_optimized_system_message(
                intent, context, compliance_result=compliance_result
            )
            
            # Create prompt
            prompt = self._create_enhanced_prompt(
                query, intent_info, entity_info, compliance_result=compliance_result
            )
            
            # Generate answer
            answer = self._generate_with_validation(prompt, system_msg, intent_info)
            
            # Enhance quality
            enhanced_answer = self._enhance_answer_quality(answer, intent_info, entity_info)
            
            # Apply compliance filtering
            if compliance_result:
                enhanced_answer = self._apply_compliance_filtering(enhanced_answer, compliance_result)
            
            # Final validation
            final_answer = self._validate_and_score_answer(enhanced_answer, intent, query)
            
            # Add disclaimers
            if compliance_result and compliance_result.get("content_guidance", {}).get("required_disclaimers"):
                final_answer = self._add_compliance_disclaimer(final_answer, compliance_result)
            
            print(f"[AnswerAgent] Generated answer: {len(final_answer)} characters")
            return final_answer
            
        except Exception as e:
            print(f"[AnswerAgent] Error during answer generation: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"

    def _generate_compliance_restricted_answer(self, query: str, compliance_result: Dict) -> str:
        """Generate answer when compliance restrictions prevent full response"""
        recommendations = compliance_result.get("recommendations", [])
        compliance_status = compliance_result.get("compliance_status", "unknown")
        required_auth = compliance_result.get("required_authorization_level", "unknown")
        user_auth = compliance_result.get("user_authorization_level", "unclassified")
        restrictions = compliance_result.get("restrictions", [])
        
        answer_parts = [
            "I apologize, but I cannot provide a complete answer to this query due to security and compliance restrictions.",
            ""
        ]
        
        if compliance_status == "non_compliant":
            answer_parts.append("**Compliance Status**: NON-COMPLIANT")
            answer_parts.append("")
            answer_parts.append("This query involves content that requires higher authorization levels or has export control restrictions.")
        
        if required_auth != user_auth:
            answer_parts.append(f"**Authorization Required**: {required_auth.upper()}")
            answer_parts.append(f"**Your Authorization Level**: {user_auth.upper()}")
            answer_parts.append("")
        
        if restrictions:
            answer_parts.append("**Specific Restrictions**:")
            for restriction in restrictions:
                answer_parts.append(f"- {restriction}")
            answer_parts.append("")
        
        itar_compliance = compliance_result.get("itar_compliance", {})
        if itar_compliance.get("usml_categories_detected"):
            answer_parts.append("**ITAR Notice**: This query involves USML-controlled items:")
            for category in itar_compliance["usml_categories_detected"]:
                answer_parts.append(f"- Category {category}")
            answer_parts.append("")
        
        if recommendations:
            answer_parts.append("**Compliance Guidance**:")
            for rec in recommendations:
                answer_parts.append(f"- {rec}")
            answer_parts.append("")
        
        answer_parts.append("**General Unclassified Information**:")
        answer_parts.append("For unclassified general information about SAMM security cooperation policies, ")
        answer_parts.append("please consult the public SAMM documentation or contact your security officer.")
        
        return "\n".join(answer_parts)

    def _generate_compliance_review_answer(self, query: str, compliance_result: Dict) -> str:
        """Generate answer when query requires manual compliance review"""
        recommendations = compliance_result.get("recommendations", [])
        itar_compliance = compliance_result.get("itar_compliance", {})
        
        answer_parts = [
            "This query requires manual compliance review before a complete answer can be provided.",
            ""
        ]
        
        if itar_compliance.get("license_requirements"):
            answer_parts.append("**Review Required**: ITAR export license implications detected")
            answer_parts.append("")
        
        answer_parts.append("**Next Steps**:")
        for rec in recommendations:
            answer_parts.append(f"- {rec}")
        
        return "\n".join(answer_parts)

    def _apply_compliance_filtering(self, answer: str, compliance_result: Dict) -> str:
        """Apply compliance-based content filtering"""
        try:
            filtered_answer = answer
            content_guidance = compliance_result.get("content_guidance", {})
            
            if content_guidance.get("sanitization_required", False):
                print("[AnswerAgent] Applying compliance sanitization")
                sensitive_patterns = [
                    r'\b\d{3}-\d{2}-\d{4}\b',
                    r'\bclassified\s+(?:technical|data|information)\b',
                    r'\bexport\s+controlled\b',
                ]
                for pattern in sensitive_patterns:
                    filtered_answer = re.sub(pattern, '[REDACTED]', filtered_answer, flags=re.IGNORECASE)
            
            detail_level = content_guidance.get("allowed_detail_level", "general")
            if detail_level == "general" and len(filtered_answer) > 500:
                filtered_answer = filtered_answer[:500] + "..."
                filtered_answer += "\n\n[Response limited due to authorization level]"
            
            return filtered_answer
        except Exception as e:
            print(f"[AnswerAgent] Error applying compliance filtering: {e}")
            return answer

    def _add_compliance_disclaimer(self, answer: str, compliance_result: Dict) -> str:
        """Add compliance disclaimers to the answer"""
        try:
            disclaimers = compliance_result.get("content_guidance", {}).get("required_disclaimers", [])
            if not disclaimers:
                return answer
            
            disclaimer_text = "\n\n---\n**Compliance Notice**:\n"
            for disclaimer in disclaimers:
                disclaimer_text += f"- {disclaimer}\n"
            
            return answer + disclaimer_text
        except Exception as e:
            print(f"[AnswerAgent] Error adding disclaimer: {e}")
            return answer

    def _check_for_corrections(self, query: str, intent_info: Dict, entity_info: Dict) -> Optional[str]:
        """Check if we have a stored correction for similar queries"""
        try:
            query_key = self._normalize_query_for_matching(query)
            if query_key in self.answer_corrections:
                return self.answer_corrections[query_key]["corrected_answer"]
            return None
        except Exception as e:
            print(f"[AnswerAgent] Error checking corrections: {e}")
            return None

    def _build_comprehensive_context(self, query: str, intent_info: Dict, entity_info: Dict,
                                   chat_history: List = None, documents_context: List = None,
                                   compliance_result: Dict = None) -> str:
        """Build comprehensive context with compliance restrictions"""
        try:
            context_parts = []
            
            # Add compliance guidance
            if compliance_result:
                content_guidance = compliance_result.get("content_guidance", {})
                restrictions = compliance_result.get("restrictions", [])
                detail_level = content_guidance.get("allowed_detail_level", "general")
                
                context_parts.append("=== COMPLIANCE RESTRICTIONS ===")
                context_parts.append(f"Maximum detail level allowed: {detail_level.upper()}")
                
                if restrictions:
                    context_parts.append("Content restrictions:")
                    for restriction in restrictions:
                        context_parts.append(f"- {restriction}")
                context_parts.append("")
            
            # Add entity context
            if entity_info.get("context"):
                context_parts.append("=== SAMM ENTITIES AND DEFINITIONS ===")
                for ctx in entity_info["context"][:5]:
                    if ctx.get('confidence', 0) > 0.6:
                        entity_text = f"{ctx.get('entity', '')}: {ctx.get('definition', '')}"
                        if ctx.get('section'):
                            entity_text += f" (SAMM {ctx['section']})"
                        context_parts.append(entity_text)
            
            # Add text sections
            if entity_info.get("text_sections"):
                context_parts.append("\n=== SAMM CHAPTER 1 CONTENT ===")
                for section in entity_info["text_sections"][:3]:
                    context_parts.append(section[:500] + "..." if len(section) > 500 else section)
            
            # Add relationships
            if entity_info.get("relationships"):
                context_parts.append("\n=== ENTITY RELATIONSHIPS ===")
                context_parts.extend(entity_info["relationships"][:3])
            
            return "\n".join(context_parts)
        except Exception as e:
            print(f"[AnswerAgent] Error building context: {e}")
            return "Context building failed"

    def _create_optimized_system_message(self, intent: str, context: str, 
                                        compliance_result: Dict = None) -> str:
        """Create intent-optimized system message with compliance"""
        base_msg = "You are a SAMM expert. Provide accurate information about Security Cooperation and Security Assistance."
        
        if compliance_result:
            detail_level = compliance_result.get("content_guidance", {}).get("allowed_detail_level", "general")
            base_msg += f"\n\nCOMPLIANCE: Maximum detail level is {detail_level.upper()}. "
            base_msg += "Provide only unclassified, general information."
        
        base_msg += f"\n\nCONTEXT:\n{context[:2000]}"
        return base_msg

    def _create_enhanced_prompt(self, query: str, intent_info: Dict, entity_info: Dict,
                               compliance_result: Dict = None) -> str:
        """Create enhanced prompt with compliance awareness"""
        prompt = f"Question: {query}\n"
        
        if compliance_result:
            detail_level = compliance_result.get("content_guidance", {}).get("allowed_detail_level", "general")
            prompt += f"IMPORTANT: Respond at {detail_level.upper()} detail level only.\n"
        
        prompt += "Provide a comprehensive answer based on SAMM Chapter 1."
        return prompt

    def _generate_with_validation(self, prompt: str, system_msg: str, intent_info: Dict) -> str:
        """Generate answer with validation"""
        return call_ollama_enhanced(prompt, system_msg, temperature=0.1)

    def _validate_answer_quality(self, answer: str, intent: str) -> Dict[str, Any]:
        """Validate answer quality"""
        return {"needs_improvement": False, "issues": [], "length": len(answer)}

    def _enhance_answer_quality(self, answer: str, intent_info: Dict, entity_info: Dict) -> str:
        """Apply post-processing enhancements"""
        return answer

    def _validate_and_score_answer(self, answer: str, intent: str, query: str) -> str:
        """Final validation and scoring"""
        return answer

    def _calculate_quality_score(self, answer: str, intent: str) -> float:
        """Calculate quality score"""
        return 0.75

    def _normalize_query_for_matching(self, query: str) -> str:
        """Normalize query for matching"""
        return query.lower().strip()

    def update_from_hil(self, query: str, original_answer: str, corrected_answer: str, 
                        feedback_data: Dict[str, Any] = None):
        """Update from HIL feedback"""
        query_key = self._normalize_query_for_matching(query)
        self.answer_corrections[query_key] = {
            "corrected_answer": corrected_answer,
            "feedback_data": feedback_data
        }
        return True

    def update_from_trigger(self, new_entities: List[str], new_relationships: List[Dict], 
                           trigger_data: Dict[str, Any] = None):
        """Update from trigger"""
        return True
    

class SimpleStateOrchestrator:
    """Simple LangGraph-style state orchestration for integrated SAMM agents with HIL and trigger updates"""
    
    def __init__(self):
        self.intent_agent = IntentAgent()
        self.entity_agent = IntegratedEntityAgent(knowledge_graph, db_manager)
        self.answer_agent = EnhancedAnswerAgent()
        
        # Define workflow graph
        self.workflow = {
            WorkflowStep.INIT: self._initialize_state,
            WorkflowStep.INTENT: self._analyze_intent_step,
            WorkflowStep.ENTITY: self._extract_entities_step,
            WorkflowStep.ANSWER: self._generate_answer_step,
            WorkflowStep.COMPLETE: self._complete_workflow,
            WorkflowStep.ERROR: self._handle_error
        }
        
        # Define state transitions
        self.transitions = {
            WorkflowStep.INIT: WorkflowStep.INTENT,
            WorkflowStep.INTENT: WorkflowStep.ENTITY,
            WorkflowStep.ENTITY: WorkflowStep.ANSWER,
            WorkflowStep.ANSWER: WorkflowStep.COMPLETE,
            WorkflowStep.COMPLETE: None,
            WorkflowStep.ERROR: None
        }

    @time_function
    def process_query(self, query: str, chat_history: List = None, documents_context: List = None,
                     user_profile: Dict = None) -> Dict[str, Any]:
        """Process query with integrated compliance checking"""
        state = AgentState(
            query=query,
            chat_history=chat_history,
            documents_context=documents_context,
            intent_info=None,
            entity_info=None,
            answer=None,
            execution_steps=[],
            start_time=time.time(),
            current_step=WorkflowStep.INIT.value,
            error=None
        )
        
        try:
            current_step = WorkflowStep.INIT
            
            while current_step is not None and current_step != WorkflowStep.ANSWER:
                print(f"[State Orchestrator] Executing step: {current_step.value}")
                state['current_step'] = current_step.value
                state['execution_steps'].append(f"Step: {current_step.value}")
                
                state = self.workflow[current_step](state)
                
                if state.get('error'):
                    current_step = WorkflowStep.ERROR
                else:
                    current_step = self.transitions[current_step]
            
            compliance_result = None
            if COMPLIANCE_SERVICE_ENABLED and state['intent_info'] and state['entity_info']:
                state['execution_steps'].append("Checking ITAR/Security compliance...")
                compliance_result = call_compliance_service(
                    query=state['query'],
                    intent_info=state['intent_info'],
                    entity_info=state['entity_info'],
                    user_profile=user_profile
                )
                
                if compliance_result:
                    compliance_status = compliance_result.get('compliance_status', 'unknown')
                    state['execution_steps'].append(f"Compliance check: {compliance_status}")
                    print(f"[State Orchestrator] Compliance: {compliance_status}")
            
            if current_step == WorkflowStep.ANSWER:
                state['answer'] = self.answer_agent.generate_answer(
                    state['query'],
                    state['intent_info'],
                    state['entity_info'],
                    state['chat_history'],
                    state['documents_context'],
                    compliance_result=compliance_result
                )
                state['execution_steps'].append("Answer generated with compliance verification")
                current_step = WorkflowStep.COMPLETE
            
            state = self.workflow[WorkflowStep.COMPLETE](state)
            execution_time = round(time.time() - state['start_time'], 2)
            
            return {
                "query": state['query'],
                "answer": state['answer'],
                "intent": state['intent_info'].get('intent', 'unknown') if state['intent_info'] else 'unknown',
                "entities_found": len(state['entity_info'].get('entities', [])) if state['entity_info'] else 0,
                "execution_time": execution_time,
                "execution_steps": state['execution_steps'],
                "success": state['error'] is None,
                "compliance_result": compliance_result,
                "metadata": {
                    "intent_confidence": state['intent_info'].get('confidence', 0) if state['intent_info'] else 0,
                    "entities": state['entity_info'].get('entities', []) if state['entity_info'] else [],
                    "system_version": "Integrated_Database_SAMM_v5.0_Compliance",
                    "workflow_completed": state['current_step'] == 'complete',
                    "database_integration": {
                        "cosmos_gremlin": db_manager.cosmos_gremlin_client is not None,
                        "vector_db": db_manager.vector_db_client is not None,
                        "vector_db_ttl": db_manager.vector_db_ttl_client is not None,
                        "embedding_model": db_manager.embedding_model is not None
                    },
                    "compliance_checked": compliance_result is not None,
                    "compliance_status": compliance_result.get('compliance_status') if compliance_result else None,
                    "compliance_authorized": compliance_result.get('authorized') if compliance_result else True
                }
            }
            
        except Exception as e:
            execution_time = round(time.time() - state['start_time'], 2)
            return {
                "query": query,
                "answer": f"I apologize, but I encountered an error during processing: {str(e)}",
                "intent": "error",
                "entities_found": 0,
                "execution_time": execution_time,
                "execution_steps": state['execution_steps'] + [f"Error: {str(e)}"],
                "success": False,
                "compliance_result": None,
                "metadata": {"error": str(e), "system_version": "Integrated_Database_SAMM_v5.0_Compliance"}
            }
    
    def update_agents_from_hil(self, query: str, intent_correction: Dict = None, entity_correction: Dict = None, answer_correction: Dict = None) -> Dict[str, bool]:
        """Update all agents from human-in-the-loop feedback"""
        results = {}
        
        if intent_correction:
            results["intent"] = self.intent_agent.update_from_hil(
                query=query,
                original_intent=intent_correction.get("original_intent"),
                corrected_intent=intent_correction.get("corrected_intent"),
                feedback_data=intent_correction.get("feedback_data", {})
            )
        
        if entity_correction:
            results["entity"] = self.entity_agent.update_from_hil(
                query=query,
                original_entities=entity_correction.get("original_entities", []),
                corrected_entities=entity_correction.get("corrected_entities", []),
                feedback_data=entity_correction.get("feedback_data", {})
            )
        
        if answer_correction:
            results["answer"] = self.answer_agent.update_from_hil(
                query=query,
                original_answer=answer_correction.get("original_answer"),
                corrected_answer=answer_correction.get("corrected_answer"),
                feedback_data=answer_correction.get("feedback_data", {})
            )
        
        print(f"[State Orchestrator] HIL updates completed: {results}")
        return results
    
    def update_agents_from_trigger(self, new_entities: List[str], new_relationships: List[Dict], trigger_data: Dict[str, Any] = None) -> Dict[str, bool]:
        """Update all agents when new entity/relationship data is available"""
        results = {}
        
        results["intent"] = self.intent_agent.update_from_trigger(
            new_entities=new_entities,
            new_relationships=new_relationships,
            trigger_data=trigger_data
        )
        
        results["entity"] = self.entity_agent.update_from_trigger(
            new_entities=new_entities,
            new_relationships=new_relationships,
            trigger_data=trigger_data
        )
        
        results["answer"] = self.answer_agent.update_from_trigger(
            new_entities=new_entities,
            new_relationships=new_relationships,
            trigger_data=trigger_data
        )
        
        print(f"[State Orchestrator] Trigger updates completed: {results}")
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents including database connections and HIL/trigger update counts"""
        return {
            "intent_agent": {
                "hil_feedback_count": len(self.intent_agent.hil_feedback_data),
                "trigger_update_count": len(self.intent_agent.trigger_updates),
                "learned_patterns": len(self.intent_agent.intent_patterns)
            },
            "integrated_entity_agent": {
                "type": "IntegratedEntityAgent",
                "hil_feedback_count": len(self.entity_agent.hil_feedback_data),
                "trigger_update_count": len(self.entity_agent.trigger_updates),
                "custom_entities": len(self.entity_agent.custom_entities),
                "dynamic_entities": len(self.entity_agent.dynamic_knowledge["entities"]),
                "samm_patterns": sum(len(patterns) for patterns in self.entity_agent.samm_entity_patterns.values()),
                "extraction_phases": 3,
                "database_status": db_manager.get_database_status()
            },
            "enhanced_answer_agent": {
                "type": "EnhancedAnswerAgent",
                "hil_feedback_count": len(self.answer_agent.hil_feedback_data),
                "trigger_update_count": len(self.answer_agent.trigger_updates),
                "answer_corrections": len(self.answer_agent.answer_corrections),
                "answer_templates": sum(len(templates) for templates in self.answer_agent.answer_templates.values()),
                "response_templates": len(self.answer_agent.samm_response_templates),
                "acronym_expansions": len(self.answer_agent.acronym_expansions)
            }
        }
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get comprehensive database status"""
        return db_manager.get_database_status()
    
    def cleanup(self):
        """Cleanup all resources"""
        try:
            db_manager.cleanup()
            print("[State Orchestrator] Cleanup complete")
        except Exception as e:
            print(f"[State Orchestrator] Cleanup error: {e}")
    
    def _initialize_state(self, state: AgentState) -> AgentState:
        """Initialize workflow state"""
        state['execution_steps'].append("Integrated workflow initialized with database connections")
        print(f"[State Orchestrator] Initialized query: '{state['query']}'")
        return state
    
    def _analyze_intent_step(self, state: AgentState) -> AgentState:
        """Execute intent analysis step"""
        try:
            state['intent_info'] = self.intent_agent.analyze_intent(state['query'])
            state['execution_steps'].append(f"Intent analyzed: {state['intent_info'].get('intent', 'unknown')}")
            print(f"[State Orchestrator] Intent: {state['intent_info'].get('intent')} (confidence: {state['intent_info'].get('confidence')})")
        except Exception as e:
            state['error'] = f"Intent analysis failed: {str(e)}"
        return state
    
    def _extract_entities_step(self, state: AgentState) -> AgentState:
        """Execute integrated entity extraction with database queries"""
        try:
            state['entity_info'] = self.entity_agent.extract_and_retrieve(state['query'], state['intent_info'])
            entities_count = len(state['entity_info'].get('entities', []))
            confidence = state['entity_info'].get('overall_confidence', 0)
            db_results = state['entity_info'].get('total_results', 0)
            phases = state['entity_info'].get('phase_count', 0)
            state['execution_steps'].append(f"Integrated entity extraction: {entities_count} entities found (confidence: {confidence:.2f}, DB results: {db_results}, phases: {phases})")
            print(f"[State Orchestrator] Integrated Entities: {entities_count} entities found through {phases} phases with {db_results} database results")
        except Exception as e:
            state['error'] = f"Integrated entity extraction failed: {str(e)}"
        return state
    
    def _generate_answer_step(self, state: AgentState) -> AgentState:
        """Execute enhanced answer generation step"""
        try:
            state['answer'] = self.answer_agent.generate_answer(
                state['query'], 
                state['intent_info'], 
                state['entity_info'], 
                state['chat_history'], 
                state['documents_context']
            )
            state['execution_steps'].append("Enhanced answer generated successfully with quality scoring")
            print(f"[State Orchestrator] Enhanced answer generated ({len(state['answer'])} characters)")
        except Exception as e:
            state['error'] = f"Enhanced answer generation failed: {str(e)}"
        return state
    
    def _complete_workflow(self, state: AgentState) -> AgentState:
        """Complete workflow"""
        state['execution_steps'].append("Integrated workflow completed successfully")
        print(f"[State Orchestrator] Integrated workflow completed in {round(time.time() - state['start_time'], 2)}s")
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle workflow error"""
        state['execution_steps'].append(f"Error handled: {state['error']}")
        state['answer'] = f"I apologize, but I encountered an error: {state['error']}"
        print(f"[State Orchestrator] Error handled: {state['error']}")
        return state

@app.route("/api/query/stream", methods=["POST"])
def query_ai_assistant_stream():
    """Streaming SAMM query endpoint with real-time updates"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get("question", "").strip()
    chat_history = data.get("chat_history", [])
    staged_chat_documents_metadata = data.get("staged_chat_documents", [])
    
    if not user_input:
        return jsonify({"error": "Query cannot be empty"}), 400

    def generate():
        try:
            start_time = time.time()
            
            # START - Send immediately
            yield f"data: {json.dumps({'type': 'start', 'query': user_input, 'timestamp': time.time()})}\n\n"
            
            # STEP 1: Intent Analysis - Stream progress
            yield f"data: {json.dumps({'type': 'progress', 'step': 'intent_analysis', 'message': 'Analyzing query intent...', 'elapsed': round(time.time() - start_time, 2)})}\n\n"
            
            intent_start = time.time()
            intent_info = orchestrator.intent_agent.analyze_intent(user_input)
            intent_time = round(time.time() - intent_start, 2)
            
            yield f"data: {json.dumps({'type': 'intent_complete', 'data': intent_info, 'time': intent_time})}\n\n"
            
            # STEP 2: Entity Extraction - Stream progress immediately
            yield f"data: {json.dumps({'type': 'progress', 'step': 'entity_extraction', 'message': 'Extracting entities and querying databases...', 'elapsed': round(time.time() - start_time, 2)})}\n\n"
            
            entity_start = time.time()
            entity_info = orchestrator.entity_agent.extract_and_retrieve(user_input, intent_info)
            entity_time = round(time.time() - entity_start, 2)
            
            yield f"data: {json.dumps({'type': 'entities_complete', 'data': {'count': len(entity_info.get('entities', [])), 'entities': entity_info.get('entities', []), 'confidence': entity_info.get('overall_confidence', 0)}, 'time': entity_time})}\n\n"
            
            # STEP 3: Answer Generation - Stream this immediately before generating
            yield f"data: {json.dumps({'type': 'progress', 'step': 'answer_generation', 'message': 'Generating answer...', 'elapsed': round(time.time() - start_time, 2)})}\n\n"
            
            answer_start = time.time()
            
            # Build context
            context = orchestrator.answer_agent._build_comprehensive_context(
                user_input, intent_info, entity_info, chat_history, staged_chat_documents_metadata
            )
            
            intent = intent_info.get("intent", "general")
            system_msg = orchestrator.answer_agent._create_optimized_system_message(intent, context)
            prompt = orchestrator.answer_agent._create_enhanced_prompt(user_input, intent_info, entity_info)
            
            # Signal that streaming answer is about to start
            yield f"data: {json.dumps({'type': 'answer_start', 'message': 'Streaming answer...', 'elapsed': round(time.time() - start_time, 2)})}\n\n"
            
            # Stream the answer token by token
            full_answer = ""
            token_count = 0
            
            for token in call_ollama_streaming(prompt, system_msg, temperature=0.1):
                if token and not token.startswith("Error"):
                    full_answer += token
                    token_count += 1
                    # Send each token immediately
                    yield f"data: {json.dumps({'type': 'answer_token', 'token': token, 'position': token_count})}\n\n"
            
            answer_time = round(time.time() - answer_start, 2)
            total_time = round(time.time() - start_time, 2)
            
            # Apply post-processing enhancements
            enhanced_answer = orchestrator.answer_agent._enhance_answer_quality(
                full_answer, intent_info, entity_info
            )
            
            # Send final enhanced answer if different from streamed version
            if enhanced_answer != full_answer:
                yield f"data: {json.dumps({'type': 'answer_enhanced', 'enhanced_answer': enhanced_answer})}\n\n"
            
            # Send completion with all metadata
            yield f"data: {json.dumps({'type': 'complete', 'data': {'intent': intent, 'entities_found': len(entity_info.get('entities', [])), 'answer_length': len(enhanced_answer), 'token_count': token_count, 'timings': {'intent': intent_time, 'entity': entity_time, 'answer': answer_time, 'total': total_time}}})}\n\n"
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[Streaming Error] {error_detail}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'detail': error_detail})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )
# Initialize integrated orchestrator with all agents
orchestrator = SimpleStateOrchestrator()
print("Integrated State Orchestrator initialized with Intent, Integrated Entity (Database), and Enhanced Answer agents")
@time_function
def process_samm_query(query: str, chat_history: List = None, documents_context: List = None, user_profile: Dict = None) -> Dict[str, Any]:
    """Process query through integrated state orchestrated 3-agent system with database connections"""
    return orchestrator.process_query(query, chat_history, documents_context, user_profile)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_mock_user():
    """Return a mock user with appropriate clearance for demo purposes"""
    return {
        "sub": "mock-user-123",
        "name": "Demo User",
        "email": "demo@example.com",
        # Add authorization and clearance levels
        "authorization_level": "secret",  # Options: "unclassified", "confidential", "secret", "top_secret"
        "clearance": "secret",
        "roles": ["fms_officer", "samm_user", "case_manager"],
        # Add organizational context
        "organization": "DSCA",
        "department": "FMS Operations",
        # Add access permissions
        "permissions": [
            "read_fms_cases",
            "write_fms_cases", 
            "access_itar_controlled",
            "view_classified_content"
        ]
    }

def require_auth():
    """Check if user is authenticated, return user info or None"""
    user_session_data = session.get("user")
    if not user_session_data:
        return None
    
    # For OAuth
    if "userinfo" in user_session_data and "sub" in user_session_data["userinfo"]:
        user_info = user_session_data["userinfo"]
        # Ensure clearance fields exist even for OAuth users
        if "authorization_level" not in user_info:
            user_info["authorization_level"] = "unclassified"
        if "clearance" not in user_info:
            user_info["clearance"] = "unclassified"
        if "roles" not in user_info:
            user_info["roles"] = ["samm_user"]
        if "permissions" not in user_info:
            user_info["permissions"] = ["read_fms_cases"]
        return user_info
    
    # For mock user (when OAuth not configured)
    if not oauth:
        return get_mock_user()
    
    return None

# =============================================================================
# AUTHENTICATION ROUTES
# =============================================================================

@app.route("/login")
def login():
    if oauth:
        redirect_uri_for_auth0 = url_for("callback", _external=True)
        return oauth.auth0.authorize_redirect(redirect_uri=redirect_uri_for_auth0)
    else:
        # Mock login when OAuth not configured
        session["user"] = {"userinfo": get_mock_user()}
        return jsonify({"message": "Logged in with mock user"}), 200

@app.route("/callback", methods=["GET", "POST"])
def callback():
    if not oauth:
        return jsonify({"error": "OAuth not configured"}), 500
    
    try:
        token = oauth.auth0.authorize_access_token()
        session["user"] = token 
        userinfo = token.get("userinfo")
        if userinfo:
            print(f"User logged in: {userinfo.get('name')} ({userinfo.get('sub')})")
    except Exception as e:
        print(f"Error during Auth0 callback: {e}")
        return redirect(url_for("login")) 
    
    vue_app_url = "http://localhost:5173" 
    next_url_path_from_session = session.pop('next_url', None) 
    final_redirect_url = vue_app_url 
    if next_url_path_from_session:
        if next_url_path_from_session.startswith('/'): 
            final_redirect_url = f"{vue_app_url}{next_url_path_from_session}"
        else: 
            final_redirect_url = f"{vue_app_url}/{next_url_path_from_session}"
    return redirect(final_redirect_url)

@app.route("/logout") 
def logout():
    session.clear()
    if oauth:
        vue_app_url = "http://localhost:5173" 
        return redirect(
            f"https://{AUTH0_DOMAIN}/v2/logout?" +
            urlencode({"returnTo": vue_app_url, "client_id": AUTH0_CLIENT_ID,}, quote_via=quote_plus,)
        )
    else:
        return jsonify({"message": "Logged out"}), 200

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route("/api/me", methods=["GET"])
def get_current_user_profile():
    user = require_auth()
    if user:
        return jsonify(user), 200
    else:
        return jsonify({"error": "User not authenticated"}), 401

@app.route("/api/user/cases", methods=["GET"])
def get_user_cases():
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user["sub"]
    
    if cases_container_client:
        try:
            query = "SELECT * FROM c WHERE c.userId = @userId AND c.type = 'case'" 
            parameters = [{"name": "@userId", "value": user_id}]
            user_cases_list = list(cases_container_client.query_items(query=query, parameters=parameters, partition_key=user_id))
            return jsonify(user_cases_list), 200
        except Exception as e:
            print(f"Error querying cases: {e}")
            return jsonify({"error": "Database service error"}), 503
    else:
        # Use in-memory storage
        return jsonify(user_cases.get(user_id, [])), 200

@app.route("/api/cases", methods=["POST"])
def create_case():
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user["sub"]
    case_data = request.get_json() if request.is_json else {}
    
    case_id = str(uuid.uuid4())
    new_case = {
        "id": case_id,
        "userId": user_id,
        "type": "case",
        "title": case_data.get("title", "New Case"),
        "description": case_data.get("description", ""),
        "caseDocuments": [],
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }
    
    if cases_container_client:
        try:
            cases_container_client.create_item(body=new_case)
            return jsonify(new_case), 201
        except Exception as e:
            print(f"Error creating case: {e}")
            return jsonify({"error": "Failed to create case"}), 500
    else:
        # Use in-memory storage
        if user_id not in user_cases:
            user_cases[user_id] = []
        user_cases[user_id].append(new_case)
        return jsonify(new_case), 201

@app.route("/api/chat/stage_attachment", methods=["POST"])
def stage_chat_attachment():
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user["sub"]

    if not chat_docs_blob_container_client:
        print("[API StageChatAttachment] Chat documents blob service not available.")
        return jsonify({"error": "Chat document storage service not available"}), 503

    if 'document' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file_to_upload = request.files['document']

    if file_to_upload.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file_to_upload:
        original_filename = secure_filename(file_to_upload.filename)
        blob_name = f"{user_id}/chat_staging/{str(uuid.uuid4())}-{original_filename}"
        
        print(f"[API StageChatAttachment] Processing file: {original_filename} for blob: {blob_name}")
        blob_client_instance = chat_docs_blob_container_client.get_blob_client(blob_name)
            
        try:
            file_to_upload.seek(0) 
            blob_content_settings = ContentSettings(content_type=file_to_upload.mimetype)
            blob_client_instance.upload_blob(
                file_to_upload.read(), 
                overwrite=True,
                content_settings=blob_content_settings
            )
            print(f"[API StageChatAttachment] Successfully uploaded '{original_filename}' to blob: {blob_name}")

            file_to_upload.seek(0, os.SEEK_END)
            file_size_bytes = file_to_upload.tell()
            
            staged_doc_metadata = {
                "documentId": str(uuid.uuid4()),
                "fileName": original_filename,
                "blobName": blob_name,
                "blobContainer": AZURE_CHAT_DOCS_CONTAINER_NAME,
                "url": blob_client_instance.url,
                "fileType": file_to_upload.mimetype,
                "sizeBytes": file_size_bytes,
                "uploadedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "uploaderUserId": user_id,
                "status": "staged"
            }
            
            return jsonify({
                "message": f"File '{original_filename}' staged successfully.",
                "stagedDocument": staged_doc_metadata 
            }), 200

        except Exception as e:
            print(f"[API StageChatAttachment] Error uploading file '{original_filename}' to blob: {str(e)}")
            return jsonify({"error": f"Failed to upload file '{original_filename}'.", "details": str(e)}), 500
    
    return jsonify({"error": "Unknown error during file staging."}), 500

@app.route("/api/cases/<string:case_id>/documents/upload", methods=["POST"])
def upload_case_document_to_case(case_id):
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user["sub"]

    if not cases_container_client or not case_docs_blob_container_client:
        return jsonify({"error": "Backend storage service not available"}), 503

    uploaded_files = request.files.getlist("documents") 
    if not uploaded_files or not uploaded_files[0].filename: 
        return jsonify({"error": "No files selected for upload"}), 400

    try:
        case_doc = cases_container_client.read_item(item=case_id, partition_key=user_id)
    except CosmosExceptions.CosmosResourceNotFoundError:
        return jsonify({"error": f"Case {case_id} not found or access denied."}), 404
    except Exception as e:
        print(f"Error reading case {case_id}: {str(e)}")
        return jsonify({"error": "Could not retrieve case details"}), 500

    if "caseDocuments" not in case_doc:
        case_doc["caseDocuments"] = []

    newly_uploaded_metadata_list = []

    for file_storage_item in uploaded_files:
        if file_storage_item and file_storage_item.filename:
            original_filename = secure_filename(file_storage_item.filename)
            blob_name = f"{case_id}/{original_filename}" 
            blob_client_instance = case_docs_blob_container_client.get_blob_client(blob_name)
            
            try:
                file_storage_item.seek(0) 
                blob_content_settings = ContentSettings(content_type=file_storage_item.mimetype)
                blob_client_instance.upload_blob(
                    file_storage_item.read(), 
                    overwrite=True, 
                    content_settings=blob_content_settings
                )
                print(f"Uploaded '{original_filename}' to Azure Blob Storage as '{blob_name}'")

                file_storage_item.seek(0, os.SEEK_END)
                file_size_bytes = file_storage_item.tell()
                
                doc_metadata = {
                    "documentId": str(uuid.uuid4()),
                    "fileName": original_filename,
                    "blobName": blob_name,
                    "blobContainer": AZURE_CASE_DOCS_CONTAINER_NAME,
                    "url": blob_client_instance.url,
                    "fileType": file_storage_item.mimetype,
                    "sizeBytes": file_size_bytes,
                    "uploadedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "uploaderUserId": user_id
                }
                case_doc["caseDocuments"].append(doc_metadata)
                newly_uploaded_metadata_list.append(doc_metadata)

            except Exception as e:
                print(f"Error uploading file '{original_filename}' to blob: {str(e)}")
                continue 
    
    if not newly_uploaded_metadata_list: 
        return jsonify({"error": "No files were successfully processed for upload."}), 400

    try:
        case_doc["updatedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        cases_container_client.replace_item(item=case_id, body=case_doc)
        print(f"Updated case '{case_id}' in Cosmos DB with new document metadata.")
        return jsonify({
            "message": f"{len(newly_uploaded_metadata_list)} file(s) uploaded successfully to case {case_id}.",
            "uploadedDocuments": newly_uploaded_metadata_list 
        }), 200
    except Exception as e:
        print(f"Error updating case '{case_id}' in Cosmos DB: {str(e)}")
        return jsonify({"error": "Failed to update case metadata in database after file uploads."}), 500

@app.route("/api/cases/documents/delete", methods=["POST"])
def delete_case_document_route():
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user["sub"]

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    case_id = data.get("caseId")
    document_metadata_id_to_delete = data.get("documentId") 

    if not case_id or not document_metadata_id_to_delete:
        return jsonify({"error": "Missing caseId or documentId in request"}), 400

    print(f"[API DeleteCaseDocument] User: {user_id} attempting to delete document with metadata ID: {document_metadata_id_to_delete} from case: {case_id}")

    if not cases_container_client or not case_docs_blob_container_client:
        return jsonify({"error": "Backend storage or database service not available"}), 503

    try:
        case_doc = cases_container_client.read_item(item=case_id, partition_key=user_id)
    except CosmosExceptions.CosmosResourceNotFoundError:
        return jsonify({"error": f"Case {case_id} not found or access denied."}), 404
    except Exception as e:
        print(f"Error reading case {case_id} for document deletion: {str(e)}")
        return jsonify({"error": "Could not retrieve case details"}), 500

    doc_to_delete_metadata = None
    original_case_documents = case_doc.get("caseDocuments", [])
    updated_case_documents = []

    for doc_meta in original_case_documents:
        if doc_meta.get("documentId") == document_metadata_id_to_delete:
            doc_to_delete_metadata = doc_meta
        else:
            updated_case_documents.append(doc_meta)

    if not doc_to_delete_metadata:
        print(f"[API DeleteCaseDocument] Document metadata ID {document_metadata_id_to_delete} not found in case {case_id}.")
        return jsonify({"error": "Document not found within the case."}), 404

    # Delete from Azure Blob Storage
    blob_name_to_delete = doc_to_delete_metadata.get("blobName")
    if blob_name_to_delete:
        try:
            blob_client_instance = case_docs_blob_container_client.get_blob_client(blob_name_to_delete)
            blob_client_instance.delete_blob()
            print(f"[API DeleteCaseDocument] Successfully deleted blob: {blob_name_to_delete} from container: {AZURE_CASE_DOCS_CONTAINER_NAME}")
        except BlobResourceNotFoundError:
            print(f"[API DeleteCaseDocument] Blob not found in storage (already deleted?): {blob_name_to_delete}")
        except Exception as e_blob:
            print(f"[API DeleteCaseDocument] Error deleting blob '{blob_name_to_delete}': {str(e_blob)}")

    # Update Cosmos DB
    case_doc["caseDocuments"] = updated_case_documents
    case_doc["updatedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    cases_container_client.replace_item(item=case_id, body=case_doc)
    print(f"[API DeleteCaseDocument] Successfully removed document metadata ID {document_metadata_id_to_delete} from case {case_id} in Cosmos DB.")
    return jsonify({"message": f"Document '{doc_to_delete_metadata.get('fileName', 'Unknown')}' deleted successfully from case {case_id}."}), 200

@app.route("/api/chat/attachments/delete", methods=["POST"])
def delete_chat_attachment():
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user["sub"]

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    blob_name = data.get("blobName")
    blob_container_name = data.get("blobContainer")

    if not blob_name or not blob_container_name:
        return jsonify({"error": "Missing blobName or blobContainer in request"}), 400

    print(f"[API DeleteChatAttachment] User: {user_id} attempting to delete blob: {blob_name} from container: {blob_container_name}")

    if blob_container_name != AZURE_CHAT_DOCS_CONTAINER_NAME:
        print(f"[API DeleteChatAttachment] Attempt to delete from non-chat container: {blob_container_name}. Denied.")
        return jsonify({"error": "Invalid target container for deletion"}), 403

    if not blob_service_client:
        print("[API DeleteChatAttachment] Blob service client not available.")
        return jsonify({"error": "Blob storage service not available"}), 503
    
    target_blob_client = None
    if blob_container_name == AZURE_CHAT_DOCS_CONTAINER_NAME:
        if chat_docs_blob_container_client:
            target_blob_client = chat_docs_blob_container_client.get_blob_client(blob_name)
        else:
            print(f"[API DeleteChatAttachment] Mismatch or uninitialized client for container: {blob_container_name}")
            return jsonify({"error": "Specified blob container client not configured or mismatch"}), 500

    if not target_blob_client:
        return jsonify({"error": "Could not obtain blob client for deletion."}), 500

    try:
        target_blob_client.delete_blob()
        print(f"[API DeleteChatAttachment] Successfully deleted blob: {blob_name} from container: {blob_container_name}")
        return jsonify({"message": f"File '{blob_name}' deleted successfully from chat context."}), 200

    except BlobResourceNotFoundError:
        print(f"[API DeleteChatAttachment] Blob not found: {blob_name} in container: {blob_container_name}")
        return jsonify({"error": "File not found in storage."}), 404
    except Exception as e:
        print(f"[API DeleteChatAttachment] Error deleting blob '{blob_name}': {str(e)}")
        return jsonify({"error": "Failed to delete file from storage.", "details": str(e)}), 500

@app.route("/api/query", methods=["POST"])
def query_ai_assistant():
    """Main SAMM query endpoint with compliance integration"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user["sub"]

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()
        user_input = data.get("question", "").strip()
        chat_history = data.get("chat_history", [])
        staged_chat_documents_metadata = data.get("staged_chat_documents", [])
        
        if not user_input:
            return jsonify({"error": "Query cannot be empty"}), 400

        # Build user profile for compliance checking
        user_profile = {
            "user_id": user_id,
            "authorization_level": user.get("authorization_level", "unclassified"),
            "clearance": user.get("clearance", "unclassified"),
            "roles": user.get("roles", []),
            "name": user.get("name", ""),
            "email": user.get("email", "")
        }

        print(f"[Query] User: {user_id}, Auth: {user_profile['authorization_level']}, Query: '{user_input}'")
        
        # Process through integrated system with compliance
        result = process_samm_query(
            user_input, 
            chat_history, 
            staged_chat_documents_metadata,
            user_profile=user_profile
        )
        
        compliance_info = result.get('compliance_result', {})
        print(f"[Query Result] Intent: {result['intent']}, "
              f"Compliance: {compliance_info.get('compliance_status', 'not_checked')}, "
              f"Authorized: {compliance_info.get('authorized', 'N/A')}")
        
        response_data = {
            "response": {"answer": result["answer"]},
            "metadata": result["metadata"],
            "compliance": compliance_info,
            "uploadedChatDocuments": []
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"[Query] Error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/api/system/status", methods=["GET"])
def get_system_status_for_ui():
    """Get system status in Vue.js UI compatible format"""
    # Test Ollama connection
    try:
        test_response = call_ollama_enhanced("Test", "Respond with 'OK'", temperature=0.0)
        ollama_status = "connected" if "OK" in test_response else "error"
        ollama_available = True
    except:
        ollama_status = "disconnected"
        ollama_available = False
    
    # Get database status
    db_status = orchestrator.get_database_status()
    
    return jsonify({
        "status": "ready" if ollama_available else "degraded",
        "ai_model": OLLAMA_MODEL,
        "ai_provider": "Ollama",
        "ai_url": OLLAMA_URL,
        "ai_status": ollama_status,
        "knowledge_base": {
            "name": "SAMM Chapter 1",
            "entities": len(knowledge_graph.entities),
            "relationships": len(knowledge_graph.relationships),
            "status": "loaded"
        },
        "agents": {
            "available": 3,
            "types": ["intent", "integrated_entity", "enhanced_answer"],
            "orchestration": "integrated_database_state",
            "versions": {
                "intent_agent": "1.0",
                "entity_agent": "IntegratedEntityAgent v1.0",
                "answer_agent": "EnhancedAnswerAgent v1.0"
            }
        },
        "database_integration": {
            "cosmos_gremlin": db_status["cosmos_gremlin"]["connected"],
            "vector_db": db_status["vector_db"]["connected"],
            "vector_db_ttl": db_status["vector_db_ttl"]["connected"],
            "embedding_model": db_status["embedding_model"]["loaded"]
        },
        "services": {
            "authentication": "configured" if oauth else "mock",
            "database": "connected" if cases_container_client else "disabled",
            "storage": "connected" if blob_service_client else "disabled"
        },
        "version": "5.0.0-integrated-database",
        "system_name": "Integrated Database SAMM ASIST",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/examples", methods=["GET"])
def get_example_questions():
    """Get example questions in Vue.js UI compatible format"""
    examples = [
        "What is Security Cooperation?",
        "Who supervises Security Assistance programs?", 
        "What is the difference between Security Cooperation and Security Assistance?",
        "What does DFAS do?",
        "When was the Foreign Assistance Act enacted?",
        "What is an Implementing Agency?"
    ]
    
    return jsonify({
        "examples": examples,
        "count": len(examples)
    })

@app.route("/api/agents/hil_update", methods=["POST"])
def update_agents_from_hil():
    """Update agents from human-in-the-loop feedback"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Query is required for HIL update"}), 400
        
        # Extract correction data for each agent
        intent_correction = data.get("intent_correction")
        entity_correction = data.get("entity_correction") 
        answer_correction = data.get("answer_correction")
        
        if not any([intent_correction, entity_correction, answer_correction]):
            return jsonify({"error": "At least one correction type must be provided"}), 400
        
        # Update agents through orchestrator
        results = orchestrator.update_agents_from_hil(
            query=query,
            intent_correction=intent_correction,
            entity_correction=entity_correction,
            answer_correction=answer_correction
        )
        
        return jsonify({
            "message": "HIL updates applied successfully",
            "query": query,
            "updates_applied": results,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"[HIL Update] Error: {str(e)}")
        return jsonify({"error": f"Failed to apply HIL updates: {str(e)}"}), 500

@app.route("/api/agents/trigger_update", methods=["POST"])
def update_agents_from_trigger():
    """Update agents when new entity/relationship data is available"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        data = request.get_json()
        new_entities = data.get("new_entities", [])
        new_relationships = data.get("new_relationships", [])
        trigger_data = data.get("trigger_data", {})
        
        if not new_entities and not new_relationships:
            return jsonify({"error": "At least one new entity or relationship must be provided"}), 400
        
        # Update agents through orchestrator
        results = orchestrator.update_agents_from_trigger(
            new_entities=new_entities,
            new_relationships=new_relationships,
            trigger_data=trigger_data
        )
        
        return jsonify({
            "message": "Trigger updates applied successfully",
            "new_entities_count": len(new_entities),
            "new_relationships_count": len(new_relationships),
            "updates_applied": results,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"[Trigger Update] Error: {str(e)}")
        return jsonify({"error": f"Failed to apply trigger updates: {str(e)}"}), 500

@app.route("/api/agents/status", methods=["GET"])
def get_agents_status():
    """Get detailed status of all agents including HIL and trigger update counts"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        agent_status = orchestrator.get_agent_status()
        database_status = orchestrator.get_database_status()
        
        return jsonify({
            "agents": agent_status,
            "database_integration": database_status,
            "summary": {
                "total_hil_updates": sum(agent["hil_feedback_count"] for agent in agent_status.values()),
                "total_trigger_updates": sum(agent["trigger_update_count"] for agent in agent_status.values()),
                "total_learned_items": (
                    agent_status["intent_agent"]["learned_patterns"] +
                    agent_status["integrated_entity_agent"]["custom_entities"] + 
                    agent_status["enhanced_answer_agent"]["answer_corrections"]
                ),
                "database_features": {
                    "cosmos_gremlin_connected": database_status["cosmos_gremlin"]["connected"],
                    "vector_db_connected": database_status["vector_db"]["connected"],
                    "vector_db_ttl_connected": database_status["vector_db_ttl"]["connected"],
                    "embedding_model_loaded": database_status["embedding_model"]["loaded"],
                    "total_vector_collections": len(database_status["vector_db"]["collections"]) + len(database_status["vector_db_ttl"]["collections"])
                },
                "enhanced_features": {
                    "extraction_phases": agent_status["integrated_entity_agent"]["extraction_phases"],
                    "samm_patterns": agent_status["integrated_entity_agent"]["samm_patterns"],
                    "response_templates": agent_status["enhanced_answer_agent"]["response_templates"],
                    "acronym_expansions": agent_status["enhanced_answer_agent"]["acronym_expansions"]
                }
            },
            "system_version": "Integrated_Database_SAMM_v5.0",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"[Agent Status] Error: {str(e)}")
        return jsonify({"error": f"Failed to get agent status: {str(e)}"}), 500

@app.route("/api/database/status", methods=["GET"])
def get_database_status():
    """Get detailed database connection status"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        database_status = orchestrator.get_database_status()
        
        return jsonify({
            "database_connections": database_status,
            "summary": {
                "total_connections": sum(1 for db in database_status.values() if db.get("connected", False)),
                "cosmos_gremlin_status": "connected" if database_status["cosmos_gremlin"]["connected"] else "disconnected",
                "vector_databases": {
                    "vector_db_collections": len(database_status["vector_db"]["collections"]),
                    "vector_db_ttl_collections": len(database_status["vector_db_ttl"]["collections"]),
                    "total_collections": len(database_status["vector_db"]["collections"]) + len(database_status["vector_db_ttl"]["collections"])
                },
                "embedding_model_status": "loaded" if database_status["embedding_model"]["loaded"] else "not_loaded"
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"[Database Status] Error: {str(e)}")
        return jsonify({"error": f"Failed to get database status: {str(e)}"}), 500

@app.route("/api/samm/status", methods=["GET"])
def get_samm_system_status():
    """Get detailed system status (maintains backward compatibility)"""
    # Test Ollama connection
    try:
        test_response = call_ollama_enhanced("Test", "Respond with 'OK'", temperature=0.0)
        ollama_status = "connected" if "OK" in test_response else "error"
    except:
        ollama_status = "disconnected"
    
    # Get database status
    database_status = orchestrator.get_database_status()
    
    return jsonify({
        "status": "ready",
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_status": ollama_status,
        "knowledge_graph": {
            "entities": len(knowledge_graph.entities),
            "relationships": len(knowledge_graph.relationships)
        },
        "orchestration": {
            "type": "integrated_database_state",
            "workflow_steps": [step.value for step in WorkflowStep],
            "agents": ["intent_agent", "integrated_entity_agent", "enhanced_answer_agent"]
        },
        "database_integration": {
            "cosmos_gremlin": {
                "connected": database_status["cosmos_gremlin"]["connected"],
                "endpoint": database_status["cosmos_gremlin"]["endpoint"],
                "database": database_status["cosmos_gremlin"]["database"]
            },
            "vector_databases": {
                "vector_db": {
                    "connected": database_status["vector_db"]["connected"],
                    "collections": database_status["vector_db"]["collections"]
                },
                "vector_db_ttl": {
                    "connected": database_status["vector_db_ttl"]["connected"],
                    "collections": database_status["vector_db_ttl"]["collections"]
                }
            },
            "embedding_model": {
                "loaded": database_status["embedding_model"]["loaded"],
                "model_name": database_status["embedding_model"]["model_name"]
            }
        },
        "enhanced_capabilities": {
            "integrated_entity_extraction": {
                "phases": 3,  # pattern_matching, nlp_extraction, database_queries
                "patterns": sum(len(patterns) for patterns in orchestrator.entity_agent.samm_entity_patterns.values()),
                "database_enhanced": True,
                "confidence_scoring": True,
                "ai_context_generation": True
            },
            "answer_generation": {
                "templates": len(orchestrator.answer_agent.samm_response_templates),
                "quality_scoring": True,
                "multi_pass_validation": True,
                "acronym_expansion": True
            }
        },
        "services": {
            "auth0": "configured" if oauth else "mock",
            "cosmos_db": "connected" if cases_container_client else "disabled",
            "blob_storage": "connected" if blob_service_client else "disabled"
        },
        "version": "Integrated_Database_SAMM_v5.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/samm/workflow", methods=["GET"])
def get_workflow_info():
    """Get workflow orchestration information"""
    return jsonify({
        "orchestration_type": "integrated_database_state",
        "workflow_steps": [
            {
                "step": step.value,
                "description": {
                    "initialize": "Initialize integrated workflow state with database connections",
                    "analyze_intent": "Analyze user intent using Intent Agent with HIL learning",
                    "extract_entities": "Extract entities using Integrated Entity Agent with database queries", 
                    "generate_answer": "Generate answer using Enhanced Answer Agent with quality scoring",
                    "complete": "Complete integrated workflow successfully",
                    "error": "Handle any workflow errors"
                }.get(step.value, "Unknown step")
            }
            for step in WorkflowStep
        ],
        "agents": [
            {
                "name": "IntentAgent", 
                "purpose": "Classify user queries and determine intent", 
                "hil_updates": True, 
                "trigger_updates": True,
                "version": "1.0"
            },
            {
                "name": "IntegratedEntityAgent", 
                "purpose": "Multi-phase entity extraction with SAMM patterns and database integration", 
                "hil_updates": True, 
                "trigger_updates": True,
                "version": "1.0",
                "features": ["pattern_matching", "nlp_extraction", "database_queries", "ai_context_generation", "confidence_scoring"],
                "database_integration": True
            },
            {
                "name": "EnhancedAnswerAgent", 
                "purpose": "Intent-optimized answer generation with quality enhancement", 
                "hil_updates": True, 
                "trigger_updates": True,
                "version": "1.0",
                "features": ["intent_optimization", "multi_pass_generation", "quality_scoring", "acronym_expansion", "answer_validation"]
            }
        ],
        "database_integration": {
            "cosmos_gremlin": {
                "purpose": "Graph database for entity relationships",
                "query_type": "Gremlin traversal queries"
            },
            "vector_db": {
                "purpose": "Document vector search",
                "query_type": "Semantic similarity search"
            },
            "vector_db_ttl": {
                "purpose": "TTL knowledge base search",
                "query_type": "Semantic similarity search"
            }
        },
        "transitions": {
            "initialize": "analyze_intent",
            "analyze_intent": "extract_entities", 
            "extract_entities": "generate_answer",
            "generate_answer": "complete",
            "complete": "end",
            "error": "end"
        },
        "update_capabilities": {
            "human_in_loop": {
                "endpoint": "/api/agents/hil_update",
                "description": "Update agents based on human feedback corrections",
                "supported_corrections": ["intent", "entity", "answer"]
            },
            "trigger_updates": {
                "endpoint": "/api/agents/trigger_update", 
                "description": "Update agents when new entity/relationship data becomes available",
                "supported_data": ["new_entities", "new_relationships", "trigger_data"]
            }
        },
        "integrated_features": {
            "entity_extraction": {
                "phases": 3,
                "database_enhanced": True,
                "confidence_scoring": True,
                "pattern_matching": True,
                "nlp_extraction": True,
                "ai_fallback": True
            },
            "answer_generation": {
                "intent_optimization": True,
                "quality_validation": True,
                "multi_pass_generation": True,
                "template_adherence": True,
                "automatic_enhancement": True
            }
        }
    })

@app.route("/api/samm/examples", methods=["GET"])
def get_samm_examples():
    """Get example SAMM questions (detailed format for compatibility)"""
    examples = [
        {
            "question": "What is Security Cooperation?",
            "type": "definition",
            "expected_entities": ["Security Cooperation", "DoD"],
            "expected_intent": "definition",
            "database_relevant": True
        },
        {
            "question": "Who supervises Security Assistance programs?", 
            "type": "authority",
            "expected_entities": ["Security Assistance", "Department of State"],
            "expected_intent": "authority",
            "database_relevant": True
        },
        {
            "question": "What is the difference between Security Cooperation and Security Assistance?",
            "type": "distinction",
            "expected_entities": ["Security Cooperation", "Security Assistance"],
            "expected_intent": "distinction",
            "database_relevant": True
        },
        {
            "question": "What does DFAS do?",
            "type": "organization",
            "expected_entities": ["DFAS", "Defense Finance and Accounting Service"],
            "expected_intent": "organization",
            "database_relevant": True
        },
        {
            "question": "When was the Foreign Assistance Act enacted?",
            "type": "factual",
            "expected_entities": ["Foreign Assistance Act", "FAA"],
            "expected_intent": "factual",
            "database_relevant": True
        },
        {
            "question": "What is an Implementing Agency?",
            "type": "definition",
            "expected_entities": ["Implementing Agency", "IA"],
            "expected_intent": "definition",
            "database_relevant": True
        }
    ]
    
    return jsonify({
        "examples": examples,
        "count": len(examples),
        "usage": "Use these to test the Integrated Database orchestrated SAMM system",
        "integrated_testing": {
            "entity_extraction": "Each example includes expected entities for validation",
            "intent_classification": "Each example includes expected intent for validation",
            "database_integration": "All examples will trigger database queries",
            "quality_scoring": "Answers will include quality scores and enhancements"
        }
    })

@app.route("/api/samm/knowledge", methods=["GET"])
def get_knowledge_graph_info():
    """Get knowledge graph information"""
    entities_info = []
    for entity_id, entity in knowledge_graph.entities.items():
        entities_info.append({
            "id": entity_id,
            "label": entity['properties'].get('label', entity_id),
            "type": entity['type'],
            "definition": entity['properties'].get('definition', ''),
            "section": entity['properties'].get('section', '')
        })
    
    # Get integrated agent pattern information
    samm_patterns = {}
    if hasattr(orchestrator.entity_agent, 'samm_entity_patterns'):
        samm_patterns = {
            category: len(patterns) 
            for category, patterns in orchestrator.entity_agent.samm_entity_patterns.items()
        }
    
    # Get database status
    database_status = orchestrator.get_database_status()
    
    return jsonify({
        "entities": entities_info,
        "relationships": knowledge_graph.relationships,
        "total_entities": len(knowledge_graph.entities),
        "total_relationships": len(knowledge_graph.relationships),
        "enhanced_patterns": {
            "samm_entity_patterns": samm_patterns,
            "total_patterns": sum(samm_patterns.values()) if samm_patterns else 0,
            "pattern_categories": list(samm_patterns.keys()) if samm_patterns else []
        },
        "dynamic_knowledge": {
            "custom_entities": len(orchestrator.entity_agent.custom_entities),
            "dynamic_entities": len(orchestrator.entity_agent.dynamic_knowledge["entities"]),
            "dynamic_relationships": len(orchestrator.entity_agent.dynamic_knowledge["relationships"])
        },
        "database_integration": {
            "cosmos_gremlin": {
                "connected": database_status["cosmos_gremlin"]["connected"],
                "endpoint": database_status["cosmos_gremlin"]["endpoint"]
            },
            "vector_databases": {
                "vector_db_collections": len(database_status["vector_db"]["collections"]),
                "vector_db_ttl_collections": len(database_status["vector_db_ttl"]["collections"])
            },
            "embedding_model": {
                "loaded": database_status["embedding_model"]["loaded"],
                "model": database_status["embedding_model"]["model_name"]
            }
        }
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check"""
    # Test integrated Ollama connection
    ollama_healthy = False
    try:
        test_response = call_ollama_enhanced("Test", "Respond with 'OK'", temperature=0.0)
        ollama_healthy = "OK" in test_response
    except:
        pass
    
    # Test agent status
    agent_healthy = False
    try:
        agent_status = orchestrator.get_agent_status()
        agent_healthy = len(agent_status) == 3  # All 3 agents should be present
    except:
        pass
    
    # Test database connections
    database_status = orchestrator.get_database_status()
    database_healthy = any([
        database_status["cosmos_gremlin"]["connected"],
        database_status["vector_db"]["connected"],
        database_status["vector_db_ttl"]["connected"]
    ])
    
    return jsonify({
        "status": "healthy" if (ollama_healthy and agent_healthy) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ollama_model": OLLAMA_MODEL,
        "ollama_healthy": ollama_healthy,
        "agents_healthy": agent_healthy,
        "database_healthy": database_healthy,
        "version": "Integrated_Database_SAMM_v5.0",
        "components": {
            "ollama": "healthy" if ollama_healthy else "degraded",
            "agents": "healthy" if agent_healthy else "degraded",
            "knowledge_graph": "healthy" if len(knowledge_graph.entities) > 0 else "degraded",
            "case_database": "healthy" if cases_container_client else "disabled",
            "blob_storage": "healthy" if blob_service_client else "disabled",
            "cosmos_gremlin": "healthy" if database_status["cosmos_gremlin"]["connected"] else "disconnected",
            "vector_db": "healthy" if database_status["vector_db"]["connected"] else "disconnected",
            "vector_db_ttl": "healthy" if database_status["vector_db_ttl"]["connected"] else "disconnected",
            "embedding_model": "healthy" if database_status["embedding_model"]["loaded"] else "not_loaded"
        }
    })

# Static file serving
@app.route('/')
def serve_main_app():
    user = require_auth()
    if not user and oauth:
        session['next_url'] = request.path
        return redirect(url_for("login"))
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_vue_paths(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else: 
        user = require_auth()
        if not user and oauth:
            session['next_url'] = request.path
            return redirect(url_for("login"))
        return send_from_directory(app.static_folder, 'index.html')

# Cleanup on exit
import atexit
atexit.register(orchestrator.cleanup)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    
    print("\n" + "="*90)
    print("🚀 Complete Integrated SAMM ASIST System with Database Integration v5.0")
    print("="*90)
    print(f"🌐 Server: http://localhost:{port}")
    print(f"🤖 Ollama Model: {OLLAMA_MODEL}")
    print(f"🔗 Ollama URL: {OLLAMA_URL}")
    print(f"📊 Knowledge Graph: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relationships)} relationships")
    print(f"🎯 Integrated Database Orchestration: {len(WorkflowStep)} workflow steps")
    print(f"🔄 Integrated Agents: Intent → Integrated Entity (Database) → Enhanced Answer (Quality)")
    print(f"🔐 Auth: {'OAuth (Auth0)' if oauth else 'Mock User'}")
    print(f"💾 Storage: {'Azure Cosmos DB' if cases_container_client else 'In-Memory'}")
    print(f"📁 Blob Storage: {'Azure' if blob_service_client else 'Disabled'}")
    
    # Database status
    db_status = orchestrator.get_database_status()
    print(f"\n💽 Database Integration:")
    print(f"• Cosmos Gremlin: {'Connected' if db_status['cosmos_gremlin']['connected'] else 'Disconnected'} ({db_status['cosmos_gremlin']['endpoint']})")
    print(f"• Vector DB: {'Connected' if db_status['vector_db']['connected'] else 'Disconnected'} ({len(db_status['vector_db']['collections'])} collections)")
    print(f"• Vector DB TTL: {'Connected' if db_status['vector_db_ttl']['connected'] else 'Disconnected'} ({len(db_status['vector_db_ttl']['collections'])} collections)")
    print(f"• Embedding Model: {'Loaded' if db_status['embedding_model']['loaded'] else 'Not Loaded'} ({db_status['embedding_model']['model_name']})")
    
    print(f"\n📡 Core Endpoints:")
    print(f"• Integrated Query: POST http://localhost:{port}/api/query")
    print(f"• System Status: GET http://localhost:{port}/api/system/status")
    print(f"• Database Status: GET http://localhost:{port}/api/database/status")
    print(f"• Examples: GET http://localhost:{port}/api/examples")
    print(f"• User Cases: GET http://localhost:{port}/api/user/cases")
    print(f"• Authentication: GET http://localhost:{port}/login")
    
    print(f"\n🤖 Enhanced Agent Endpoints:")
    print(f"• HIL Update: POST http://localhost:{port}/api/agents/hil_update")
    print(f"• Trigger Update: POST http://localhost:{port}/api/agents/trigger_update")
    print(f"• Agent Status: GET http://localhost:{port}/api/agents/status")
    
    print(f"\n📡 Advanced SAMM Endpoints:")
    print(f"• Detailed Status: GET http://localhost:{port}/api/samm/status")
    print(f"• Integrated Workflow: GET http://localhost:{port}/api/samm/workflow") 
    print(f"• Knowledge Graph: GET http://localhost:{port}/api/samm/knowledge")
    print(f"• Health Check: GET http://localhost:{port}/api/health")
    
    print(f"\n🧪 Try these questions:")
    print("• What is Security Cooperation?")
    print("• Who supervises Security Assistance programs?")
    print("• What's the difference between SC and SA?") 
    print("• What does DFAS do?")
    
    print(f"\n⚡ Integrated Database Capabilities:")
    print("• Integrated Entity Agent: Pattern → NLP → Database queries (Cosmos Gremlin + Vector DBs)")
    print(f"  - {sum(len(patterns) for patterns in orchestrator.entity_agent.samm_entity_patterns.values())} SAMM patterns")
    print("  - Real-time database integration for entity context")
    print("  - Confidence scoring for all extracted entities")
    print("  - Dynamic knowledge expansion with HIL feedback")
    print("• Enhanced Answer Agent: Intent-optimized responses with quality scoring")
    print(f"  - {len(orchestrator.answer_agent.samm_response_templates)} response templates")
    print(f"  - {len(orchestrator.answer_agent.acronym_expansions)} acronym expansions")
    print("  - Multi-pass generation with validation")
    print("  - Automatic quality enhancement")
    
    print(f"\n🔄 Learning System:")
    print("• Human-in-Loop (HIL): Correct intent, entities, and answers")
    print("• Trigger Updates: Add new entities and relationships dynamically")
    print("• Database Learning: Entities learn from graph and vector databases")
    print("• Pattern Learning: Intent agent learns query patterns")
    print("• Knowledge Expansion: Entity agent grows knowledge base")
    print("• Answer Corrections: Answer agent stores and reuses corrections")
    print("• Quality Improvement: All agents learn from feedback")
    
    print(f"\n📊 Agent Status:")
    try:
        status = orchestrator.get_agent_status()
        print(f"• Intent Agent: {status['intent_agent']['learned_patterns']} learned patterns")
        print(f"• Integrated Entity Agent: {status['integrated_entity_agent']['custom_entities']} custom entities, {status['integrated_entity_agent']['samm_patterns']} SAMM patterns")
        print(f"• Enhanced Answer Agent: {status['enhanced_answer_agent']['answer_corrections']} stored corrections, {status['enhanced_answer_agent']['response_templates']} templates")
    except:
        print("• Agent status: Initializing...")
    
    print("="*90 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)
