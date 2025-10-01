# integrated_entity_agent_fixed.py
# Enhanced Entity Agent with Database Integration - Windows Compatible
# Connects to Cosmos DB, ChromaDB vector_db, and ChromaDB vector_db_ttl

import os
import json
import re
import requests
import time
import asyncio
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

# Fix for Windows asyncio issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Database imports
try:
    from gremlin_python.driver import client, serializer
    from gremlin_python.driver.protocol import GremlinServerError
    print("Gremlin client imported successfully")
except ImportError:
    print("Please install: pip install gremlinpython")

try:
    import chromadb
    print("ChromaDB imported successfully")
except ImportError:
    print("Please install: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformers imported successfully")
except ImportError:
    print("Please install: pip install sentence-transformers")

# Configuration from your agents
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Cosmos DB Configuration
COSMOS_CONFIG = {
    'endpoint': os.getenv("COSMOS_GREMLIN_ENDPOINT", "asist-graph-db.gremlin.cosmos.azure.com").replace('wss://', '').replace(':443/', ''),
    'database': os.getenv("COSMOS_GREMLIN_DATABASE", "ASIST-Agent-1DB"),
    'graph': os.getenv("COSMOS_GREMLIN_COLLECTION", "Agent1"),
    'password': os.getenv("COSMOS_GREMLIN_KEY", "ZqMUqmEndGSWdLfvUABbXLbEyNPgILiPaL2ngiYS9i8FMs14F7AabwzuPfpJIfYTXTS3LxLTlh3pACDbFsME7g==")
}

# Vector Database Paths
VECTOR_DB_PATH = "./vector_db"
VECTOR_DB_TTL_PATH = "./vector_db_ttl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
                "num_ctx": 4096,
                "num_predict": 512  # Reduced for more reliable JSON parsing
            }
        }
        
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        print(f"[EntityAgent] Ollama API error: {e}")
        return f"Error calling Ollama API: {str(e)}"
    except Exception as e:
        print(f"[EntityAgent] Processing error: {e}")
        return f"Error processing with Ollama: {str(e)}"

class DatabaseManager:
    """
    Manages connections to all three databases with improved error handling
    """
    
    def __init__(self):
        self.cosmos_client = None
        self.vector_db_client = None
        self.vector_db_ttl_client = None
        self.embedding_model = None
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize all database connections with better error handling"""
        print("[DatabaseManager] Initializing database connections...")
        
        # Initialize Cosmos DB connection
        self._init_cosmos_db()
        # Initialize ChromaDB connections
        self._init_vector_dbs()
        # Initialize embedding model
        self._init_embedding_model()
    
    def _init_cosmos_db(self):
        """Initialize Cosmos DB with proper cleanup"""
        try:
            username = f"/dbs/{COSMOS_CONFIG['database']}/colls/{COSMOS_CONFIG['graph']}"
            endpoint_url = f"wss://{COSMOS_CONFIG['endpoint']}:443/gremlin"
            
            self.cosmos_client = client.Client(
                url=endpoint_url,
                traversal_source="g",
                username=username,
                password=COSMOS_CONFIG['password'],
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            # Test connection with timeout
            result = self.cosmos_client.submit("g.V().limit(1).count()").all().result()
            print(f"[DatabaseManager] Cosmos DB connected successfully - {result[0]} vertices available")
            
        except Exception as e:
            print(f"[DatabaseManager] Cosmos DB connection failed: {e}")
            self.cosmos_client = None
    
    def _init_vector_dbs(self):
        """Initialize vector databases"""
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
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"[DatabaseManager] Embedding model loaded: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"[DatabaseManager] Embedding model failed to load: {e}")
            self.embedding_model = None
    
    def query_cosmos_graph(self, query_text: str, entities: List[str] = None) -> List[Dict]:
        """Query Cosmos DB graph database with better error handling"""
        if not self.cosmos_client:
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
                        vertex_results = self.cosmos_client.submit(vertex_query).all().result()
                        
                        for vertex in vertex_results:
                            results.append({
                                "type": "vertex",
                                "data": vertex,
                                "source": "cosmos_db",
                                "entity": entity
                            })
                        
                        # Query for relationships involving this entity (limited)
                        edge_query = f"g.V().has('name', containing('{entity_clean}')).bothE().limit(5)"
                        edge_results = self.cosmos_client.submit(edge_query).all().result()
                        
                        for edge in edge_results:
                            results.append({
                                "type": "edge", 
                                "data": edge,
                                "source": "cosmos_db",
                                "entity": entity
                            })
                            
                    except Exception as entity_error:
                        print(f"[DatabaseManager] Error querying entity '{entity}': {entity_error}")
                        continue
            else:
                # General query for high-level entities
                general_query = "g.V().limit(10)"
                general_results = self.cosmos_client.submit(general_query).all().result()
                
                for vertex in general_results:
                    results.append({
                        "type": "vertex",
                        "data": vertex,
                        "source": "cosmos_db"
                    })
            
            print(f"[DatabaseManager] Cosmos DB query returned {len(results)} results")
            
        except Exception as e:
            print(f"[DatabaseManager] Cosmos DB query error: {e}")
        
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
            if self.cosmos_client:
                self.cosmos_client.close()
                print("[DatabaseManager] Cosmos DB connection closed")
        except Exception as e:
            print(f"[DatabaseManager] Error closing Cosmos DB: {e}")
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get status of all database connections"""
        status = {
            "cosmos_db": {
                "connected": self.cosmos_client is not None,
                "endpoint": COSMOS_CONFIG['endpoint'],
                "database": COSMOS_CONFIG['database'],
                "graph": COSMOS_CONFIG['graph']
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

class IntegratedEntityAgent:
    """
    Integrated Entity Agent with improved error handling and performance
    """
    
    def __init__(self):
        print("[IntegratedEntityAgent] Initializing with database connections...")
        
        self.db_manager = DatabaseManager()
        
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
            ]
        }
        
        print("[IntegratedEntityAgent] Initialization complete")
    
    def extract_and_retrieve_from_all_sources(self, query: str, intent_info: Dict = None) -> Dict[str, Any]:
        """
        Main method with improved error handling and performance
        """
        print(f"[IntegratedEntityAgent] Processing query: '{query}'")
        
        try:
            # Phase 1: Extract entities
            entities = self._extract_entities_enhanced(query, intent_info or {})
            print(f"[IntegratedEntityAgent] Extracted entities: {entities}")
            
            # Phase 2: Query all data sources
            all_results = {
                "query": query,
                "entities": entities,
                "intent_info": intent_info,
                "timestamp": datetime.now().isoformat(),
                "data_sources": {}
            }
            
            # Query each source with error handling
            cosmos_results = self._safe_query_cosmos(query, entities)
            vector_results = self._safe_query_vector(query)
            vector_ttl_results = self._safe_query_vector_ttl(query)
            
            all_results["data_sources"] = {
                "cosmos_db": {
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
            
            # Phase 3: Aggregate results
            aggregated_data = self._aggregate_results(all_results)
            all_results["aggregated_data"] = aggregated_data
            all_results["total_results"] = len(cosmos_results) + len(vector_results) + len(vector_ttl_results)
            
            print(f"[IntegratedEntityAgent] Query complete: {all_results['total_results']} total results")
            return all_results
            
        except Exception as e:
            print(f"[IntegratedEntityAgent] Error processing query: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "total_results": 0
            }
    
    def _safe_query_cosmos(self, query: str, entities: List[str]) -> List[Dict]:
        """Safely query Cosmos DB"""
        try:
            print("[IntegratedEntityAgent] Querying Cosmos DB...")
            return self.db_manager.query_cosmos_graph(query, entities)
        except Exception as e:
            print(f"[IntegratedEntityAgent] Cosmos DB query failed: {e}")
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
        """Enhanced entity extraction with better error handling"""
        entities = []
        query_lower = query.lower()
        
        # Pattern matching (always works)
        for category, patterns in self.samm_entity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in query_lower:
                    entities.append(pattern)
        
        # NLP extraction (with fallback)
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
            import re
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
    
    def _aggregate_results(self, all_results: Dict) -> Dict[str, Any]:
        """Aggregate and score results from all data sources"""
        aggregated = {
            "high_confidence_entities": [],
            "relevant_documents": [],
            "graph_relationships": [],
            "ttl_triples": [],
            "summary": {
                "total_entities": 0,
                "total_documents": 0,
                "total_relationships": 0,
                "total_triples": 0
            }
        }
        
        # Process results with summary counts
        cosmos_results = all_results["data_sources"]["cosmos_db"]["results"]
        vector_results = all_results["data_sources"]["vector_db"]["results"]
        vector_ttl_results = all_results["data_sources"]["vector_db_ttl"]["results"]
        
        # Count and sample results
        for result in cosmos_results[:5]:  # Limit to 5 samples
            if result["type"] == "vertex":
                aggregated["high_confidence_entities"].append(result)
                aggregated["summary"]["total_entities"] += 1
            elif result["type"] == "edge":
                aggregated["graph_relationships"].append(result)
                aggregated["summary"]["total_relationships"] += 1
        
        for result in vector_results[:3]:  # Limit to 3 samples
            aggregated["relevant_documents"].append(result)
            aggregated["summary"]["total_documents"] += 1
        
        for result in vector_ttl_results[:2]:  # Limit to 2 samples
            aggregated["ttl_triples"].append(result)
            aggregated["summary"]["total_triples"] += 1
        
        return aggregated
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        db_status = self.db_manager.get_database_status()
        
        return {
            "agent_type": "IntegratedEntityAgent",
            "timestamp": datetime.now().isoformat(),
            "database_connections": db_status,
            "capabilities": {
                "entity_extraction": True,
                "graph_queries": db_status["cosmos_db"]["connected"],
                "document_search": db_status["vector_db"]["connected"],
                "ttl_search": db_status["vector_db_ttl"]["connected"],
                "nlp_processing": True
            },
            "total_entity_patterns": sum(len(patterns) for patterns in self.samm_entity_patterns.values())
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.db_manager.cleanup()
            print("[IntegratedEntityAgent] Cleanup complete")
        except Exception as e:
            print(f"[IntegratedEntityAgent] Cleanup error: {e}")

# Test function with better error handling
def test_integrated_agent():
    """Test the integrated entity agent with improved error handling"""
    print("="*60)
    print("Testing Fixed Integrated Entity Agent")
    print("="*60)
    
    agent = None
    
    try:
        # Initialize agent
        agent = IntegratedEntityAgent()
        
        # Check system status
        status = agent.get_system_status()
        print(f"\nSystem Status:")
        print(f"- Cosmos DB: {status['database_connections']['cosmos_db']['connected']}")
        print(f"- Vector DB: {status['database_connections']['vector_db']['connected']}")
        print(f"- Vector DB TTL: {status['database_connections']['vector_db_ttl']['connected']}")
        print(f"- Embedding Model: {status['database_connections']['embedding_model']['loaded']}")
        
        # Test queries
        test_queries = [
            "What does DSCA do?",
            "What is Security Cooperation?",
            "Who supervises Security Assistance programs?",
            "What's the difference between SC and SA?"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing: {query} ---")
            
            try:
                results = agent.extract_and_retrieve_from_all_sources(query)
                
                if "error" in results:
                    print(f"Error: {results['error']}")
                    continue
                
                print(f"Entities: {results['entities']}")
                print(f"Cosmos DB results: {results['data_sources']['cosmos_db']['count']}")
                print(f"Vector DB results: {results['data_sources']['vector_db']['count']}")
                print(f"Vector DB TTL results: {results['data_sources']['vector_db_ttl']['count']}")
                print(f"Total results: {results['total_results']}")
                
                # Show sample aggregated data
                if 'aggregated_data' in results:
                    summary = results['aggregated_data']['summary']
                    print(f"Summary - Entities: {summary['total_entities']}, Docs: {summary['total_documents']}, Relations: {summary['total_relationships']}, Triples: {summary['total_triples']}")
                
            except Exception as e:
                print(f"Query Error: {e}")
                
    except Exception as e:
        print(f"Initialization Error: {e}")
        
    finally:
        # Cleanup
        if agent:
            try:
                agent.cleanup()
            except:
                pass

if __name__ == "__main__":
    test_integrated_agent()