# alternative_turtle_parser.py
# Simple turtle parser using only built-in Python libraries
# No external RDF dependencies required

import os
import json
import uuid
import time
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

# Only Gremlin dependency needed
try:
    from gremlin_python.driver import client, serializer
    from gremlin_python.driver.protocol import GremlinServerError
    GREMLIN_AVAILABLE = True
except ImportError:
    print("gremlinpython not available. Install with: pip install gremlinpython")
    GREMLIN_AVAILABLE = False

# Environment setup
from dotenv import load_dotenv
load_dotenv()
# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # Updated to Llama 3.2
# Azure Cosmos DB Configuration
COSMOS_GREMLIN_ENDPOINT = os.getenv("COSMOS_GREMLIN_ENDPOINT", "wss://asist-graph-db.gremlin.cosmos.azure.com:443/")
COSMOS_GREMLIN_KEY = os.getenv("COSMOS_GREMLIN_KEY", "ZqMUqmEndGSWdLfvUABbXLbEyNPgILiPaL2ngiYS9i8FMs14F7AabwzuPfpJIfYTXTS3LxLTlh3pACDbFsME7g==")
COSMOS_GREMLIN_DATABASE = os.getenv("COSMOS_GREMLIN_DATABASE", "asist-graph-db")
COSMOS_GREMLIN_COLLECTION = os.getenv("COSMOS_GREMLIN_COLLECTION", "asist-samm-graph")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ParsedTriple:
    """Represents a parsed RDF triple"""
    subject: str
    predicate: str
    object: str
    object_type: str  # 'uri' or 'literal'
    
@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    uri: str
    label: str
    entity_type: str
    section_code: str = ""
    description: str = ""
    alt_labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def gremlin_id(self) -> str:
        """Generate Gremlin-compatible ID"""
        if "#" in self.uri:
            return self.uri.split("#")[-1]
        elif "/" in self.uri:
            return self.uri.split("/")[-1]
        else:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, self.uri))

@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship"""
    subject_uri: str
    predicate_uri: str
    object_uri: str
    predicate_label: str = ""
    relationship_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def subject_id(self) -> str:
        return self._extract_id_from_uri(self.subject_uri)
    
    @property
    def object_id(self) -> str:
        return self._extract_id_from_uri(self.object_uri)
    
    def _extract_id_from_uri(self, uri: str) -> str:
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        else:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, uri))

class SimpleTurtleParser:
    """Simple turtle parser using regex and string manipulation"""
    
    def __init__(self):
        self.prefixes: Dict[str, str] = {}
        self.triples: List[ParsedTriple] = []
        
    def parse_turtle_content(self, turtle_content: str) -> bool:
        """Parse turtle content using simple regex patterns"""
        logger.info("Parsing turtle content with simple parser")
        
        try:
            lines = turtle_content.strip().split('\n')
            current_subject = None
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse prefix declarations
                    if line.startswith('@prefix'):
                        self._parse_prefix(line)
                        continue
                    
                    # Parse triples
                    if self._is_triple_line(line):
                        triple = self._parse_triple_line(line, current_subject)
                        if triple:
                            self.triples.append(triple)
                            current_subject = triple.subject
                
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {line} - {e}")
            
            logger.info(f"Parsed {len(self.triples)} triples")
            return len(self.triples) > 0
            
        except Exception as e:
            logger.error(f"Failed to parse turtle content: {e}")
            return False
    
    def _parse_prefix(self, line: str):
        """Parse @prefix declarations"""
        # Example: @prefix : <http://example.com#> .
        match = re.match(r'@prefix\s+([^:]*):?\s+<([^>]+)>', line)
        if match:
            prefix = match.group(1).strip()
            namespace = match.group(2)
            self.prefixes[prefix] = namespace
            logger.debug(f"Found prefix: {prefix} -> {namespace}")
    
    def _is_triple_line(self, line: str) -> bool:
        """Check if line contains a triple"""
        return (':' in line and not line.startswith('@') and 
                ('rdf:type' in line or 'rdfs:label' in line or 
                 'skos:' in line or 'owl:' in line or
                 any(pred in line for pred in ['managedBy', 'oversees', 'approves', 'regulates', 'sellsTo'])))
    
    def _parse_triple_line(self, line: str, current_subject: Optional[str] = None) -> Optional[ParsedTriple]:
        """Parse a single triple line"""
        # Remove trailing semicolon or period
        line = line.rstrip(' ;.')
        
        # Split by whitespace, but be careful with quoted strings
        parts = self._smart_split(line)
        
        if len(parts) < 3:
            return None
        
        subject = parts[0] if not current_subject or parts[0] != ';' else current_subject
        predicate = parts[1] if parts[0] != ';' else parts[0]
        obj = ' '.join(parts[2:])  # Object might have spaces
        
        # Expand prefixed URIs
        subject = self._expand_uri(subject)
        predicate = self._expand_uri(predicate)
        
        # Determine object type and expand if necessary
        if obj.startswith('"') and obj.endswith('"'):
            object_type = 'literal'
            obj = obj[1:-1]  # Remove quotes
        elif obj.startswith('<') and obj.endswith('>'):
            object_type = 'uri'
            obj = obj[1:-1]  # Remove angle brackets
        else:
            object_type = 'uri'
            obj = self._expand_uri(obj)
        
        return ParsedTriple(subject, predicate, obj, object_type)
    
    def _smart_split(self, line: str) -> List[str]:
        """Split line while preserving quoted strings"""
        parts = []
        current = ""
        in_quotes = False
        in_angles = False
        
        for char in line:
            if char == '"' and not in_angles:
                in_quotes = not in_quotes
                current += char
            elif char == '<' and not in_quotes:
                in_angles = True
                current += char
            elif char == '>' and not in_quotes:
                in_angles = False
                current += char
            elif char.isspace() and not in_quotes and not in_angles:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts
    
    def _expand_uri(self, uri: str) -> str:
        """Expand prefixed URIs to full URIs"""
        if ':' in uri and not uri.startswith('http'):
            prefix, local = uri.split(':', 1)
            if prefix in self.prefixes:
                return self.prefixes[prefix] + local
        return uri
    
    def extract_entities_and_relationships(self) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Extract entities and relationships from parsed triples"""
        logger.info("Extracting entities and relationships")
        
        # Find all entities (subjects that have rdf:type owl:NamedIndividual)
        entities = {}
        relationships = []
        
        # First pass: identify entities
        for triple in self.triples:
            if (triple.predicate.endswith('type') and 
                triple.object.endswith('NamedIndividual')):
                entities[triple.subject] = ExtractedEntity(
                    uri=triple.subject,
                    label=self._extract_label_from_uri(triple.subject),
                    entity_type="Entity"
                )
        
        # Second pass: collect properties for entities
        for triple in self.triples:
            if triple.subject in entities:
                entity = entities[triple.subject]
                
                if triple.predicate.endswith('label'):
                    entity.label = triple.object
                elif triple.predicate.endswith('definition'):
                    entity.description = triple.object
                elif triple.predicate.endswith('altLabel'):
                    entity.alt_labels.append(triple.object)
                elif triple.predicate.endswith('type') and not triple.object.endswith('NamedIndividual'):
                    entity.entity_type = self._extract_label_from_uri(triple.object)
                else:
                    # Store as additional property
                    prop_name = self._extract_label_from_uri(triple.predicate)
                    entity.properties[prop_name] = triple.object
        
        # Third pass: identify relationships between entities
        for triple in self.triples:
            if (triple.subject in entities and 
                triple.object_type == 'uri' and 
                triple.object in entities and
                not triple.predicate.endswith(('type', 'label', 'definition', 'altLabel'))):
                
                relationship = ExtractedRelationship(
                    subject_uri=triple.subject,
                    predicate_uri=triple.predicate,
                    object_uri=triple.object,
                    predicate_label=self._extract_label_from_uri(triple.predicate),
                    relationship_type=self._extract_label_from_uri(triple.predicate).lower()
                )
                relationships.append(relationship)
        
        entity_list = list(entities.values())
        logger.info(f"Extracted {len(entity_list)} entities and {len(relationships)} relationships")
        
        return entity_list, relationships
    
    def _extract_label_from_uri(self, uri: str) -> str:
        """Extract a readable label from a URI"""
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.split('/')[-1]
        return uri

# Rest of the classes remain the same (GremlinVertex, GremlinEdge, etc.)
@dataclass
class GremlinVertex:
    """Gremlin-compatible vertex format"""
    id: str
    label: str
    properties: Dict[str, Any]
    
    def to_gremlin_query(self) -> str:
        """Generate Gremlin query string for vertex creation"""
        query = f"g.addV('{self.label}').property('id', '{self.id}')"
        
        for key, value in self.properties.items():
            if value is not None and str(value).strip() != "":
                if isinstance(value, bool):
                    query += f".property('{key}', {str(value).lower()})"
                elif isinstance(value, (int, float)):
                    query += f".property('{key}', {value})"
                else:
                    escaped_value = str(value).replace("'", "\\'").replace("\\", "\\\\")
                    query += f".property('{key}', '{escaped_value}')"
        
        return query

@dataclass
class GremlinEdge:
    """Gremlin-compatible edge format"""
    label: str
    from_id: str
    to_id: str
    properties: Dict[str, Any]
    
    def to_gremlin_query(self) -> str:
        """Generate Gremlin query string for edge creation"""
        query = f"g.V('{self.from_id}').addE('{self.label}').to(g.V('{self.to_id}'))"
        
        for key, value in self.properties.items():
            if value is not None and str(value).strip() != "":
                if isinstance(value, bool):
                    query += f".property('{key}', {str(value).lower()})"
                elif isinstance(value, (int, float)):
                    query += f".property('{key}', {value})"
                else:
                    escaped_value = str(value).replace("'", "\\'").replace("\\", "\\\\")
                    query += f".property('{key}', '{escaped_value}')"
        
        return query

class SimpleTurtleToGremlinAgent:
    """Simplified agent using basic turtle parsing"""
    
    def __init__(self):
        self.parser = SimpleTurtleParser()
        self.entities: List[ExtractedEntity] = []
        self.relationships: List[ExtractedRelationship] = []
        
        if GREMLIN_AVAILABLE:
            self.gremlin_client = None
    
    def process_turtle_content(self, turtle_content: str) -> Dict[str, Any]:
        """Process turtle content and load into Gremlin"""
        logger.info("Starting simple turtle processing pipeline")
        
        results = {
            "success": False,
            "start_time": datetime.now().isoformat(),
            "errors": []
        }
        
        try:
            # Parse turtle content
            if not self.parser.parse_turtle_content(turtle_content):
                results["errors"].append("Failed to parse turtle content")
                return results
            
            # Extract entities and relationships
            self.entities, self.relationships = self.parser.extract_entities_and_relationships()
            
            results["entities_found"] = len(self.entities)
            results["relationships_found"] = len(self.relationships)
            
            # Connect to Gremlin if available
            if GREMLIN_AVAILABLE:
                success = self._connect_to_gremlin()
                if success:
                    self._load_to_gremlin()
                    results["loaded_to_gremlin"] = True
                else:
                    results["errors"].append("Failed to connect to Gremlin")
            else:
                results["gremlin_unavailable"] = True
            
            results["success"] = len(self.entities) > 0
            results["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            results["errors"].append(str(e))
        
        return results
    
    def _connect_to_gremlin(self) -> bool:
        """Connect to Gremlin API"""
        try:
            self.gremlin_client = client.Client(
                COSMOS_GREMLIN_ENDPOINT,
                'g',
                username=f"/dbs/{COSMOS_GREMLIN_DATABASE}/colls/{COSMOS_GREMLIN_COLLECTION}",
                password=COSMOS_GREMLIN_KEY,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            # Test connection
            result = self.gremlin_client.submit("g.V().count()").all().result()
            logger.info("Connected to Gremlin API successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Gremlin: {e}")
            return False
    
    def _load_to_gremlin(self):
        """Load entities and relationships to Gremlin"""
        # Load entities
        for entity in self.entities:
            vertex = GremlinVertex(
                id=entity.gremlin_id,
                label=entity.entity_type.lower().replace(" ", "_"),
                properties={
                    "name": entity.label,
                    "uri": entity.uri,
                    "description": entity.description,
                    **entity.properties
                }
            )
            
            try:
                query = vertex.to_gremlin_query()
                self.gremlin_client.submit(query).all().result()
                logger.info(f"Loaded entity: {entity.label}")
            except Exception as e:
                logger.warning(f"Failed to load entity {entity.label}: {e}")
        
        # Load relationships
        for rel in self.relationships:
            edge = GremlinEdge(
                label=rel.relationship_type.replace(" ", "_"),
                from_id=rel.subject_id,
                to_id=rel.object_id,
                properties={"predicate_uri": rel.predicate_uri}
            )
            
            try:
                query = edge.to_gremlin_query()
                self.gremlin_client.submit(query).all().result()
                logger.info(f"Loaded relationship: {rel.predicate_label}")
            except Exception as e:
                logger.warning(f"Failed to load relationship {rel.predicate_label}: {e}")

def main():
    """Test the simplified parser"""
    print("Simple Turtle to Gremlin Parser (No rdflib required)")
    print("=" * 50)
    
    # Test data
    turtle_content = """
@prefix : <http://www.semanticweb.org/travis.eiswerth/ontologies/2025/7/FMS_Domain#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

:DSCA rdf:type owl:NamedIndividual ,
              :Organization ;
      rdfs:label "Defense Security Cooperation Agency" ;
      skos:definition "US DoD agency responsible for foreign military sales" .

:DoD rdf:type owl:NamedIndividual ,
             :Organization ;
     rdfs:label "Department of Defense" ;
     skos:definition "United States Department of Defense" .

:FMS rdf:type owl:NamedIndividual ,
             :Program ;
     rdfs:label "Foreign Military Sales" ;
     skos:definition "Program for selling defense articles and services to foreign governments" .

:DSCA :managedBy :DoD .
:DSCA :oversees :FMS .
"""
    
    agent = SimpleTurtleToGremlinAgent()
    results = agent.process_turtle_content(turtle_content)
    
    print(f"Results: {json.dumps(results, indent=2)}")
    
    if agent.entities:
        print("\nFound entities:")
        for entity in agent.entities:
            print(f"  - {entity.label} ({entity.entity_type})")
    
    if agent.relationships:
        print("\nFound relationships:")
        for rel in agent.relationships:
            print(f"  - {rel.predicate_label}")

if __name__ == "__main__":
    main()