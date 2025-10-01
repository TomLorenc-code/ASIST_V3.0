# itar_compliance_microservice.py - ITAR and Security Compliance Microservice
import os
import json
import asyncio
import sys
import requests
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import logging

# Fix for Windows asyncio issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Flask & Extensions
from flask import Flask, request, jsonify
from flask_cors import CORS

# Environment
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Database imports
try:
    from gremlin_python.driver import client, serializer
    from gremlin_python.driver.protocol import GremlinServerError
    print("Gremlin client imported successfully")
except ImportError:
    print("Gremlin client not available - compliance features limited")
    client = None

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    print("Vector database libraries imported successfully")
except ImportError:
    print("Vector database libraries not available - compliance features limited")
    chromadb = None
    SentenceTransformer = None

# =============================================================================
# CONFIGURATION
# =============================================================================

# Microservice Configuration
COMPLIANCE_PORT = int(os.getenv("COMPLIANCE_PORT", 3002))
MAIN_APP_URL = os.getenv("MAIN_APP_URL", "http://localhost:3000")

# Database Configuration
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

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# COMPLIANCE ENUMS AND TYPES
# =============================================================================

class AuthorizationLevel(Enum):
    """Authorization levels for SC/A operations"""
    UNCLASSIFIED = "unclassified"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"
    SCI = "sci"  # Sensitive Compartmented Information

class ComplianceStatus(Enum):
    """Compliance check results"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    REQUIRES_REVIEW = "requires_review"
    INSUFFICIENT_DATA = "insufficient_data"

class PolicyDomain(Enum):
    """Policy domains for compliance checking"""
    ITAR = "itar"  # International Traffic in Arms Regulations
    AECA = "aeca"  # Arms Export Control Act
    SAMM = "samm"  # Security Assistance Management Manual
    EAR = "ear"    # Export Administration Regulations
    OFAC = "ofac"  # Office of Foreign Assets Control
    NDAA = "ndaa"  # National Defense Authorization Act

class ITARCategory(Enum):
    """ITAR United States Munitions List (USML) Categories"""
    CAT_I = "I"        # Firearms, Close Assault Weapons and Combat Shotguns
    CAT_II = "II"      # Guns and Armament
    CAT_III = "III"    # Ammunition/Ordnance
    CAT_IV = "IV"      # Launch Vehicles, Guided Missiles, Ballistic Missiles
    CAT_V = "V"        # Explosives and Energetic Materials
    CAT_VI = "VI"      # Surface Vessels of War and Special Naval Equipment
    CAT_VII = "VII"    # Ground Vehicles
    CAT_VIII = "VIII"  # Aircraft and Associated Equipment
    CAT_IX = "IX"      # Military Training Equipment
    CAT_X = "X"        # Personal Protective Equipment
    CAT_XI = "XI"      # Military Electronics
    CAT_XII = "XII"    # Fire Control, Laser, Imaging and Guidance Equipment
    CAT_XIII = "XIII"  # Materials and Miscellaneous Articles
    CAT_XIV = "XIV"    # Toxicological Agents
    CAT_XV = "XV"      # Spacecraft and Related Articles
    CAT_XVI = "XVI"    # Nuclear Weapons Related Articles
    CAT_XVII = "XVII"  # Classified Articles
    CAT_XVIII = "XVIII" # Directed Energy Weapons
    CAT_XIX = "XIX"    # Gas Turbine Engines and Associated Equipment
    CAT_XX = "XX"      # Submersible Vessels and Related Articles
    CAT_XXI = "XXI"    # Articles, Services and Related Technical Data

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__)
CORS(app)

# =============================================================================
# OLLAMA CALL FUNCTION
# =============================================================================

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
                "num_predict": 2048
            }
        }
        
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=data, timeout=90)
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API error: {e}")
        return f"Error calling Ollama API: {str(e)}"
    except Exception as e:
        logger.error(f"Ollama processing error: {e}")
        return f"Error processing with Ollama: {str(e)}"

# =============================================================================
# ITAR SECURITY COMPLIANCE AGENT
# =============================================================================

class ITARSecurityComplianceAgent:
    """
    ITAR and Security Compliance Agent for SC/A authorization and compliance verification
    
    This agent:
    1. Verifies user authorization levels against query content
    2. Checks ITAR/AECA/SAMM policy compliance for SC/A operations
    3. Identifies potential violations or required reviews
    4. Provides compliance guidance and recommendations
    5. Integrates with main app's answer generation
    """
    
    def __init__(self):
        self.cosmos_gremlin_client = None
        self.vector_db_client = None
        self.vector_db_ttl_client = None
        self.embedding_model = None
        
        # ITAR-specific knowledge base
        self.itar_usml_categories = self._initialize_itar_categories()
        self.policy_frameworks = self._initialize_policy_frameworks()
        self.authorization_matrix = self._initialize_authorization_matrix()
        self.compliance_rules = self._initialize_compliance_rules()
        self.country_classifications = self._initialize_country_classifications()
        
        # Initialize database connections
        self._initialize_connections()
        
        logger.info("ITAR Security Compliance Agent initialized")
    
    def _initialize_connections(self):
        """Initialize database connections"""
        try:
            # Initialize Cosmos DB Gremlin
            if client and COSMOS_GREMLIN_CONFIG['password']:
                username = f"/dbs/{COSMOS_GREMLIN_CONFIG['database']}/colls/{COSMOS_GREMLIN_CONFIG['graph']}"
                endpoint_url = f"wss://{COSMOS_GREMLIN_CONFIG['endpoint']}:443/gremlin"
                
                self.cosmos_gremlin_client = client.Client(
                    url=endpoint_url,
                    traversal_source="g",
                    username=username,
                    password=COSMOS_GREMLIN_CONFIG['password'],
                    message_serializer=serializer.GraphSONSerializersV2d0()
                )
                logger.info("Cosmos Gremlin connection established")
            
            # Initialize Vector DBs
            if chromadb:
                if Path(VECTOR_DB_PATH).exists():
                    self.vector_db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
                    logger.info("Vector DB connected")
                
                if Path(VECTOR_DB_TTL_PATH).exists():
                    self.vector_db_ttl_client = chromadb.PersistentClient(path=VECTOR_DB_TTL_PATH)
                    logger.info("Vector DB TTL connected")
            
            # Initialize embedding model
            if SentenceTransformer:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info("Embedding model loaded")
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _initialize_itar_categories(self) -> Dict[str, Dict]:
        """Initialize ITAR USML categories with detailed information"""
        return {
            ITARCategory.CAT_I.value: {
                "name": "Firearms, Close Assault Weapons and Combat Shotguns",
                "description": "Military firearms and related equipment",
                "risk_level": "high",
                "common_items": ["military rifles", "machine guns", "combat shotguns"],
                "license_required": True
            },
            ITARCategory.CAT_VIII.value: {
                "name": "Aircraft and Associated Equipment",
                "description": "Military aircraft, helicopters, and related systems",
                "risk_level": "very_high",
                "common_items": ["fighter aircraft", "military helicopters", "UAVs"],
                "license_required": True
            },
            ITARCategory.CAT_XI.value: {
                "name": "Military Electronics",
                "description": "Electronic systems and equipment for military use",
                "risk_level": "high",
                "common_items": ["radar systems", "military communications", "electronic warfare"],
                "license_required": True
            },
            ITARCategory.CAT_XIII.value: {
                "name": "Materials and Miscellaneous Articles",
                "description": "Special materials and miscellaneous defense articles",
                "risk_level": "medium",
                "common_items": ["armor materials", "special alloys", "protective equipment"],
                "license_required": True
            },
            ITARCategory.CAT_XXI.value: {
                "name": "Articles, Services and Related Technical Data",
                "description": "Defense services and technical data not elsewhere specified",
                "risk_level": "variable",
                "common_items": ["technical assistance", "training", "maintenance"],
                "license_required": True
            }
        }
    
    def _initialize_policy_frameworks(self) -> Dict[str, Dict]:
        """Initialize policy framework definitions"""
        return {
            PolicyDomain.ITAR.value: {
                "name": "International Traffic in Arms Regulations",
                "authority": "Department of State, Directorate of Defense Trade Controls (DDTC)",
                "classification_levels": [AuthorizationLevel.UNCLASSIFIED, AuthorizationLevel.CONFIDENTIAL, 
                                        AuthorizationLevel.SECRET, AuthorizationLevel.TOP_SECRET],
                "key_sections": ["120.1", "120.3", "121.1", "126.1", "127.1"],
                "controlled_items": ["defense_articles", "defense_services", "technical_data"],
                "prohibited_countries": ["Country Group D:1", "Country Group D:3", "Country Group D:4", "Country Group D:5"],
                "license_types": ["DSP-5", "DSP-73", "DSP-83", "TAA", "MLA"],
                "congressional_notification": True,
                "end_use_monitoring": True
            },
            PolicyDomain.AECA.value: {
                "name": "Arms Export Control Act",
                "authority": "Department of State",
                "classification_levels": [AuthorizationLevel.UNCLASSIFIED, AuthorizationLevel.CONFIDENTIAL],
                "key_sections": ["Section 3", "Section 38", "Section 40A", "Section 36"],
                "programs": ["FMS", "FMF", "IMET", "DCS"],
                "congressional_notification": True,
                "threshold_amounts": {"major_defense_equipment": 14000000, "defense_articles_services": 50000000}
            },
            PolicyDomain.SAMM.value: {
                "name": "Security Assistance Management Manual",
                "authority": "DSCA",
                "classification_levels": [AuthorizationLevel.UNCLASSIFIED],
                "chapters": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
                "case_types": ["FMS", "FMF", "IMET", "Building_Partner_Capacity"],
                "processes": ["case_development", "congressional_notification", "implementation"]
            },
            PolicyDomain.EAR.value: {
                "name": "Export Administration Regulations",
                "authority": "Department of Commerce, Bureau of Industry and Security (BIS)",
                "classification_levels": [AuthorizationLevel.UNCLASSIFIED],
                "controlled_items": ["dual_use_items", "commercial_items"],
                "license_types": ["individual", "validated_end_user", "special_comprehensive"]
            }
        }
    
    def _initialize_authorization_matrix(self) -> Dict[str, Dict]:
        """Initialize authorization level requirements matrix"""
        return {
            "query_analysis": {
                AuthorizationLevel.UNCLASSIFIED.value: {
                    "allowed_domains": [PolicyDomain.SAMM.value, PolicyDomain.AECA.value],
                    "itar_categories": [],  # No ITAR access at unclassified
                    "restricted_terms": ["classified", "proprietary", "sensitive"],
                    "max_detail_level": "general"
                },
                AuthorizationLevel.CONFIDENTIAL.value: {
                    "allowed_domains": [PolicyDomain.SAMM.value, PolicyDomain.AECA.value, PolicyDomain.ITAR.value],
                    "itar_categories": [ITARCategory.CAT_XIII.value, ITARCategory.CAT_XXI.value],
                    "restricted_terms": ["secret", "top_secret", "sci"],
                    "max_detail_level": "detailed"
                },
                AuthorizationLevel.SECRET.value: {
                    "allowed_domains": list(PolicyDomain),
                    "itar_categories": [cat.value for cat in ITARCategory],
                    "restricted_terms": ["top_secret", "sci"],
                    "max_detail_level": "comprehensive"
                },
                AuthorizationLevel.TOP_SECRET.value: {
                    "allowed_domains": list(PolicyDomain),
                    "itar_categories": [cat.value for cat in ITARCategory],
                    "restricted_terms": ["sci"],
                    "max_detail_level": "full"
                }
            },
            "case_access": {
                AuthorizationLevel.UNCLASSIFIED.value: {
                    "case_types": ["FMS_unclassified", "IMET_unclassified"],
                    "country_restrictions": ["Country Group D:1", "Country Group D:3", "Country Group D:4", "Country Group D:5"],
                    "value_limits": {"case_value": 10000000, "individual_item": 1000000}
                },
                AuthorizationLevel.CONFIDENTIAL.value: {
                    "case_types": ["FMS_all", "FMF_standard", "IMET_all", "DCS_commercial"],
                    "country_restrictions": ["Country Group D:1", "Country Group D:3"],
                    "value_limits": {"case_value": 100000000, "individual_item": 10000000}
                },
                AuthorizationLevel.SECRET.value: {
                    "case_types": ["all"],
                    "country_restrictions": [],
                    "value_limits": {"case_value": -1, "individual_item": -1}  # No limits
                }
            }
        }
    
    def _initialize_compliance_rules(self) -> List[Dict]:
        """Initialize compliance checking rules"""
        return [
            {
                "rule_id": "ITAR_001",
                "domain": PolicyDomain.ITAR.value,
                "description": "ITAR-controlled items require appropriate export authorization",
                "trigger_terms": ["defense_article", "technical_data", "defense_service", "usml"],
                "required_authorization": AuthorizationLevel.CONFIDENTIAL.value,
                "compliance_check": "verify_itar_authorization",
                "severity": "high"
            },
            {
                "rule_id": "ITAR_002",
                "domain": PolicyDomain.ITAR.value,
                "description": "USML Category I-XX items require specific licensing",
                "trigger_terms": ["category i", "category ii", "category iii", "category iv", "category v",
                                "category vi", "category vii", "category viii", "category ix", "category x",
                                "category xi", "category xii", "category xiii", "category xiv", "category xv",
                                "category xvi", "category xvii", "category xviii", "category xix", "category xx"],
                "required_authorization": AuthorizationLevel.SECRET.value,
                "compliance_check": "verify_usml_category_authorization",
                "severity": "very_high"
            },
            {
                "rule_id": "AECA_001", 
                "domain": PolicyDomain.AECA.value,
                "description": "AECA programs require congressional notification thresholds",
                "trigger_terms": ["FMS", "major_defense_equipment", "congressional notification"],
                "required_authorization": AuthorizationLevel.UNCLASSIFIED.value,
                "compliance_check": "verify_congressional_notification",
                "severity": "medium"
            },
            {
                "rule_id": "COUNTRY_001",
                "domain": "country_policy",
                "description": "Prohibited countries require special authorization",
                "trigger_terms": ["Country Group D:1", "embargoed_country", "arms_embargo"],
                "required_authorization": AuthorizationLevel.SECRET.value,
                "compliance_check": "verify_country_authorization",
                "severity": "very_high"
            },
            {
                "rule_id": "CLASSIFICATION_001",
                "domain": "classification",
                "description": "Classified information requires appropriate clearance",
                "trigger_terms": ["classified", "secret", "confidential", "top_secret", "sci"],
                "required_authorization": AuthorizationLevel.CONFIDENTIAL.value,
                "compliance_check": "verify_classification_authorization",
                "severity": "high"
            },
            {
                "rule_id": "TECHNICAL_DATA_001",
                "domain": PolicyDomain.ITAR.value,
                "description": "Technical data transfers require ITAR compliance review",
                "trigger_terms": ["technical_data", "blueprints", "specifications", "software", "technology_transfer"],
                "required_authorization": AuthorizationLevel.CONFIDENTIAL.value,
                "compliance_check": "verify_technical_data_transfer",
                "severity": "high"
            }
        ]
    
    def _initialize_country_classifications(self) -> Dict[str, Dict]:
        """Initialize country classifications for export control"""
        return {
            "Country Group D:1": {
                "description": "Countries subject to arms embargo",
                "countries": ["Specified in CFR Title 22"],
                "restrictions": "Complete arms embargo",
                "license_policy": "Denial",
                "risk_level": "maximum"
            },
            "Country Group D:3": {
                "description": "Countries of concern for missile technology",
                "restrictions": "Missile technology restrictions",
                "license_policy": "Case-by-case review",
                "risk_level": "high"
            },
            "Country Group D:4": {
                "description": "Countries subject to certain restrictions",
                "restrictions": "Specific item restrictions",
                "license_policy": "Enhanced review",
                "risk_level": "medium"
            },
            "NATO_Allies": {
                "description": "NATO member countries",
                "restrictions": "Reduced restrictions",
                "license_policy": "Generally favorable",
                "risk_level": "low"
            },
            "Major_Non_NATO_Allies": {
                "description": "Major Non-NATO Allies (MNNA)",
                "restrictions": "Reduced restrictions for certain items",
                "license_policy": "Generally favorable",
                "risk_level": "low"
            }
        }
    
    async def verify_compliance(self, query: str, intent_info: Dict, entity_info: Dict, 
                               user_profile: Dict = None) -> Dict[str, Any]:
        """
        Main compliance verification method
        
        Args:
            query: User's query text
            intent_info: Intent analysis from main app
            entity_info: Entity extraction from main app
            user_profile: User's authorization profile
            
        Returns:
            Comprehensive compliance verification result
        """
        try:
            logger.info(f"Verifying ITAR/Security compliance for query: {query[:100]}...")
            
            # Extract user authorization level
            user_auth_level = self._extract_user_auth_level(user_profile or {})
            
            # Analyze query for ITAR and compliance requirements
            compliance_analysis = await self._analyze_compliance_requirements(query, intent_info, entity_info)
            
            # Check authorization against requirements
            auth_result = self._check_authorization_compliance(user_auth_level, compliance_analysis)
            
            # Check policy compliance
            policy_compliance = await self._check_policy_compliance(query, compliance_analysis)
            
            # Check ITAR-specific compliance
            itar_compliance = await self._check_itar_compliance(query, compliance_analysis)
            
            # Generate compliance recommendations
            recommendations = self._generate_compliance_recommendations(
                auth_result, policy_compliance, itar_compliance, user_auth_level
            )
            
            # Generate compliance-aware content guidance
            content_guidance = self._generate_content_guidance(compliance_analysis, auth_result)
            
            return {
                "compliance_status": self._determine_overall_compliance_status(
                    policy_compliance, itar_compliance
                ),
                "authorized": auth_result["authorized"],
                "user_authorization_level": user_auth_level.value,
                "required_authorization_level": auth_result["required_level"],
                "policy_compliance": policy_compliance,
                "itar_compliance": itar_compliance,
                "compliance_analysis": compliance_analysis,
                "recommendations": recommendations,
                "content_guidance": content_guidance,
                "restrictions": self._get_content_restrictions(auth_result, compliance_analysis),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_version": "ITARSecurityComplianceAgent_v1.0"
            }
            
        except Exception as e:
            logger.error(f"Compliance verification error: {e}")
            return {
                "compliance_status": ComplianceStatus.INSUFFICIENT_DATA.value,
                "authorized": False,
                "error": str(e),
                "recommendations": ["Contact system administrator for compliance verification"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _extract_user_auth_level(self, user_profile: Dict) -> AuthorizationLevel:
        """Extract user's authorization level from profile"""
        auth_level_str = user_profile.get("authorization_level", "unclassified").lower()
        
        try:
            return AuthorizationLevel(auth_level_str)
        except ValueError:
            logger.warning(f"Unknown authorization level: {auth_level_str}, defaulting to UNCLASSIFIED")
            return AuthorizationLevel.UNCLASSIFIED
    
    async def _analyze_compliance_requirements(self, query: str, intent_info: Dict, entity_info: Dict) -> Dict[str, Any]:
        """Analyze query for compliance requirements using AI and knowledge bases"""
        analysis = {
            "query": query,
            "intent": intent_info.get("intent", "unknown"),
            "entities": entity_info.get("entities", []),
            "detected_domains": [],
            "itar_categories": [],
            "sensitive_terms": [],
            "country_mentions": [],
            "required_auth_level": AuthorizationLevel.UNCLASSIFIED,
            "risk_indicators": [],
            "technical_data_indicators": [],
            "export_control_indicators": []
        }
        
        try:
            query_lower = query.lower()
            
            # Check for policy domain indicators
            for domain in PolicyDomain:
                domain_indicators = self._get_domain_indicators(domain)
                if any(indicator in query_lower for indicator in domain_indicators):
                    analysis["detected_domains"].append(domain.value)
            
            # Check for ITAR USML categories
            analysis["itar_categories"] = self._detect_itar_categories(query)
            
            # Check for sensitive terms
            analysis["sensitive_terms"] = self._detect_sensitive_terms(query)
            
            # Check for country mentions
            analysis["country_mentions"] = self._detect_country_mentions(query)
            
            # Check for technical data indicators
            analysis["technical_data_indicators"] = self._detect_technical_data_indicators(query)
            
            # Check for export control indicators
            analysis["export_control_indicators"] = self._detect_export_control_indicators(query)
            
            # Determine required authorization level
            analysis["required_auth_level"] = self._determine_required_auth_level(analysis)
            
            # Identify risk indicators
            analysis["risk_indicators"] = self._identify_compliance_risk_indicators(query, analysis)
            
            # Use AI for enhanced compliance analysis
            ai_analysis = await self._ai_enhanced_compliance_analysis(query, analysis)
            analysis["ai_insights"] = ai_analysis
            
            logger.info(f"Compliance analysis complete: {len(analysis['detected_domains'])} domains, "
                       f"{len(analysis['itar_categories'])} ITAR categories, "
                       f"{len(analysis['risk_indicators'])} risk indicators")
            
        except Exception as e:
            logger.error(f"Compliance analysis error: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _get_domain_indicators(self, domain: PolicyDomain) -> List[str]:
        """Get indicator terms for a policy domain"""
        indicators = {
            PolicyDomain.ITAR: ["itar", "defense article", "technical data", "export license", "usml", 
                               "munitions list", "ddtc", "defense trade controls"],
            PolicyDomain.AECA: ["aeca", "fms", "foreign military sales", "arms export", "congressional notification"],
            PolicyDomain.SAMM: ["samm", "security assistance", "security cooperation", "dsca"],
            PolicyDomain.EAR: ["ear", "dual use", "commerce control list", "ccl", "bis"],
            PolicyDomain.OFAC: ["ofac", "sanctions", "embargo", "sdn list"],
            PolicyDomain.NDAA: ["ndaa", "national defense authorization", "section 1226"]
        }
        return indicators.get(domain, [])
    
    def _detect_itar_categories(self, query: str) -> List[str]:
        """Detect ITAR USML categories mentioned in query"""
        categories = []
        query_lower = query.lower()
        
        # Check for explicit category mentions
        for category in ITARCategory:
            if f"category {category.value.lower()}" in query_lower:
                categories.append(category.value)
        
        # Check for category-specific items
        category_items = {
            ITARCategory.CAT_I.value: ["firearms", "machine guns", "rifles", "pistols"],
            ITARCategory.CAT_VIII.value: ["aircraft", "helicopters", "fighter", "bomber", "uav", "drone"],
            ITARCategory.CAT_XI.value: ["radar", "electronics", "communications", "jamming"],
            ITARCategory.CAT_XIII.value: ["armor", "materials", "alloys", "composites"],
            ITARCategory.CAT_XXI.value: ["services", "training", "maintenance", "technical assistance"]
        }
        
        for category, items in category_items.items():
            if any(item in query_lower for item in items):
                if category not in categories:
                    categories.append(category)
        
        return categories
    
    def _detect_sensitive_terms(self, query: str) -> List[str]:
        """Detect sensitive terms in query"""
        sensitive_terms = []
        query_lower = query.lower()
        
        # Classification indicators
        classification_terms = ["classified", "confidential", "secret", "top secret", "sci", "noforn", "proprietary"]
        sensitive_terms.extend([term for term in classification_terms if term in query_lower])
        
        # ITAR-controlled terms
        itar_terms = ["defense article", "technical data", "defense service", "significant military equipment", 
                     "munitions", "weapons", "military technology"]
        sensitive_terms.extend([term for term in itar_terms if term in query_lower])
        
        # Export control terms
        export_terms = ["export", "re-export", "transfer", "foreign national", "third country"]
        sensitive_terms.extend([term for term in export_terms if term in query_lower])
        
        return sensitive_terms
    
    def _detect_country_mentions(self, query: str) -> List[str]:
        """Detect country mentions and classify them"""
        countries = []
        query_lower = query.lower()
        
        # Check for country group mentions
        country_groups = ["country group d:1", "country group d:3", "country group d:4", "country group d:5"]
        countries.extend([group for group in country_groups if group in query_lower])
        
        # Check for embargo indicators
        embargo_terms = ["embargoed", "sanctioned", "restricted country", "arms embargo"]
        countries.extend([term for term in embargo_terms if term in query_lower])
        
        return countries
    
    def _detect_technical_data_indicators(self, query: str) -> List[str]:
        """Detect technical data transfer indicators"""
        indicators = []
        query_lower = query.lower()
        
        technical_terms = ["blueprints", "specifications", "software", "source code", "algorithms", 
                          "design data", "manufacturing data", "test data", "technology transfer"]
        indicators.extend([term for term in technical_terms if term in query_lower])
        
        return indicators
    
    def _detect_export_control_indicators(self, query: str) -> List[str]:
        """Detect export control indicators"""
        indicators = []
        query_lower = query.lower()
        
        export_control_terms = ["export license", "license exception", "deemed export", "technology transfer",
                               "foreign person", "end user", "end use", "diversion"]
        indicators.extend([term for term in export_control_terms if term in query_lower])
        
        return indicators
    
    def _determine_required_auth_level(self, analysis: Dict) -> AuthorizationLevel:
        """Determine required authorization level based on analysis"""
        required_level = AuthorizationLevel.UNCLASSIFIED
        
        # Check for classification indicators
        if any(term in ["classified", "confidential"] for term in analysis["sensitive_terms"]):
            required_level = AuthorizationLevel.CONFIDENTIAL
        elif any(term in ["secret", "top secret"] for term in analysis["sensitive_terms"]):
            required_level = AuthorizationLevel.SECRET
        elif "sci" in analysis["sensitive_terms"]:
            required_level = AuthorizationLevel.SCI
        
        # Check for ITAR requirements
        if PolicyDomain.ITAR.value in analysis["detected_domains"]:
            if analysis["itar_categories"]:
                # High-risk ITAR categories require higher authorization
                high_risk_categories = [ITARCategory.CAT_I.value, ITARCategory.CAT_VIII.value, 
                                      ITARCategory.CAT_XI.value, ITARCategory.CAT_XVI.value]
                if any(cat in high_risk_categories for cat in analysis["itar_categories"]):
                    required_level = max(required_level, AuthorizationLevel.SECRET, key=lambda x: list(AuthorizationLevel).index(x))
                else:
                    required_level = max(required_level, AuthorizationLevel.CONFIDENTIAL, key=lambda x: list(AuthorizationLevel).index(x))
        
        # Check for country restrictions
        if analysis["country_mentions"]:
            if any("d:1" in country or "embargoed" in country for country in analysis["country_mentions"]):
                required_level = max(required_level, AuthorizationLevel.SECRET, key=lambda x: list(AuthorizationLevel).index(x))
        
        return required_level
    
    def _identify_compliance_risk_indicators(self, query: str, analysis: Dict) -> List[str]:
        """Identify potential compliance risk indicators"""
        risk_indicators = []
        
        # High authorization requirement risk
        if analysis["required_auth_level"] != AuthorizationLevel.UNCLASSIFIED:
            risk_indicators.append(f"Requires {analysis['required_auth_level'].value} authorization")
        
        # Multiple policy domain risk
        if len(analysis["detected_domains"]) > 2:
            risk_indicators.append("Multiple policy domains involved")
        
        # ITAR-specific risks
        if PolicyDomain.ITAR.value in analysis["detected_domains"]:
            risk_indicators.append("ITAR-controlled content detected")
            
            if analysis["itar_categories"]:
                risk_indicators.append(f"USML categories detected: {', '.join(analysis['itar_categories'])}")
        
        # Technical data transfer risk
        if analysis["technical_data_indicators"]:
            risk_indicators.append("Technical data transfer indicators detected")
        
        # Export control risk
        if analysis["export_control_indicators"]:
            risk_indicators.append("Export control indicators detected")
        
        # Country risk
        if analysis["country_mentions"]:
            risk_indicators.append("Country restrictions may apply")
        
        # Sensitive terms risk
        if len(analysis["sensitive_terms"]) > 3:
            risk_indicators.append("Multiple sensitive terms detected")
        
        return risk_indicators
    
    async def _ai_enhanced_compliance_analysis(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Use AI to enhance compliance analysis"""
        system_msg = """You are an ITAR and export control compliance expert. Analyze the query for potential compliance issues.

FOCUS AREAS:
- ITAR (International Traffic in Arms Regulations) compliance
- Export control restrictions
- Technical data transfer concerns
- Country-specific restrictions
- Classification requirements

RESPONSE FORMAT (JSON):
{
    "compliance_concerns": ["concern1", "concern2"],
    "recommended_review_level": "standard|enhanced|legal_review",
    "potential_violations": ["violation1", "violation2"],
    "mitigation_suggestions": ["suggestion1", "suggestion2"]
}"""
        
        prompt = f"""Query: "{query}"

Detected domains: {analysis.get('detected_domains', [])}
ITAR categories: {analysis.get('itar_categories', [])}
Sensitive terms: {analysis.get('sensitive_terms', [])}
Country mentions: {analysis.get('country_mentions', [])}

Provide compliance analysis:"""
        
        try:
            response = call_ollama_enhanced(prompt, system_msg, temperature=0.1)
            
            # Try to parse JSON response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_part = response[json_start:json_end]
                return json.loads(json_part)
        except Exception as e:
            logger.error(f"AI compliance analysis error: {e}")
        
        # Fallback analysis
        return {
            "compliance_concerns": ["Standard compliance review recommended"],
            "recommended_review_level": "standard",
            "potential_violations": [],
            "mitigation_suggestions": ["Consult compliance officer for guidance"]
        }
    
    def _check_authorization_compliance(self, user_auth_level: AuthorizationLevel, 
                                      compliance_analysis: Dict) -> Dict[str, Any]:
        """Check if user authorization meets compliance requirements"""
        required_level = compliance_analysis["required_auth_level"]
        
        # Define authorization hierarchy
        auth_hierarchy = {
            AuthorizationLevel.UNCLASSIFIED: 0,
            AuthorizationLevel.CONFIDENTIAL: 1,
            AuthorizationLevel.SECRET: 2,
            AuthorizationLevel.TOP_SECRET: 3,
            AuthorizationLevel.SCI: 4
        }
        
        user_level_value = auth_hierarchy.get(user_auth_level, 0)
        required_level_value = auth_hierarchy.get(required_level, 0)
        
        authorized = user_level_value >= required_level_value
        
        return {
            "authorized": authorized,
            "user_level": user_auth_level.value,
            "required_level": required_level.value,
            "level_sufficient": authorized,
            "authorization_gap": max(0, required_level_value - user_level_value),
            "access_restrictions": self._get_access_restrictions(user_auth_level, compliance_analysis)
        }
    
    def _get_access_restrictions(self, user_auth_level: AuthorizationLevel, 
                                compliance_analysis: Dict) -> List[str]:
        """Get access restrictions based on authorization level"""
        restrictions = []
        
        if user_auth_level in self.authorization_matrix["query_analysis"]:
            matrix = self.authorization_matrix["query_analysis"][user_auth_level.value]
            
            # Domain restrictions
            all_domains = [domain.value for domain in PolicyDomain]
            restricted_domains = [domain for domain in all_domains 
                                if domain not in matrix["allowed_domains"]]
            if restricted_domains and any(domain in compliance_analysis["detected_domains"] 
                                       for domain in restricted_domains):
                restrictions.append(f"Restricted domains: {', '.join(restricted_domains)}")
            
            # ITAR category restrictions
            if "itar_categories" in matrix:
                restricted_categories = [cat for cat in compliance_analysis["itar_categories"]
                                       if cat not in matrix["itar_categories"]]
                if restricted_categories:
                    restrictions.append(f"Restricted ITAR categories: {', '.join(restricted_categories)}")
        
        return restrictions
    
    async def _check_policy_compliance(self, query: str, compliance_analysis: Dict) -> Dict[str, Any]:
        """Check policy compliance against defined rules"""
        compliance_result = {
            "status": ComplianceStatus.COMPLIANT.value,
            "violations": [],
            "warnings": [],
            "applicable_rules": []
        }
        
        try:
            query_lower = query.lower()
            
            # Check each compliance rule
            for rule in self.compliance_rules:
                rule_triggered = any(term in query_lower for term in rule["trigger_terms"])
                
                if rule_triggered:
                    compliance_result["applicable_rules"].append(rule["rule_id"])
                    
                    # Check if authorization meets rule requirements
                    required_auth = AuthorizationLevel(rule["required_authorization"])
                    user_auth = compliance_analysis.get("required_auth_level", AuthorizationLevel.UNCLASSIFIED)
                    
                    auth_hierarchy = {
                        AuthorizationLevel.UNCLASSIFIED: 0,
                        AuthorizationLevel.CONFIDENTIAL: 1,
                        AuthorizationLevel.SECRET: 2,
                        AuthorizationLevel.TOP_SECRET: 3,
                        AuthorizationLevel.SCI: 4
                    }
                    
                    if auth_hierarchy.get(user_auth, 0) < auth_hierarchy.get(required_auth, 0):
                        violation = {
                            "rule_id": rule["rule_id"],
                            "description": rule["description"],
                            "required_authorization": required_auth.value,
                            "violation_type": "insufficient_authorization",
                            "severity": rule.get("severity", "medium")
                        }
                        compliance_result["violations"].append(violation)
                        
                        if rule.get("severity") in ["high", "very_high"]:
                            compliance_result["status"] = ComplianceStatus.NON_COMPLIANT.value
                        elif compliance_result["status"] == ComplianceStatus.COMPLIANT.value:
                            compliance_result["status"] = ComplianceStatus.WARNING.value
            
            logger.info(f"Policy compliance check: {compliance_result['status']}, "
                       f"{len(compliance_result['violations'])} violations")
            
        except Exception as e:
            logger.error(f"Policy compliance check error: {e}")
            compliance_result["status"] = ComplianceStatus.INSUFFICIENT_DATA.value
            compliance_result["error"] = str(e)
        
        return compliance_result
    
    async def _check_itar_compliance(self, query: str, compliance_analysis: Dict) -> Dict[str, Any]:
        """Check ITAR-specific compliance"""
        itar_result = {
            "status": ComplianceStatus.COMPLIANT.value,
            "usml_categories_detected": compliance_analysis.get("itar_categories", []),
            "license_requirements": [],
            "restrictions": [],
            "end_use_monitoring": False,
            "congressional_notification": False
        }
        
        try:
            # Check USML categories
            for category in compliance_analysis.get("itar_categories", []):
                if category in self.itar_usml_categories:
                    cat_info = self.itar_usml_categories[category]
                    
                    if cat_info["license_required"]:
                        itar_result["license_requirements"].append({
                            "category": category,
                            "name": cat_info["name"],
                            "risk_level": cat_info["risk_level"]
                        })
                    
                    if cat_info["risk_level"] in ["high", "very_high"]:
                        itar_result["end_use_monitoring"] = True
                        
                        if cat_info["risk_level"] == "very_high":
                            itar_result["congressional_notification"] = True
            
            # Check for technical data transfers
            if compliance_analysis.get("technical_data_indicators"):
                itar_result["restrictions"].append("Technical data transfer restrictions apply")
                itar_result["status"] = ComplianceStatus.REQUIRES_REVIEW.value
            
            # Check country restrictions
            for country in compliance_analysis.get("country_mentions", []):
                if "d:1" in country or "embargoed" in country:
                    itar_result["restrictions"].append(f"Country restriction: {country}")
                    itar_result["status"] = ComplianceStatus.NON_COMPLIANT.value
            
            logger.info(f"ITAR compliance check: {itar_result['status']}, "
                       f"{len(itar_result['license_requirements'])} license requirements")
            
        except Exception as e:
            logger.error(f"ITAR compliance check error: {e}")
            itar_result["status"] = ComplianceStatus.INSUFFICIENT_DATA.value
            itar_result["error"] = str(e)
        
        return itar_result
    
    def _determine_overall_compliance_status(self, policy_compliance: Dict, itar_compliance: Dict) -> str:
        """Determine overall compliance status"""
        statuses = [policy_compliance.get("status"), itar_compliance.get("status")]
        
        if ComplianceStatus.NON_COMPLIANT.value in statuses:
            return ComplianceStatus.NON_COMPLIANT.value
        elif ComplianceStatus.REQUIRES_REVIEW.value in statuses:
            return ComplianceStatus.REQUIRES_REVIEW.value
        elif ComplianceStatus.WARNING.value in statuses:
            return ComplianceStatus.WARNING.value
        elif ComplianceStatus.INSUFFICIENT_DATA.value in statuses:
            return ComplianceStatus.INSUFFICIENT_DATA.value
        else:
            return ComplianceStatus.COMPLIANT.value
    
    def _generate_compliance_recommendations(self, auth_result: Dict, policy_compliance: Dict, 
                                           itar_compliance: Dict, user_auth_level: AuthorizationLevel) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Authorization recommendations
        if not auth_result["authorized"]:
            recommendations.append(
                f"Insufficient authorization: {auth_result['required_level']} clearance required"
            )
            recommendations.append("Contact security officer for authorization upgrade")
        
        # Policy violation recommendations
        for violation in policy_compliance.get("violations", []):
            if violation["violation_type"] == "insufficient_authorization":
                recommendations.append(
                    f"Policy compliance issue: {violation['description']} "
                    f"(requires {violation['required_authorization']} authorization)"
                )
        
        # ITAR-specific recommendations
        if itar_compliance.get("license_requirements"):
            recommendations.append("ITAR export license may be required")
            recommendations.append("Consult with DDTC or export control office")
        
        if itar_compliance.get("end_use_monitoring"):
            recommendations.append("End-use monitoring requirements may apply")
        
        if itar_compliance.get("congressional_notification"):
            recommendations.append("Congressional notification may be required")
        
        # General recommendations
        if not recommendations:
            recommendations.append("No compliance issues detected at current authorization level")
        
        return recommendations
    
    def _generate_content_guidance(self, compliance_analysis: Dict, auth_result: Dict) -> Dict[str, Any]:
        """Generate guidance for content generation"""
        guidance = {
            "allowed_detail_level": "general",
            "content_restrictions": [],
            "required_disclaimers": [],
            "sanitization_required": False
        }
        
        try:
            user_auth_level = AuthorizationLevel(auth_result["user_level"])
            
            if user_auth_level in self.authorization_matrix["query_analysis"]:
                matrix = self.authorization_matrix["query_analysis"][user_auth_level.value]
                guidance["allowed_detail_level"] = matrix["max_detail_level"]
            
            # Add content restrictions based on detected issues
            if compliance_analysis.get("itar_categories"):
                guidance["content_restrictions"].append("ITAR-controlled information must be sanitized")
                guidance["sanitization_required"] = True
            
            if compliance_analysis.get("sensitive_terms"):
                guidance["content_restrictions"].append("Classified information must be removed")
                guidance["sanitization_required"] = True
            
            if compliance_analysis.get("technical_data_indicators"):
                guidance["content_restrictions"].append("Technical data transfer restrictions apply")
            
            # Add required disclaimers
            if PolicyDomain.ITAR.value in compliance_analysis.get("detected_domains", []):
                guidance["required_disclaimers"].append(
                    "This information may be subject to ITAR export control restrictions"
                )
            
            if compliance_analysis.get("country_mentions"):
                guidance["required_disclaimers"].append(
                    "Country-specific export restrictions may apply"
                )
        
        except Exception as e:
            logger.error(f"Content guidance generation error: {e}")
        
        return guidance
    
    def _get_content_restrictions(self, auth_result: Dict, compliance_analysis: Dict) -> List[str]:
        """Get content restrictions for response generation"""
        restrictions = []
        
        if not auth_result["authorized"]:
            restrictions.append("Content must be limited to unclassified, general information")
        
        if compliance_analysis.get("itar_categories"):
            restrictions.append("ITAR-controlled technical details must be omitted")
        
        if compliance_analysis.get("technical_data_indicators"):
            restrictions.append("Specific technical data must not be provided")
        
        if compliance_analysis.get("country_mentions"):
            restrictions.append("Country-specific sensitive information must be omitted")
        
        return restrictions

# Initialize the ITAR compliance agent
itar_agent = ITARSecurityComplianceAgent()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "service": "ITAR Security Compliance Microservice",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route("/api/compliance/verify", methods=["POST"])
def verify_compliance():
    """Main compliance verification endpoint"""
    try:
        data = request.get_json()
        
        query = data.get("query", "")
        intent_info = data.get("intent_info", {})
        entity_info = data.get("entity_info", {})
        user_profile = data.get("user_profile", {})
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Run compliance verification
        result = asyncio.run(itar_agent.verify_compliance(
            query=query,
            intent_info=intent_info,
            entity_info=entity_info,
            user_profile=user_profile
        ))
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Compliance verification error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/compliance/status", methods=["GET"])
def get_compliance_status():
    """Get compliance agent status"""
    try:
        # Test database connections
        db_status = {
            "cosmos_gremlin": itar_agent.cosmos_gremlin_client is not None,
            "vector_db": itar_agent.vector_db_client is not None,
            "vector_db_ttl": itar_agent.vector_db_ttl_client is not None,
            "embedding_model": itar_agent.embedding_model is not None
        }
        
        return jsonify({
            "service": "ITAR Security Compliance Agent",
            "status": "ready",
            "database_connections": db_status,
            "policy_frameworks": len(itar_agent.policy_frameworks),
            "compliance_rules": len(itar_agent.compliance_rules),
            "itar_categories": len(itar_agent.itar_usml_categories),
            "authorization_levels": len(AuthorizationLevel),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/compliance/policies", methods=["GET"])
def get_policy_info():
    """Get policy framework information"""
    return jsonify({
        "policy_frameworks": itar_agent.policy_frameworks,
        "itar_categories": itar_agent.itar_usml_categories,
        "country_classifications": itar_agent.country_classifications,
        "compliance_rules": itar_agent.compliance_rules
    })

# =============================================================================
# INTEGRATION FUNCTION FOR MAIN APP
# =============================================================================

def call_compliance_microservice(query: str, intent_info: Dict, entity_info: Dict, 
                                user_profile: Dict = None) -> Dict[str, Any]:
    """
    Function to call this compliance microservice from the main app
    
    Usage in main app:
    from itar_compliance_microservice import call_compliance_microservice
    compliance_result = call_compliance_microservice(query, intent_info, entity_info, user_profile)
    """
    try:
        data = {
            "query": query,
            "intent_info": intent_info,
            "entity_info": entity_info,
            "user_profile": user_profile or {}
        }
        
        response = requests.post(
            f"http://localhost:{COMPLIANCE_PORT}/api/compliance/verify",
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        logger.error(f"Error calling compliance microservice: {e}")
        return {
            "compliance_status": ComplianceStatus.INSUFFICIENT_DATA.value,
            "authorized": False,
            "error": str(e),
            "recommendations": ["Compliance service unavailable"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔒 ITAR Security Compliance Microservice v1.0")
    print("="*80)
    print(f"🌐 Service URL: http://localhost:{COMPLIANCE_PORT}")
    print(f"🤖 Ollama Model: {OLLAMA_MODEL}")
    print(f"🔗 Ollama URL: {OLLAMA_URL}")
    print(f"🔐 Main App Integration: {MAIN_APP_URL}")
    
    # Database status
    print(f"\n💽 Database Connections:")
    print(f"• Cosmos Gremlin: {'Connected' if itar_agent.cosmos_gremlin_client else 'Disconnected'}")
    print(f"• Vector DB: {'Connected' if itar_agent.vector_db_client else 'Disconnected'}")
    print(f"• Vector DB TTL: {'Connected' if itar_agent.vector_db_ttl_client else 'Disconnected'}")
    print(f"• Embedding Model: {'Loaded' if itar_agent.embedding_model else 'Not Loaded'}")
    
    print(f"\n📊 Compliance Knowledge Base:")
    print(f"• Policy Frameworks: {len(itar_agent.policy_frameworks)}")
    print(f"• ITAR USML Categories: {len(itar_agent.itar_usml_categories)}")
    print(f"• Compliance Rules: {len(itar_agent.compliance_rules)}")
    print(f"• Authorization Levels: {len(AuthorizationLevel)}")
    print(f"• Country Classifications: {len(itar_agent.country_classifications)}")
    
    print(f"\n📡 API Endpoints:")
    print(f"• Compliance Verification: POST http://localhost:{COMPLIANCE_PORT}/api/compliance/verify")
    print(f"• Service Status: GET http://localhost:{COMPLIANCE_PORT}/api/compliance/status")
    print(f"• Policy Information: GET http://localhost:{COMPLIANCE_PORT}/api/compliance/policies")
    print(f"• Health Check: GET http://localhost:{COMPLIANCE_PORT}/")
    
    print(f"\n🔗 Main App Integration:")
    print("Add to main app.py:")
    print("```python")
    print("import requests")
    print("# Call compliance microservice before generating answer")
    print("compliance_result = requests.post(")
    print(f"    'http://localhost:{COMPLIANCE_PORT}/api/compliance/verify',")
    print("    json={'query': query, 'intent_info': intent_info, 'entity_info': entity_info}")
    print(").json()")
    print("```")
    
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=COMPLIANCE_PORT, debug=True)