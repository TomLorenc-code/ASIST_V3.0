# integration_code.py - Code to add to your main app.py for ITAR compliance integration

# =============================================================================
# ADD THESE IMPORTS TO THE TOP OF YOUR MAIN APP.PY
# =============================================================================

import requests
from typing import Dict, List, Any, Optional

# =============================================================================
# ADD THIS CONFIGURATION SECTION TO YOUR MAIN APP.PY
# =============================================================================

# ITAR Compliance Microservice Configuration
ITAR_COMPLIANCE_SERVICE_URL = os.getenv("ITAR_COMPLIANCE_SERVICE_URL", "http://localhost:3002")
ITAR_COMPLIANCE_ENABLED = os.getenv("ITAR_COMPLIANCE_ENABLED", "true").lower() == "true"

# =============================================================================
# ADD THIS ITAR COMPLIANCE CLIENT CLASS TO YOUR MAIN APP.PY
# =============================================================================

class ITARComplianceClient:
    """Client for communicating with ITAR Compliance Microservice"""
    
    def __init__(self, service_url: str = ITAR_COMPLIANCE_SERVICE_URL):
        self.service_url = service_url
        self.enabled = ITAR_COMPLIANCE_ENABLED
        
    def verify_compliance(self, query: str, intent_info: Dict, entity_info: Dict, 
                         user_profile: Dict = None, timeout: int = 30) -> Dict[str, Any]:
        """
        Verify compliance for a query through the ITAR microservice
        
        Args:
            query: User's query text
            intent_info: Intent analysis from main app
            entity_info: Entity extraction from main app
            user_profile: User's authorization profile
            timeout: Request timeout in seconds
            
        Returns:
            Compliance verification result
        """
        if not self.enabled:
            return self._get_disabled_response()
        
        try:
            data = {
                "query": query,
                "intent_info": intent_info,
                "entity_info": entity_info,
                "user_profile": user_profile or {}
            }
            
            response = requests.post(
                f"{self.service_url}/api/compliance/verify",
                json=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"ITAR compliance service returned status {response.status_code}")
                return self._get_error_response(f"Service returned status {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("ITAR compliance service timeout")
            return self._get_error_response("Compliance service timeout")
        except requests.exceptions.ConnectionError:
            logger.error("ITAR compliance service connection error")
            return self._get_error_response("Compliance service unavailable")
        except Exception as e:
            logger.error(f"ITAR compliance service error: {e}")
            return self._get_error_response(str(e))
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get ITAR compliance service status"""
        if not self.enabled:
            return {"status": "disabled", "enabled": False}
        
        try:
            response = requests.get(f"{self.service_url}/api/compliance/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}
    
    def _get_disabled_response(self) -> Dict[str, Any]:
        """Return response when service is disabled"""
        return {
            "compliance_status": "not_checked",
            "authorized": True,  # Default to authorized when disabled
            "user_authorization_level": "unclassified",
            "required_authorization_level": "unclassified",
            "policy_compliance": {"status": "not_checked"},
            "itar_compliance": {"status": "not_checked"},
            "recommendations": ["ITAR compliance checking is disabled"],
            "content_guidance": {
                "allowed_detail_level": "general",
                "content_restrictions": [],
                "required_disclaimers": [],
                "sanitization_required": False
            },
            "restrictions": [],
            "service_enabled": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response when service fails"""
        return {
            "compliance_status": "service_error",
            "authorized": True,  # Default to authorized on service error
            "user_authorization_level": "unclassified",
            "required_authorization_level": "unclassified",
            "policy_compliance": {"status": "service_error"},
            "itar_compliance": {"status": "service_error"},
            "recommendations": [f"Compliance service error: {error_msg}"],
            "content_guidance": {
                "allowed_detail_level": "general",
                "content_restrictions": ["Service unavailable - exercise caution"],
                "required_disclaimers": ["Compliance verification unavailable"],
                "sanitization_required": True
            },
            "restrictions": ["Compliance verification unavailable"],
            "service_error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# =============================================================================
# ADD THIS TO YOUR MAIN APP.PY AFTER OTHER INITIALIZATIONS
# =============================================================================

# Initialize ITAR Compliance Client
itar_compliance_client = ITARComplianceClient()
print(f"ITAR Compliance: {'Enabled' if ITAR_COMPLIANCE_ENABLED else 'Disabled'} ({ITAR_COMPLIANCE_SERVICE_URL})")

# =============================================================================
# MODIFY YOUR EnhancedAnswerAgent CLASS
# =============================================================================

class EnhancedAnswerAgentWithCompliance(EnhancedAnswerAgent):
    """
    Enhanced Answer Agent with ITAR Compliance Integration
    
    This extends your existing EnhancedAnswerAgent to include compliance checking
    """
    
    def __init__(self):
        super().__init__()
        self.compliance_client = itar_compliance_client
        print("[EnhancedAnswerAgentWithCompliance] Initialized with ITAR compliance integration")
    
    def generate_answer(self, query: str, intent_info: Dict, entity_info: Dict, 
                       chat_history: List = None, documents_context: List = None,
                       user_profile: Dict = None) -> Dict[str, Any]:
        """
        Enhanced answer generation with ITAR compliance verification
        
        Returns:
            Dict containing answer and compliance information
        """
        try:
            print(f"[AnswerAgent] Generating compliance-aware answer for query: '{query[:50]}...'")
            
            # Step 1: Verify compliance first
            compliance_result = self.compliance_client.verify_compliance(
                query=query,
                intent_info=intent_info,
                entity_info=entity_info,
                user_profile=user_profile
            )
            
            print(f"[AnswerAgent] Compliance status: {compliance_result.get('compliance_status', 'unknown')}")
            print(f"[AnswerAgent] Authorized: {compliance_result.get('authorized', False)}")
            
            # Step 2: Check if user is authorized to receive full answer
            if not compliance_result.get("authorized", False):
                return self._generate_restricted_answer(query, compliance_result)
            
            # Step 3: Generate answer with compliance guidance
            content_guidance = compliance_result.get("content_guidance", {})
            answer = self._generate_compliance_aware_answer(
                query, intent_info, entity_info, chat_history, 
                documents_context, content_guidance
            )
            
            # Step 4: Apply content restrictions and sanitization
            final_answer = self._apply_content_restrictions(answer, compliance_result)
            
            # Step 5: Add required disclaimers
            final_answer = self._add_compliance_disclaimers(final_answer, compliance_result)
            
            return {
                "answer": final_answer,
                "compliance_result": compliance_result,
                "content_restricted": compliance_result.get("content_guidance", {}).get("sanitization_required", False),
                "restrictions_applied": len(compliance_result.get("restrictions", [])) > 0
            }
            
        except Exception as e:
            logger.error(f"Compliance-aware answer generation error: {e}")
            # Fallback to original method on error
            original_answer = super().generate_answer(query, intent_info, entity_info, chat_history, documents_context)
            return {
                "answer": original_answer,
                "compliance_result": {"status": "error", "error": str(e)},
                "content_restricted": False,
                "restrictions_applied": False
            }
    
    def _generate_restricted_answer(self, query: str, compliance_result: Dict) -> Dict[str, Any]:
        """Generate restricted answer for unauthorized users"""
        restricted_answer = f"""I'm unable to provide a detailed answer to your query about "{query}" due to authorization restrictions.

{' '.join(compliance_result.get('recommendations', []))}

For general information about Security Cooperation and Security Assistance, please consult publicly available SAMM documentation or contact your security officer for guidance."""
        
        return {
            "answer": restricted_answer,
            "compliance_result": compliance_result,
            "content_restricted": True,
            "restrictions_applied": True
        }
    
    def _generate_compliance_aware_answer(self, query: str, intent_info: Dict, entity_info: Dict,
                                        chat_history: List, documents_context: List,
                                        content_guidance: Dict) -> str:
        """Generate answer with compliance awareness"""
        
        # Determine allowed detail level
        detail_level = content_guidance.get("allowed_detail_level", "general")
        
        # Modify system message based on compliance guidance
        compliance_aware_system_msg = self._create_compliance_aware_system_message(
            intent_info.get("intent", "general"), content_guidance
        )
        
        # Generate answer using parent class method but with compliance-aware system message
        answer = super()._generate_with_validation(
            self._create_enhanced_prompt(query, intent_info, entity_info),
            compliance_aware_system_msg,
            intent_info
        )
        
        return answer
    
    def _create_compliance_aware_system_message(self, intent: str, content_guidance: Dict) -> str:
        """Create system message with compliance awareness"""
        base_msg = super()._create_optimized_system_message(intent, "")
        
        # Add compliance restrictions to system message
        compliance_instructions = "\n\nCOMPLIANCE REQUIREMENTS:"
        
        detail_level = content_guidance.get("allowed_detail_level", "general")
        compliance_instructions += f"\n- Response detail level: {detail_level}"
        
        if content_guidance.get("content_restrictions"):
            compliance_instructions += "\n- Content restrictions:"
            for restriction in content_guidance["content_restrictions"]:
                compliance_instructions += f"\n  • {restriction}"
        
        if content_guidance.get("sanitization_required"):
            compliance_instructions += "\n- CRITICAL: Remove any ITAR-controlled technical details"
            compliance_instructions += "\n- CRITICAL: Remove any classified information"
            compliance_instructions += "\n- Provide only general, unclassified information"
        
        return base_msg + compliance_instructions
    
    def _apply_content_restrictions(self, answer: str, compliance_result: Dict) -> str:
        """Apply content restrictions to the generated answer"""
        restricted_answer = answer
        
        # Apply sanitization if required
        if compliance_result.get("content_guidance", {}).get("sanitization_required"):
            restricted_answer = self._sanitize_content(restricted_answer, compliance_result)
        
        # Apply specific restrictions
        restrictions = compliance_result.get("restrictions", [])
        if restrictions:
            print(f"[AnswerAgent] Applying {len(restrictions)} content restrictions")
        
        return restricted_answer
    
    def _sanitize_content(self, answer: str, compliance_result: Dict) -> str:
        """Sanitize content by removing sensitive information"""
        sanitized = answer
        
        # Remove specific technical details that might be ITAR-controlled
        sensitive_patterns = [
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'\b[A-Z]{2,}\s*-\s*\d+\b',  # Technical specifications
            r'\bclassified\s+as\s+\w+\b',  # Classification statements
        ]
        
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[RESTRICTED]', sanitized, flags=re.IGNORECASE)
        
        # Add sanitization notice if content was modified
        if sanitized != answer:
            sanitized += "\n\n[Note: Some technical details have been restricted for compliance reasons.]"
        
        return sanitized
    
    def _add_compliance_disclaimers(self, answer: str, compliance_result: Dict) -> str:
        """Add required compliance disclaimers"""
        disclaimers = compliance_result.get("content_guidance", {}).get("required_disclaimers", [])
        
        if disclaimers:
            answer += "\n\nCompliance Notice:\n"
            for disclaimer in disclaimers:
                answer += f"• {disclaimer}\n"
        
        return answer

# =============================================================================
# MODIFY YOUR SimpleStateOrchestrator CLASS
# =============================================================================

class SimpleStateOrchestratorWithCompliance(SimpleStateOrchestrator):
    """
    State Orchestrator with ITAR Compliance Integration
    """
    
    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        # Replace answer agent with compliance-aware version
        self.answer_agent = EnhancedAnswerAgentWithCompliance()
        self.compliance_client = itar_compliance_client
        
        print("[State Orchestrator] Initialized with ITAR compliance integration")
    
    def process_query(self, query: str, chat_history: List = None, documents_context: List = None,
                     user_profile: Dict = None) -> Dict[str, Any]:
        """Process query with compliance integration"""
        # Initialize state with user profile
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
        
        # Add user profile to state
        state['user_profile'] = user_profile
        
        try:
            # Execute workflow (same as parent but with compliance integration)
            current_step = WorkflowStep.INIT
            
            while current_step is not None:
                print(f"[State Orchestrator] Executing step: {current_step.value}")
                state['current_step'] = current_step.value
                state['execution_steps'].append(f"Step: {current_step.value}")
                
                # Execute step
                if current_step == WorkflowStep.ANSWER:
                    # Enhanced answer step with compliance
                    state = self._generate_compliance_aware_answer_step(state)
                else:
                    # Use parent class methods for other steps
                    state = self.workflow[current_step](state)
                
                # Check for error
                if state.get('error'):
                    current_step = WorkflowStep.ERROR
                else:
                    # Move to next step
                    current_step = self.transitions[current_step]
            
            execution_time = round(time.time() - state['start_time'], 2)
            
            # Extract compliance information from answer
            answer_result = state.get('answer_result', {})
            compliance_result = answer_result.get('compliance_result', {})
            
            return {
                "query": state['query'],
                "answer": answer_result.get('answer', state.get('answer', '')),
                "intent": state['intent_info'].get('intent', 'unknown') if state['intent_info'] else 'unknown',
                "entities_found": len(state['entity_info'].get('entities', [])) if state['entity_info'] else 0,
                "execution_time": execution_time,
                "execution_steps": state['execution_steps'],
                "success": state['error'] is None,
                "compliance_status": compliance_result.get('compliance_status', 'not_checked'),
                "content_restricted": answer_result.get('content_restricted', False),
                "restrictions_applied": answer_result.get('restrictions_applied', False),
                "metadata": {
                    "intent_confidence": state['intent_info'].get('confidence', 0) if state['intent_info'] else 0,
                    "entities": state['entity_info'].get('entities', []) if state['entity_info'] else [],
                    "system_version": "Integrated_Database_SAMM_with_ITAR_Compliance_v6.0",
                    "workflow_completed": state['current_step'] == 'complete',
                    "compliance_integration": {
                        "enabled": ITAR_COMPLIANCE_ENABLED,
                        "service_url": ITAR_COMPLIANCE_SERVICE_URL,
                        "compliance_status": compliance_result.get('compliance_status', 'not_checked'),
                        "authorized": compliance_result.get('authorized', True),
                        "restrictions_count": len(compliance_result.get('restrictions', []))
                    },
                    # Keep legacy metadata structure for Vue.js compatibility
                    "intent": state['intent_info'].get('intent', 'unknown') if state['intent_info'] else 'unknown',
                    "entities_found": len(state['entity_info'].get('entities', [])) if state['entity_info'] else 0,
                    "execution_time_seconds": execution_time,
                    # Add database integration status
                    "database_integration": {
                        "cosmos_gremlin": db_manager.cosmos_gremlin_client is not None,
                        "vector_db": db_manager.vector_db_client is not None,
                        "vector_db_ttl": db_manager.vector_db_ttl_client is not None,
                        "embedding_model": db_manager.embedding_model is not None
                    }
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
                "compliance_status": "error",
                "content_restricted": False,
                "restrictions_applied": False,
                "metadata": {"error": str(e), "system_version": "Integrated_Database_SAMM_with_ITAR_Compliance_v6.0"}
            }
    
    def _generate_compliance_aware_answer_step(self, state: AgentState) -> AgentState:
        """Execute compliance-aware answer generation step"""
        try:
            answer_result = self.answer_agent.generate_answer(
                state['query'], 
                state['intent_info'], 
                state['entity_info'], 
                state['chat_history'], 
                state['documents_context'],
                state.get('user_profile', {})
            )
            
            state['answer'] = answer_result['answer']
            state['answer_result'] = answer_result
            
            compliance_status = answer_result.get('compliance_result', {}).get('compliance_status', 'unknown')
            content_restricted = answer_result.get('content_restricted', False)
            
            state['execution_steps'].append(
                f"Compliance-aware answer generated (status: {compliance_status}, restricted: {content_restricted})"
            )
            
            print(f"[State Orchestrator] Compliance-aware answer generated "
                  f"(status: {compliance_status}, restricted: {content_restricted})")
                  
        except Exception as e:
            state['error'] = f"Compliance-aware answer generation failed: {str(e)}"
        
        return state
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get ITAR compliance service status"""
        return self.compliance_client.get_service_status()

# =============================================================================
# REPLACE YOUR ORCHESTRATOR INITIALIZATION
# =============================================================================

# Replace the existing orchestrator with compliance-aware version
# Comment out or replace this line in your main app:
# orchestrator = SimpleStateOrchestrator()

# Add this instead:
orchestrator_with_compliance = SimpleStateOrchestratorWithCompliance()
print("Integrated State Orchestrator with ITAR Compliance initialized")

# =============================================================================
# MODIFY YOUR MAIN QUERY ENDPOINT
# =============================================================================

def process_samm_query_with_compliance(query: str, chat_history: List = None, 
                                     documents_context: List = None, 
                                     user_profile: Dict = None) -> Dict[str, Any]:
    """Process query through integrated state orchestrated 3-agent system with ITAR compliance"""
    return orchestrator_with_compliance.process_query(query, chat_history, documents_context, user_profile)

# =============================================================================
# ADD NEW API ENDPOINTS TO YOUR MAIN APP.PY
# =============================================================================

@app.route("/api/compliance/status", methods=["GET"])
def get_itar_compliance_status():
    """Get ITAR compliance service status"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        status = orchestrator_with_compliance.get_compliance_status()
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/query_with_compliance", methods=["POST"])
def query_ai_assistant_with_compliance():
    """Enhanced SAMM query endpoint with ITAR compliance verification"""
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
        
        # Extract user profile for compliance checking
        user_profile = {
            "user_id": user_id,
            "name": user.get("name", ""),
            "email": user.get("email", ""),
            "authorization_level": data.get("authorization_level", "unclassified"),  # Should come from user management system
            "clearance_level": data.get("clearance_level", "unclassified"),
            "department": data.get("department", ""),
            "need_to_know": data.get("need_to_know", [])
        }
        
        if not user_input:
            return jsonify({"error": "Query cannot be empty"}), 400

        print(f"[ITAR-Integrated SAMM Query] User: {user_id}, Processing: '{user_input}'")
        print(f"[ITAR-Integrated SAMM Query] Authorization Level: {user_profile.get('authorization_level', 'unclassified')}")
        
        # Process through integrated system with ITAR compliance
        result = process_samm_query_with_compliance(user_input, chat_history, staged_chat_documents_metadata, user_profile)
        
        print(f"[ITAR-Integrated Result] Compliance Status: {result.get('compliance_status', 'unknown')}")
        print(f"[ITAR-Integrated Result] Content Restricted: {result.get('content_restricted', False)}")
        print(f"[ITAR-Integrated Result] Intent: {result['intent']}, Entities: {result['entities_found']}, Time: {result['execution_time']}s")
        
        # Return response in enhanced format with compliance information
        response_data = {
            "response": {"answer": result["answer"]},
            "metadata": result["metadata"],
            "compliance": {
                "status": result.get("compliance_status", "not_checked"),
                "content_restricted": result.get("content_restricted", False),
                "restrictions_applied": result.get("restrictions_applied", False),
                "service_enabled": ITAR_COMPLIANCE_ENABLED
            },
            "uploadedChatDocuments": []
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"[ITAR-Integrated SAMM Query] Error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# =============================================================================
# UPDATE YOUR SYSTEM STATUS ENDPOINT
# =============================================================================

@app.route("/api/system/status_with_compliance", methods=["GET"])
def get_system_status_with_compliance():
    """Get system status including ITAR compliance integration"""
    # Test Ollama connection
    try:
        test_response = call_ollama_enhanced("Test", "Respond with 'OK'", temperature=0.0)
        ollama_status = "connected" if "OK" in test_response else "error"
        ollama_available = True
    except:
        ollama_status = "disconnected"
        ollama_available = False
    
    # Get database status
    db_status = orchestrator_with_compliance.get_database_status()
    
    # Get compliance status
    compliance_status = orchestrator_with_compliance.get_compliance_status()
    
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
            "types": ["intent", "integrated_entity", "enhanced_answer_with_compliance"],
            "orchestration": "integrated_database_state_with_itar_compliance",
            "versions": {
                "intent_agent": "1.0",
                "entity_agent": "IntegratedEntityAgent v1.0",
                "answer_agent": "EnhancedAnswerAgentWithCompliance v1.0"
            }
        },
        "compliance_integration": {
            "itar_service_enabled": ITAR_COMPLIANCE_ENABLED,
            "itar_service_url": ITAR_COMPLIANCE_SERVICE_URL,
            "itar_service_status": compliance_status.get("status", "unknown"),
            "policy_frameworks": compliance_status.get("policy_frameworks", 0),
            "compliance_rules": compliance_status.get("compliance_rules", 0),
            "itar_categories": compliance_status.get("itar_categories", 0)
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
            "storage": "connected" if blob_service_client else "disabled",
            "itar_compliance": "enabled" if ITAR_COMPLIANCE_ENABLED else "disabled"
        },
        "version": "6.0.0-integrated-database-with-itar-compliance",
        "system_name": "Integrated Database SAMM ASIST with ITAR Compliance",
        "timestamp": datetime.now().isoformat()
    })

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

"""
INTEGRATION STEPS:

1. Save the ITAR compliance microservice as 'itar_compliance_microservice.py'

2. Add the code above to your main app.py file:
   - Add imports at the top
   - Add configuration variables
   - Add the ITARComplianceClient class
   - Replace EnhancedAnswerAgent with EnhancedAnswerAgentWithCompliance
   - Replace SimpleStateOrchestrator with SimpleStateOrchestratorWithCompliance
   - Add new API endpoints
   - Update existing endpoints

3. Set environment variables:
   ITAR_COMPLIANCE_SERVICE_URL=http://localhost:3002
   ITAR_COMPLIANCE_ENABLED=true

4. Start the ITAR compliance microservice:
   python itar_compliance_microservice.py

5. Start your main application:
   python app.py

6. Test the integration:
   - Use /api/query_with_compliance for compliance-aware queries
   - Use /api/compliance/status to check service status
   - Use /api/system/status_with_compliance for full system status

FEATURES:
- Automatic compliance verification before answer generation
- Content restriction and sanitization based on user authorization
- ITAR USML category detection and compliance checking
- Export control and technical data transfer compliance
- Content guidance for answer generation
- Required compliance disclaimers
- Fallback behavior when compliance service is unavailable
"""