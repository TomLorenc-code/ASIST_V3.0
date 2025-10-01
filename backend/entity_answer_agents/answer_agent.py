# enhanced_answer_agent_fixed.py
# Standalone Enhanced Answer Agent for SAMM Chapter 1 - Windows Compatible
# Optimized for Llama 3.2 with advanced answer generation capabilities

import os
import json
import re
import requests
import asyncio
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Fix for Windows asyncio issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

def call_ollama_enhanced(prompt: str, system_message: str = "", temperature: float = 0.1) -> str:
    """Enhanced Ollama API call optimized for Llama 3.2 with improved error handling"""
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
                "num_ctx": 4096,  # Optimized context window for Llama 3.2
                "num_predict": 1024  # Reduced for better reliability
            }
        }
        
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        print(f"[AnswerAgent] Ollama API error: {e}")
        return f"Error calling Ollama API: {str(e)}"
    except Exception as e:
        print(f"[AnswerAgent] Processing error: {e}")
        return f"Error processing with Ollama: {str(e)}"

class EnhancedAnswerAgent:
    """
    Enhanced Answer Agent for SAMM Chapter 1 with sophisticated response generation
    
    Features:
    - Windows compatibility and improved error handling
    - Intent-optimized prompt engineering for each question type
    - SAMM-specific response templates and quality standards
    - Multi-pass answer generation with validation
    - Learning system with HIL feedback and trigger updates
    - Automatic answer enhancement (acronym expansion, section references)
    - Answer caching and correction storage
    - Quality scoring and confidence assessment
    """
    
    def __init__(self):
        """Initialize the Enhanced Answer Agent with improved error handling"""
        print("[EnhancedAnswerAgent] Initializing...")
        
        # Learning and feedback systems
        self.hil_feedback_data = []        # Human-in-the-loop feedback storage
        self.answer_templates = {}         # Intent-specific answer templates
        self.trigger_updates = []          # Trigger-based updates storage
        self.custom_knowledge = ""         # Additional knowledge from updates
        self.answer_corrections = {}       # Stored answer corrections
        
        # SAMM-specific response templates for each intent type
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
        
        # Quality enhancement patterns for post-processing
        self.quality_patterns = {
            "section_references": r"(C\d+\.\d+\.?\d*\.?\d*)",
            "acronym_detection": r"\b([A-Z]{2,})\b",
            "authority_mentions": r"(Title \d+|[A-Z]+ Act)",
            "incomplete_sentences": r"[a-z]\s*$"
        }
        
        # Enhanced acronym expansion dictionary for SAMM
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
            "FMF": "Foreign Military Financing (FMF)",
            "NIPO": "Navy International Programs Office (NIPO)",
            "USASAC": "U.S. Army Security Assistance Command (USASAC)",
            "SATFA": "Security Assistance Training Field Activity (SATFA)",
            "CCDR": "Combatant Commander (CCDR)",
            "SCO": "Security Cooperation Organization (SCO)",
            "GEF": "Guidance for Employment of the Force (GEF)"
        }
        
        # Answer quality scoring weights
        self.quality_weights = {
            "section_citation": 0.25,
            "acronym_expansion": 0.15,
            "answer_completeness": 0.25,
            "samm_terminology": 0.20,
            "structure_adherence": 0.15
        }
        
        # Response length guidelines by intent
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
    
    def generate_answer(self, query: str, intent_info: Dict, entity_info: Dict, 
                       chat_history: List = None, documents_context: List = None) -> str:
        """
        Main method for enhanced answer generation with improved error handling
        
        Uses sophisticated multi-pass approach:
        1. Check for cached corrections
        2. Build comprehensive context
        3. Create intent-optimized system message
        4. Generate enhanced prompt
        5. Multi-pass generation with validation
        6. Apply quality enhancements
        7. Score and validate final answer
        
        Args:
            query: User's question
            intent_info: Intent classification results
            entity_info: Entity extraction results
            chat_history: Previous conversation context
            documents_context: Attached document context
            
        Returns:
            Enhanced, validated answer string
        """
        intent = intent_info.get("intent", "general")
        confidence = intent_info.get("confidence", 0.5)
        
        print(f"[Enhanced AnswerAgent] Generating answer for intent: {intent} (confidence: {confidence:.2f})")
        
        try:
            # Step 1: Check for existing corrections first
            cached_answer = self._check_for_corrections(query, intent_info, entity_info)
            if cached_answer:
                print("[AnswerAgent] Using cached correction")
                return cached_answer
            
            # Step 2: Build comprehensive context from all sources
            context = self._build_comprehensive_context(
                query, intent_info, entity_info, chat_history, documents_context
            )
            
            # Step 3: Create intent-optimized system message
            system_msg = self._create_optimized_system_message(intent, context)
            
            # Step 4: Generate enhanced prompt with intent awareness
            prompt = self._create_enhanced_prompt(query, intent_info, entity_info)
            
            # Step 5: Generate answer with validation passes
            answer = self._generate_with_validation(prompt, system_msg, intent_info)
            
            # Step 6: Apply post-processing enhancements
            enhanced_answer = self._enhance_answer_quality(answer, intent_info, entity_info)
            
            # Step 7: Final validation and scoring
            final_answer = self._validate_and_score_answer(enhanced_answer, intent, query)
            
            print(f"[AnswerAgent] Generated answer: {len(final_answer)} characters")
            return final_answer
            
        except Exception as e:
            print(f"[AnswerAgent] Error during answer generation: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}. Please try rephrasing your question or check if the Ollama service is running."
    
    def _check_for_corrections(self, query: str, intent_info: Dict, entity_info: Dict) -> Optional[str]:
        """Check if we have a stored correction for similar queries"""
        try:
            query_key = self._normalize_query_for_matching(query)
            
            # Check exact matches first
            if query_key in self.answer_corrections:
                correction = self.answer_corrections[query_key]
                print(f"[AnswerAgent] Found exact correction match")
                return correction["corrected_answer"]
            
            # Check for partial matches based on intent and entities
            current_entities = set(entity_info.get("entities", []))
            current_intent = intent_info.get("intent", "general")
            
            for stored_key, correction in self.answer_corrections.items():
                stored_entities = set(correction.get("feedback_data", {}).get("entities", []))
                stored_intent = correction.get("feedback_data", {}).get("intent", "general")
                
                # If same intent and significant entity overlap (50% or more)
                if (current_intent == stored_intent and len(current_entities) > 0 and
                    len(current_entities.intersection(stored_entities)) >= min(len(current_entities), len(stored_entities)) * 0.5):
                    print(f"[AnswerAgent] Found partial correction match based on intent/entities")
                    return correction["corrected_answer"]
            
            return None
            
        except Exception as e:
            print(f"[AnswerAgent] Error checking corrections: {e}")
            return None
    
    def _build_comprehensive_context(self, query: str, intent_info: Dict, entity_info: Dict,
                                   chat_history: List = None, documents_context: List = None) -> str:
        """Build comprehensive context for answer generation with error handling"""
        try:
            context_parts = []
            
            # Add entity context with confidence weighting
            if entity_info.get("context"):
                context_parts.append("=== SAMM ENTITIES AND DEFINITIONS ===")
                for ctx in entity_info["context"][:5]:  # Limit to 5 to prevent overload
                    confidence = ctx.get('confidence', 0.5)
                    if confidence > 0.6:  # Only include high-confidence entities
                        entity_text = f"{ctx.get('entity', '')}: {ctx.get('definition', '')}"
                        if ctx.get('section'):
                            entity_text += f" (SAMM {ctx['section']})"
                        context_parts.append(entity_text)
            
            # Add relevant text sections from SAMM
            if entity_info.get("text_sections"):
                context_parts.append("\n=== SAMM CHAPTER 1 CONTENT ===")
                # Limit context to prevent overload
                text_sections = entity_info["text_sections"][:3]
                for section in text_sections:
                    # Truncate very long sections
                    truncated_section = section[:500] + "..." if len(section) > 500 else section
                    context_parts.append(truncated_section)
            
            # Add entity relationships
            if entity_info.get("relationships"):
                context_parts.append("\n=== ENTITY RELATIONSHIPS ===")
                # Limit to most relevant relationships
                context_parts.extend(entity_info["relationships"][:3])
            
            # Add custom knowledge from HIL feedback and triggers
            if self.custom_knowledge:
                context_parts.append("\n=== ADDITIONAL KNOWLEDGE ===")
                # Truncate if too long
                knowledge = self.custom_knowledge[:1000] + "..." if len(self.custom_knowledge) > 1000 else self.custom_knowledge
                context_parts.append(knowledge)
            
            # Add relevant chat history for continuity
            if chat_history and len(chat_history) > 0:
                context_parts.append("\n=== CONVERSATION CONTEXT ===")
                for msg in chat_history[-2:]:  # Last 2 messages for context
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200]  # Truncate long messages
                    context_parts.append(f"{role}: {content}")
            
            # Add document context if available
            if documents_context:
                context_parts.append("\n=== REFERENCED DOCUMENTS ===")
                for doc in documents_context[:3]:  # Limit to 3 documents
                    context_parts.append(f"- {doc.get('fileName', 'Unknown file')}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"[AnswerAgent] Error building context: {e}")
            return "Context building failed - proceeding with basic knowledge."
    
    def _create_optimized_system_message(self, intent: str, context: str) -> str:
        """Create intent-optimized system message for Llama 3.2 with error handling"""
        
        try:
            # Base instructions for each intent type
            base_instructions = {
                "definition": """You are a SAMM (Security Assistance Management Manual) expert specializing in precise definitions.

TASK: Provide authoritative definitions with exact SAMM section citations.

CRITICAL REQUIREMENTS:
- Use exact SAMM terminology and definitions from the context
- Always cite specific SAMM sections (e.g., "SAMM C1.3.2.2")
- Expand acronyms on first use (e.g., "Defense Security Cooperation Agency (DSCA)")
- Distinguish between Security Cooperation and Security Assistance
- Remember: SA is a SUBSET of SC, not the same thing
- Provide clear, complete definitions that could stand alone

RESPONSE STRUCTURE:
1. Clear definition statement
2. SAMM section citation
3. Additional context about authority/oversight if relevant""",

                "distinction": """You are a SAMM expert specializing in explaining key distinctions and differences.

TASK: Clearly explain differences between SAMM concepts with precise legal and operational distinctions.

CRITICAL REQUIREMENTS:
- Highlight key differences clearly and systematically
- Explain legal authority differences (Title 10 vs Title 22)
- Use specific examples when possible
- Always emphasize that Security Assistance is a subset of Security Cooperation
- Cite relevant SAMM sections for each concept being compared
- Address common misconceptions

RESPONSE STRUCTURE:
1. State the key distinction clearly
2. Explain each concept separately with citations
3. Highlight the differences with examples
4. Summarize the relationship""",

                "authority": """You are a SAMM expert specializing in authority and oversight structures.

TASK: Explain who has authority, oversight, and responsibility for specific programs.

CRITICAL REQUIREMENTS:
- Clearly state which organization/person has authority
- Explain the scope of authority and oversight
- Cite legal authorities (FAA, AECA, NDAA, Executive Orders)
- Distinguish between "supervision," "direction," and "oversight"
- Reference specific SAMM sections
- Explain delegation chains where applicable

RESPONSE STRUCTURE:
1. State who has the authority
2. Explain the scope and basis of authority
3. Cite legal foundations
4. Describe any delegation or coordination requirements""",

                "organization": """You are a SAMM expert specializing in organizational roles and responsibilities.

TASK: Describe organizations, their roles, and specific responsibilities.

CRITICAL REQUIREMENTS:
- Provide full organization names and acronyms
- List specific roles and responsibilities clearly
- Explain relationships between organizations
- Cite relevant SAMM sections
- Include key personnel authorities where applicable
- Describe organizational structure and reporting relationships

RESPONSE STRUCTURE:
1. Full name and acronym
2. Primary role and mission
3. Specific responsibilities
4. Reporting relationships and coordination""",

                "factual": """You are a SAMM expert providing specific factual information.

TASK: Provide accurate, specific facts from SAMM Chapter 1.

CRITICAL REQUIREMENTS:
- Provide precise, accurate information
- Include dates, numbers, and specific details
- Cite SAMM sections for verification
- Use exact terminology from SAMM
- Expand acronyms appropriately

RESPONSE STRUCTURE:
1. Direct answer to the factual question
2. Supporting context
3. Source citation""",

                "relationship": """You are a SAMM expert explaining relationships between entities and concepts.

TASK: Describe how SAMM entities, programs, and authorities relate to each other.

CRITICAL REQUIREMENTS:
- Clearly explain the nature of relationships
- Use specific examples to illustrate connections
- Cite relevant authorities and SAMM sections
- Explain the significance of relationships
- Address coordination and oversight aspects

RESPONSE STRUCTURE:
1. Describe the relationship clearly
2. Explain why the relationship exists
3. Provide examples of how it works in practice
4. Cite supporting authorities""",

                "general": """You are a SAMM (Security Assistance Management Manual) Chapter 1 expert.

TASK: Provide comprehensive, accurate information about Security Cooperation and Security Assistance.

CRITICAL REQUIREMENTS:
- Use exact SAMM terminology from the provided context
- Always cite SAMM sections when available
- Expand acronyms on first use
- Maintain distinction between SC and SA (SA is subset of SC)
- Provide authoritative, accurate information
- Structure responses logically and completely"""
            }
            
            system_msg = base_instructions.get(intent, base_instructions["general"])
            
            # Add learned improvements from HIL feedback
            if intent in self.answer_templates and self.answer_templates[intent]:
                system_msg += "\n\nIMPORTANT IMPROVEMENTS FROM FEEDBACK:"
                for template in self.answer_templates[intent][-2:]:  # Last 2 templates
                    if template.get("improvement_notes"):
                        system_msg += f"\n- {template['improvement_notes']}"
                    if template.get("key_points"):
                        system_msg += f"\n- Ensure to mention: {', '.join(template['key_points'])}"
            
            # Add template structure guidance
            if intent in self.samm_response_templates:
                template = self.samm_response_templates[intent]
                system_msg += f"\n\nRESPONSE STRUCTURE: {template['structure']}"
                system_msg += f"\nREQUIRED ELEMENTS: {', '.join(template['required_elements'])}"
                system_msg += f"\nQUALITY CRITERIA: {', '.join(template['quality_criteria'])}"
            
            # Add length guidelines
            if intent in self.length_guidelines:
                guidelines = self.length_guidelines[intent]
                system_msg += f"\n\nLENGTH GUIDELINES: Target {guidelines['target']} characters (minimum {guidelines['min']}, maximum {guidelines['max']})"
            
            # Add context (truncated if necessary)
            context_truncated = context[:2000] + "..." if len(context) > 2000 else context
            system_msg += f"\n\nCONTEXT FROM SAMM:\n{context_truncated}"
            
            return system_msg
            
        except Exception as e:
            print(f"[AnswerAgent] Error creating system message: {e}")
            return "You are a SAMM expert. Provide accurate information about Security Cooperation and Security Assistance."
    
    def _create_enhanced_prompt(self, query: str, intent_info: Dict, entity_info: Dict) -> str:
        """Create enhanced prompt with entity and intent awareness"""
        try:
            intent = intent_info.get("intent", "general")
            entities = entity_info.get("entities", [])
            confidence = intent_info.get("confidence", 0.5)
            
            prompt_parts = []
            
            # Add query with context
            prompt_parts.append(f"Question: {query}")
            
            # Add intent guidance if high confidence
            if confidence > 0.7:
                prompt_parts.append(f"This is a {intent} question requiring a {intent}-focused response.")
            
            # Add entity awareness (limit to prevent overload)
            if entities:
                limited_entities = entities[:3]  # Limit to 3 entities
                prompt_parts.append(f"Key entities mentioned: {', '.join(limited_entities)}")
            
            # Add specific instructions based on intent
            intent_instructions = {
                "definition": "Provide a complete, authoritative definition with proper SAMM section reference.",
                "distinction": "Explain the key differences clearly with specific examples and legal basis.",
                "authority": "Explain who has authority, the scope of that authority, and the legal basis.",
                "organization": "Describe the organization's full name, role, and specific responsibilities.",
                "factual": "Provide the specific factual information with proper context and citation.",
                "relationship": "Describe how the entities relate to each other and why this matters."
            }
            
            if intent in intent_instructions:
                prompt_parts.append(intent_instructions[intent])
            
            prompt_parts.append("Provide a comprehensive, accurate answer based on SAMM Chapter 1 content.")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            print(f"[AnswerAgent] Error creating prompt: {e}")
            return f"Question: {query}\nProvide a comprehensive answer based on SAMM Chapter 1."
    
    def _generate_with_validation(self, prompt: str, system_msg: str, intent_info: Dict) -> str:
        """Generate answer with multiple validation passes and improved error handling"""
        intent = intent_info.get("intent", "general")
        
        try:
            # First generation pass
            print("[AnswerAgent] First generation pass...")
            initial_answer = call_ollama_enhanced(prompt, system_msg, temperature=0.1)
            
            # Check if answer indicates an error
            if "Error" in initial_answer and len(initial_answer) < 100:
                return initial_answer  # Return error as-is
            
            # Validate answer quality
            validation_results = self._validate_answer_quality(initial_answer, intent)
            
            # If answer needs improvement, try enhanced generation
            if validation_results["needs_improvement"] and len(validation_results["issues"]) <= 2:  # Don't retry if too many issues
                print(f"[AnswerAgent] Answer needs improvement: {validation_results['issues']}")
                
                # Create enhanced prompt with specific improvement requests
                improvement_prompt = f"{prompt}\n\nIMPROVEMENT NEEDED: {', '.join(validation_results['issues'])}\n\nPlease provide a better response addressing these issues."
                
                print("[AnswerAgent] Second generation pass with improvements...")
                improved_answer = call_ollama_enhanced(improvement_prompt, system_msg, temperature=0.2)
                
                # Use improved answer if it's actually better and doesn't contain errors
                if (len(improved_answer) > len(initial_answer) * 1.1 and 
                    "Error" not in improved_answer and
                    len(improved_answer) > 50):  # At least 10% longer and valid
                    return improved_answer
            
            return initial_answer
            
        except Exception as e:
            print(f"[AnswerAgent] Error during generation with validation: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"
    
    def _validate_answer_quality(self, answer: str, intent: str) -> Dict[str, Any]:
        """Validate answer quality against SAMM standards with error handling"""
        try:
            issues = []
            needs_improvement = False
            
            # Skip validation if answer is too short (likely an error)
            if len(answer) < 20:
                return {"needs_improvement": False, "issues": ["answer_too_short"], "length": len(answer)}
            
            # Check length guidelines
            if intent in self.length_guidelines:
                guidelines = self.length_guidelines[intent]
                if len(answer) < guidelines["min"]:
                    issues.append("too short")
                    needs_improvement = True
                elif len(answer) > guidelines["max"]:
                    issues.append("too long")
            
            # Check for SAMM section references
            if not re.search(self.quality_patterns["section_references"], answer):
                issues.append("missing SAMM section reference")
                needs_improvement = True
            
            # Check for incomplete sentences
            if re.search(self.quality_patterns["incomplete_sentences"], answer):
                issues.append("incomplete sentences")
                needs_improvement = True
            
            # Intent-specific validations
            if intent == "definition" and "definition" not in answer.lower():
                issues.append("missing clear definition")
                needs_improvement = True
            
            if intent == "distinction" and not any(word in answer.lower() for word in ["difference", "differ", "distinction", "versus", "vs"]):
                issues.append("missing comparison language")
                needs_improvement = True
            
            if intent == "authority" and not any(word in answer.lower() for word in ["authority", "responsible", "oversight", "supervision"]):
                issues.append("missing authority language")
                needs_improvement = True
            
            return {
                "needs_improvement": needs_improvement,
                "issues": issues,
                "length": len(answer)
            }
            
        except Exception as e:
            print(f"[AnswerAgent] Error validating answer quality: {e}")
            return {"needs_improvement": False, "issues": [], "length": len(answer)}
    
    def _enhance_answer_quality(self, answer: str, intent_info: Dict, entity_info: Dict) -> str:
        """Apply post-processing enhancements with error handling"""
        try:
            enhanced_answer = answer
            
            # Skip enhancement if answer is too short or contains errors
            if len(answer) < 20 or "Error" in answer:
                return answer
            
            # Step 1: Add section references if missing critical ones
            if not re.search(self.quality_patterns["section_references"], enhanced_answer):
                entities = entity_info.get("entities", [])
                if entities and any(entity in ["DSCA", "Security Assistance", "Security Cooperation"] for entity in entities):
                    enhanced_answer += "\n\nRefer to relevant SAMM Chapter 1 sections for complete details."
            
            # Step 2: Expand acronyms that appear without expansion (limit to prevent overprocessing)
            acronyms_found = re.findall(self.quality_patterns["acronym_detection"], enhanced_answer)
            
            for acronym in list(set(acronyms_found))[:5]:  # Limit to 5 acronyms
                if (acronym in self.acronym_expansions and 
                    acronym in enhanced_answer and 
                    self.acronym_expansions[acronym] not in enhanced_answer):
                    # Only expand the first occurrence
                    enhanced_answer = enhanced_answer.replace(acronym, self.acronym_expansions[acronym], 1)
            
            # Step 3: Ensure proper SAMM terminology
            terminology_fixes = {
                "security cooperation": "Security Cooperation",
                "security assistance": "Security Assistance", 
                "foreign assistance act": "Foreign Assistance Act",
                "arms export control act": "Arms Export Control Act"
            }
            
            for incorrect, correct in terminology_fixes.items():
                if incorrect in enhanced_answer and correct not in enhanced_answer:
                    enhanced_answer = enhanced_answer.replace(incorrect, correct)
            
            # Step 4: Add intent-specific enhancements
            intent = intent_info.get("intent", "general")
            
            if intent == "distinction" and "subset" not in enhanced_answer.lower():
                if "Security Assistance" in enhanced_answer and "Security Cooperation" in enhanced_answer:
                    enhanced_answer += "\n\nRemember: Security Assistance is a subset of Security Cooperation."
            
            return enhanced_answer
            
        except Exception as e:
            print(f"[AnswerAgent] Error enhancing answer quality: {e}")
            return answer  # Return original if enhancement fails
    
    def _validate_and_score_answer(self, answer: str, intent: str, query: str) -> str:
        """Final validation and quality scoring of the answer with error handling"""
        try:
            # Skip scoring if answer is too short or contains errors
            if len(answer) < 20 or "Error" in answer:
                return answer
            
            # Calculate quality score
            score = self._calculate_quality_score(answer, intent)
            
            # Log quality metrics
            print(f"[AnswerAgent] Answer quality score: {score:.2f}/1.0")
            
            # If score is too low, add disclaimer
            if score < 0.6:
                print(f"[AnswerAgent] Low quality score, adding disclaimer")
                answer += "\n\nNote: For complete and authoritative information, please refer to the full SAMM Chapter 1 documentation."
            
            return answer
            
        except Exception as e:
            print(f"[AnswerAgent] Error in final validation: {e}")
            return answer  # Return original if validation fails
    
    def _calculate_quality_score(self, answer: str, intent: str) -> float:
        """Calculate quality score based on SAMM standards with error handling"""
        try:
            score = 0.0
            
            # Section citation score
            if re.search(self.quality_patterns["section_references"], answer):
                score += self.quality_weights["section_citation"]
            
            # Acronym expansion score
            acronyms_found = re.findall(self.quality_patterns["acronym_detection"], answer)
            if acronyms_found:
                expanded_count = sum(1 for acronym in acronyms_found if f"{acronym})" in answer)
                score += self.quality_weights["acronym_expansion"] * (expanded_count / len(set(acronyms_found)))
            
            # Answer completeness score (based on length guidelines)
            if intent in self.length_guidelines:
                guidelines = self.length_guidelines[intent]
                if guidelines["min"] <= len(answer) <= guidelines["max"]:
                    score += self.quality_weights["answer_completeness"]
                elif len(answer) >= guidelines["target"]:
                    score += self.quality_weights["answer_completeness"] * 0.8
            
            # SAMM terminology score
            samm_terms = ["Security Cooperation", "Security Assistance", "SAMM", "Title 10", "Title 22"]
            terms_used = sum(1 for term in samm_terms if term in answer)
            if terms_used > 0:
                score += self.quality_weights["samm_terminology"] * min(1.0, terms_used / 3)
            
            # Structure adherence score
            if intent in self.samm_response_templates:
                required_elements = self.samm_response_templates[intent]["required_elements"]
                elements_present = 0
                for element in required_elements:
                    element_keywords = element.replace("_", " ").split()
                    if any(keyword in answer.lower() for keyword in element_keywords):
                        elements_present += 1
                
                if required_elements:
                    score += self.quality_weights["structure_adherence"] * (elements_present / len(required_elements))
            
            return min(1.0, score)  # Cap at 1.0
            
        except Exception as e:
            print(f"[AnswerAgent] Error calculating quality score: {e}")
            return 0.5  # Return moderate score on error
    
    def _normalize_query_for_matching(self, query: str) -> str:
        """Normalize query for matching similar questions"""
        try:
            # Simple normalization - remove punctuation, lowercase, sort words
            words = re.findall(r'\b\w+\b', query.lower())
            # Keep only significant words (length > 2)
            significant_words = [word for word in words if len(word) > 2]
            return " ".join(sorted(significant_words))
        except Exception as e:
            print(f"[AnswerAgent] Error normalizing query: {e}")
            return query.lower()
    
    def update_from_hil(self, query: str, original_answer: str, corrected_answer: str, 
                        feedback_data: Dict[str, Any] = None):
        """
        Update agent based on human-in-the-loop feedback with improved error handling
        
        Args:
            query: Original query
            original_answer: AI-generated answer
            corrected_answer: Human-corrected answer
            feedback_data: Additional feedback information
        """
        try:
            feedback_entry = {
                "query": query,
                "original_answer": original_answer,
                "corrected_answer": corrected_answer,
                "feedback_data": feedback_data or {},
                "timestamp": datetime.now().isoformat(),
                "improvement_type": "hil_correction"
            }
            
            self.hil_feedback_data.append(feedback_entry)
            
            # Store the correction for future similar queries
            query_key = self._normalize_query_for_matching(query)
            self.answer_corrections[query_key] = {
                "corrected_answer": corrected_answer,
                "feedback_data": feedback_data,
                "original_query": query,
                "correction_date": datetime.now().isoformat()
            }
            
            # Extract and store improved patterns
            if feedback_data:
                intent = feedback_data.get("intent", "general")
                if intent not in self.answer_templates:
                    self.answer_templates[intent] = []
                
                # Store template patterns from corrections
                template_info = {
                    "query_pattern": query.lower(),
                    "improvement_notes": feedback_data.get("improvement_notes", ""),
                    "key_points": feedback_data.get("key_points", []),
                    "structure_notes": feedback_data.get("structure_notes", ""),
                    "feedback_date": datetime.now().isoformat()
                }
                self.answer_templates[intent].append(template_info)
            
            # Add any new knowledge provided in feedback
            if feedback_data and feedback_data.get("additional_knowledge"):
                self.custom_knowledge += f"\n\nHIL Update ({datetime.now().strftime('%Y-%m-%d')}):\n{feedback_data['additional_knowledge']}"
            
            print(f"[AnswerAgent HIL] Updated with correction for query: '{query[:50]}...'")
            print(f"[AnswerAgent HIL] Total corrections stored: {len(self.answer_corrections)}")
            return True
            
        except Exception as e:
            print(f"[AnswerAgent] Error updating from HIL feedback: {e}")
            return False
    
    def update_from_trigger(self, new_entities: List[str], new_relationships: List[Dict], 
                           trigger_data: Dict[str, Any] = None):
        """
        Update agent when new entity/relationship data is available with error handling
        
        Args:
            new_entities: List of new entities to incorporate
            new_relationships: List of new relationships to incorporate
            trigger_data: Additional trigger information
        """
        try:
            trigger_entry = {
                "new_entities": new_entities,
                "new_relationships": new_relationships,
                "trigger_data": trigger_data or {},
                "timestamp": datetime.now().isoformat(),
                "trigger_id": len(self.trigger_updates)
            }
            
            self.trigger_updates.append(trigger_entry)
            
            # Add new knowledge from trigger updates
            if trigger_data:
                new_knowledge_items = []
                
                # Add entity definitions
                for entity in new_entities:
                    definition = trigger_data.get("entity_definitions", {}).get(entity)
                    if definition:
                        new_knowledge_items.append(f"{entity}: {definition}")
                
                # Add relationship information
                for rel in new_relationships:
                    rel_info = f"{rel.get('source', '')} {rel.get('relationship', '')} {rel.get('target', '')}"
                    details = trigger_data.get("relationship_details", {}).get(rel_info)
                    if details:
                        new_knowledge_items.append(f"Relationship: {rel_info} - {details}")
                
                # Add any general knowledge updates
                if trigger_data.get("knowledge_updates"):
                    new_knowledge_items.extend(trigger_data["knowledge_updates"])
                
                if new_knowledge_items:
                    self.custom_knowledge += f"\n\nTrigger Update ({datetime.now().strftime('%Y-%m-%d')}):\n" + "\n".join(new_knowledge_items)
            
            print(f"[AnswerAgent Trigger] Updated with {len(new_entities)} new entities and {len(new_relationships)} relationships")
            print(f"[AnswerAgent Trigger] Total trigger updates: {len(self.trigger_updates)}")
            return True
            
        except Exception as e:
            print(f"[AnswerAgent] Error updating from trigger: {e}")
            return False
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the Answer Agent with error handling
        
        Returns:
            Dictionary containing agent status and statistics
        """
        try:
            return {
                "agent_type": "EnhancedAnswerAgent",
                "version": "1.0-Fixed",
                "initialization_time": datetime.now().isoformat(),
                "ollama_config": {
                    "base_url": OLLAMA_BASE_URL,
                    "model": OLLAMA_MODEL
                },
                "response_capabilities": {
                    "supported_intents": list(self.samm_response_templates.keys()),
                    "total_templates": len(self.samm_response_templates),
                    "quality_criteria_count": sum(len(template["quality_criteria"]) for template in self.samm_response_templates.values()),
                    "acronym_expansions": len(self.acronym_expansions)
                },
                "learning_status": {
                    "hil_feedback_count": len(self.hil_feedback_data),
                    "trigger_update_count": len(self.trigger_updates),
                    "answer_corrections_count": len(self.answer_corrections),
                    "learned_templates_count": sum(len(templates) for templates in self.answer_templates.values()),
                    "custom_knowledge_length": len(self.custom_knowledge)
                },
                "generation_features": {
                    "multi_pass_generation": True,
                    "quality_scoring": True,
                    "automatic_enhancement": True,
                    "answer_validation": True,
                    "length_optimization": True,
                    "windows_compatibility": True,
                    "error_recovery": True
                },
                "quality_weights": self.quality_weights,
                "length_guidelines": self.length_guidelines
            }
        except Exception as e:
            print(f"[AnswerAgent] Error getting agent status: {e}")
            return {"agent_type": "EnhancedAnswerAgent", "status": "error", "error": str(e)}
    
    def get_answer_analysis(self, answer: str, intent: str) -> Dict[str, Any]:
        """
        Analyze a generated answer for quality metrics with error handling
        
        Args:
            answer: The answer to analyze
            intent: The intent type for analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            analysis = {
                "length_analysis": {
                    "character_count": len(answer),
                    "word_count": len(answer.split()),
                    "meets_guidelines": False
                },
                "quality_score": self._calculate_quality_score(answer, intent),
                "samm_compliance": {
                    "has_section_reference": bool(re.search(self.quality_patterns["section_references"], answer)),
                    "acronyms_found": re.findall(self.quality_patterns["acronym_detection"], answer),
                    "authority_mentions": re.findall(self.quality_patterns["authority_mentions"], answer)
                },
                "template_adherence": {},
                "enhancement_opportunities": []
            }
            
            # Check length guidelines
            if intent in self.length_guidelines:
                guidelines = self.length_guidelines[intent]
                analysis["length_analysis"]["guidelines"] = guidelines
                analysis["length_analysis"]["meets_guidelines"] = (
                    guidelines["min"] <= len(answer) <= guidelines["max"]
                )
                
                if len(answer) < guidelines["min"]:
                    analysis["enhancement_opportunities"].append("Answer could be more detailed")
                elif len(answer) > guidelines["max"]:
                    analysis["enhancement_opportunities"].append("Answer could be more concise")
            
            # Check template adherence
            if intent in self.samm_response_templates:
                template = self.samm_response_templates[intent]
                elements_present = []
                elements_missing = []
                
                for element in template["required_elements"]:
                    element_keywords = element.replace("_", " ").split()
                    if any(keyword in answer.lower() for keyword in element_keywords):
                        elements_present.append(element)
                    else:
                        elements_missing.append(element)
                
                analysis["template_adherence"] = {
                    "required_elements": template["required_elements"],
                    "elements_present": elements_present,
                    "elements_missing": elements_missing,
                    "adherence_score": len(elements_present) / len(template["required_elements"]) if template["required_elements"] else 1.0
                }
                
                if elements_missing:
                    analysis["enhancement_opportunities"].extend([f"Consider adding {element.replace('_', ' ')}" for element in elements_missing])
            
            # Check for enhancement opportunities
            if not analysis["samm_compliance"]["has_section_reference"]:
                analysis["enhancement_opportunities"].append("Add SAMM section reference")
            
            acronyms = analysis["samm_compliance"]["acronyms_found"]
            unexpanded_acronyms = [acr for acr in acronyms if f"{acr})" not in answer and acr in self.acronym_expansions]
            if unexpanded_acronyms:
                analysis["enhancement_opportunities"].append(f"Expand acronyms: {', '.join(unexpanded_acronyms)}")
            
            return analysis
            
        except Exception as e:
            print(f"[AnswerAgent] Error analyzing answer: {e}")
            return {
                "length_analysis": {"character_count": len(answer), "word_count": len(answer.split()), "meets_guidelines": False},
                "quality_score": 0.5,
                "samm_compliance": {"has_section_reference": False, "acronyms_found": [], "authority_mentions": []},
                "template_adherence": {},
                "enhancement_opportunities": ["Error during analysis"],
                "analysis_error": str(e)
            }
    
    def search_corrections(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for stored corrections matching a term with error handling
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of matching corrections
        """
        try:
            matches = []
            search_lower = search_term.lower()
            
            for query_key, correction in self.answer_corrections.items():
                if (search_lower in correction["original_query"].lower() or
                    search_lower in correction["corrected_answer"].lower()):
                    matches.append({
                        "query": correction["original_query"],
                        "corrected_answer": correction["corrected_answer"][:200] + "...",
                        "correction_date": correction.get("correction_date", "Unknown"),
                        "feedback_data": correction.get("feedback_data", {})
                    })
            
            return matches
            
        except Exception as e:
            print(f"[AnswerAgent] Error searching corrections: {e}")
            return []
    
    def validate_answer_generation(self, test_scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Validate answer generation with test scenarios and improved error handling
        
        Args:
            test_scenarios: List of test scenarios with query, intent_info, entity_info
            
        Returns:
            Validation results
        """
        results = []
        
        for scenario in test_scenarios:
            try:
                query = scenario.get("query", "")
                intent_info = scenario.get("intent_info", {"intent": "general", "confidence": 0.8})
                entity_info = scenario.get("entity_info", {"entities": [], "context": []})
                
                # Generate answer
                answer = self.generate_answer(query, intent_info, entity_info)
                
                # Check for error answers
                is_error_answer = "Error" in answer and len(answer) < 100
                
                # Analyze quality (skip if error answer)
                if not is_error_answer:
                    analysis = self.get_answer_analysis(answer, intent_info.get("intent", "general"))
                else:
                    analysis = {
                        "quality_score": 0.0,
                        "length_analysis": {"meets_guidelines": False},
                        "samm_compliance": {"has_section_reference": False},
                        "template_adherence": {"adherence_score": 0.0}
                    }
                
                results.append({
                    "query": query,
                    "answer_length": len(answer),
                    "quality_score": analysis["quality_score"],
                    "meets_guidelines": analysis["length_analysis"]["meets_guidelines"],
                    "has_section_reference": analysis["samm_compliance"]["has_section_reference"],
                    "template_adherence": analysis["template_adherence"].get("adherence_score", 0),
                    "success": not is_error_answer,
                    "is_error_answer": is_error_answer
                })
                
            except Exception as e:
                results.append({
                    "query": scenario.get("query", "Unknown"),
                    "answer_length": 0,
                    "quality_score": 0.0,
                    "meets_guidelines": False,
                    "has_section_reference": False,
                    "template_adherence": 0.0,
                    "success": False,
                    "error": str(e),
                    "is_error_answer": True
                })
        
        # Calculate summary statistics
        successful_generations = [r for r in results if r["success"]]
        
        try:
            summary = {
                "total_scenarios": len(test_scenarios),
                "successful_generations": len(successful_generations),
                "success_rate": len(successful_generations) / len(test_scenarios) if test_scenarios else 0,
                "average_quality_score": sum(r["quality_score"] for r in successful_generations) / len(successful_generations) if successful_generations else 0,
                "average_answer_length": sum(r["answer_length"] for r in successful_generations) / len(successful_generations) if successful_generations else 0,
                "guideline_compliance_rate": sum(1 for r in successful_generations if r["meets_guidelines"]) / len(successful_generations) if successful_generations else 0,
                "section_reference_rate": sum(1 for r in successful_generations if r["has_section_reference"]) / len(successful_generations) if successful_generations else 0,
                "average_template_adherence": sum(r["template_adherence"] for r in successful_generations) / len(successful_generations) if successful_generations else 0,
                "error_count": sum(1 for r in results if r.get("is_error_answer", False))
            }
        except Exception as e:
            print(f"[AnswerAgent] Error calculating summary statistics: {e}")
            summary = {"total_scenarios": len(test_scenarios), "error": str(e)}
        
        return {
            "validation_results": results,
            "summary": summary
        }
    
    def cleanup(self):
        """Cleanup resources and connections"""
        try:
            print("[EnhancedAnswerAgent] Cleanup complete")
        except Exception as e:
            print(f"[EnhancedAnswerAgent] Cleanup error: {e}")

# Example usage and testing functions with improved error handling
def test_enhanced_answer_agent():
    """Test the Enhanced Answer Agent with sample scenarios and better error handling"""
    print("="*60)
    print("Testing Fixed Enhanced Answer Agent")
    print("="*60)
    
    agent = None
    
    try:
        # Initialize agent
        agent = EnhancedAnswerAgent()
        
        # Test scenarios
        test_scenarios = [
            {
                "query": "What does DSCA do?",
                "intent_info": {"intent": "organization", "confidence": 0.9},
                "entity_info": {
                    "entities": ["DSCA", "Defense Security Cooperation Agency"],
                    "context": [
                        {
                            "entity": "DSCA",
                            "definition": "Defense Security Cooperation Agency directs, administers, and provides guidance to DoD Components",
                            "section": "C1.3.2.2",
                            "confidence": 0.95
                        }
                    ]
                }
            },
            {
                "query": "What's the difference between Security Cooperation and Security Assistance?",
                "intent_info": {"intent": "distinction", "confidence": 0.95},
                "entity_info": {
                    "entities": ["Security Cooperation", "Security Assistance"],
                    "context": [
                        {
                            "entity": "Security Cooperation",
                            "definition": "All activities undertaken by DoD to encourage international partners",
                            "section": "C1.1.1",
                            "confidence": 0.9
                        },
                        {
                            "entity": "Security Assistance", 
                            "definition": "Group of programs authorized under Title 22 authorities",
                            "section": "C1.1.2.2",
                            "confidence": 0.9
                        }
                    ]
                }
            }
        ]
        
        print(f"Testing with {len(test_scenarios)} scenarios...")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Test {i}: {scenario['query']} ---")
            
            try:
                answer = agent.generate_answer(
                    scenario["query"],
                    scenario["intent_info"], 
                    scenario["entity_info"]
                )
                
                # Check if it's an error answer
                if "Error" in answer and len(answer) < 100:
                    print(f"Error answer received: {answer}")
                    continue
                
                # Analyze the answer
                analysis = agent.get_answer_analysis(answer, scenario["intent_info"]["intent"])
                
                print(f"Answer length: {len(answer)} characters")
                print(f"Quality score: {analysis['quality_score']:.2f}")
                print(f"Has SAMM reference: {analysis['samm_compliance']['has_section_reference']}")
                print(f"Template adherence: {analysis['template_adherence'].get('adherence_score', 0):.2f}")
                print(f"Answer preview: {answer[:150]}...")
                
            except Exception as e:
                print(f"Error during test {i}: {e}")
        
        # Test agent status
        print(f"\n--- Agent Status ---")
        status = agent.get_agent_status()
        print(f"Agent type: {status['agent_type']}")
        print(f"Version: {status.get('version', 'Unknown')}")
        print(f"Supported intents: {status['response_capabilities']['supported_intents']}")
        print(f"Learning status: {status['learning_status']}")
        print(f"Windows compatibility: {status['generation_features'].get('windows_compatibility', False)}")
        
        # Test validation
        print(f"\n--- Validation Test ---")
        validation = agent.validate_answer_generation(test_scenarios)
        summary = validation['summary']
        print(f"Success rate: {summary['success_rate']:.2f}")
        print(f"Average quality score: {summary['average_quality_score']:.2f}")
        print(f"Section reference rate: {summary['section_reference_rate']:.2f}")
        print(f"Error count: {summary.get('error_count', 0)}")
        
    except Exception as e:
        print(f"Test initialization error: {e}")
        
    finally:
        # Cleanup
        if agent:
            try:
                agent.cleanup()
            except:
                pass

if __name__ == "__main__":
    test_enhanced_answer_agent()