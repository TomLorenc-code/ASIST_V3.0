<template>
  <aside class="chat-interface-panel">
    <div class="section-title">
      <h3>AI Assistant</h3>
    </div>
    
    <div class="session-selector">
      <div class="session-dropdown">
        <select 
          id="sessionSelectVue" 
          :value="chatStore.activeSessionId" 
          @change="handleSessionSelectionChange" 
          title="Select Chat Session"
        >
          <option 
            v-for="(session, id) in chatStore.chatSessions" 
            :key="id" 
            :value="id"
          >
            {{ session.name }}
          </option>
          <option value="new-session-marker">+ New Session</option>
        </select>
      </div>
      <div class="session-controls">
        <button 
          class="session-btn" 
          @click="triggerRenameSession" 
          title="Rename Session" 
          :disabled="!chatStore.activeSessionId || chatStore.activeSessionId === 'new-session-marker'"
        >
          <span class="nav-icon"><i class="fas fa-pencil-alt"></i></span> Rename
        </button>
        <button 
          class="session-btn" 
          @click="triggerDeleteSession" 
          title="Delete Session" 
          :disabled="!chatStore.activeSessionId || chatStore.activeSessionId === 'new-session-marker' || Object.keys(chatStore.chatSessions).length <= 1"
        >
          <span class="nav-icon"><i class="fas fa-trash-alt"></i></span> Delete
        </button>
      </div>
    </div>
    
    <div class="chat-messages" ref="chatMessagesContainerRef">
      <div v-for="(message, index) in chatStore.currentSessionMessages" :key="index" 
           class="message" :class="message.sender === 'user' ? 'user-message' : 'ai-message'">
        <div class="message-avatar">{{ message.sender === 'user' ? chatStore.userInitialsForChat : 'AI' }}</div>
        <div class="message-content">
          <div v-html="renderMarkdown(message.text)"></div>
          <div v-if="message.sender === 'user' && message.sentContextDocuments && message.sentContextDocuments.length > 0" class="message-attachments sent-context">
            <strong>Sent with Context:</strong>
            <ul>
              <li v-for="att in message.sentContextDocuments" :key="att.documentId || att.fileName">
                <i class="fas fa-paperclip attachment-icon"></i> {{ att.fileName }}
              </li>
            </ul>
          </div>
          <div v-if="message.sender === 'ai' && message.aiResponseAttachments && message.aiResponseAttachments.length > 0" class="message-attachments ai-response-attachments">
            <strong>Referenced Attachments:</strong>
            <ul>
              <li v-for="att in message.aiResponseAttachments" :key="att.documentId || att.fileName">
                <a :href="att.url" target="_blank"><i class="fas fa-link attachment-icon"></i> {{ att.fileName }}</a> 
                <span v-if="att.sizeBytes">({{ formatFileSize(att.sizeBytes) }})</span>
              </li>
            </ul>
          </div>
          <!-- HIL Feedback Button for AI Messages -->
          <div v-if="message.sender === 'ai'" class="message-actions">
            <button 
              class="hil-feedback-btn" 
              @click="openHilFeedback(message, index)"
              title="Provide feedback to improve AI responses"
            >
              <i class="fas fa-edit"></i> Provide Feedback
            </button>
          </div>
        </div>
      </div>
      <div v-if="chatStore.isAiThinking" class="message ai-message">
        <div class="message-avatar">AI</div>
        <div class="message-content"><p>Thinking...</p></div>
      </div>
       <div v-if="chatStore.chatError" class="message ai-message error-message-display">
        <div class="message-avatar">AI</div>
        <div class="message-content"><p>Error: {{ chatStore.chatError }}</p></div>
      </div>
    </div>
    
    <div class="chat-input-area">
      <div class="chat-input">
        <input 
          type="text" 
          class="ai-chat-input" 
          placeholder="Ask your AI assistant..." 
          title="Type your message"
          v-model="userInput"
          @keypress.enter.exact.prevent="submitMessage"
          :disabled="chatStore.isUploadingToSession" 
        >
        <button @click="submitMessage" title="Send Message" 
                :disabled="chatStore.isAiThinking || chatStore.isUploadingToSession || !userInput.trim()">
          <i class="fas fa-paper-plane"></i>
        </button>
      </div>
      
      <!-- Human-in-the-Loop Feedback Section -->
      <div v-if="showHilFeedback" class="hil-feedback-section">
        <div class="hil-header">
          <h4><i class="fas fa-user-edit"></i> Human-in-the-Loop Feedback</h4>
          <button class="close-hil-btn" @click="closeHilFeedback" title="Close feedback">
            <i class="fas fa-times"></i>
          </button>
        </div>
        
        <div class="hil-content">
          <div class="original-response">
            <label>Original AI Response:</label>
            <div class="response-preview">{{ selectedMessage?.text || 'No message selected' }}</div>
          </div>
          
          <div class="feedback-tabs">
            <button 
              class="tab-btn" 
              :class="{ active: activeHilTab === 'intent' }"
              @click="activeHilTab = 'intent'"
            >
              Intent Correction
            </button>
            <button 
              class="tab-btn" 
              :class="{ active: activeHilTab === 'entities' }"
              @click="activeHilTab = 'entities'"
            >
              Entity Correction
            </button>
            <button 
              class="tab-btn" 
              :class="{ active: activeHilTab === 'answer' }"
              @click="activeHilTab = 'answer'"
            >
              Answer Correction
            </button>
          </div>
          
          <!-- Intent Correction Tab -->
          <div v-if="activeHilTab === 'intent'" class="feedback-tab-content">
            <div class="form-group">
              <label>Original Intent:</label>
              <input v-model="hilFeedback.intent.original" placeholder="e.g., definition" />
            </div>
            <div class="form-group">
              <label>Corrected Intent:</label>
              <select v-model="hilFeedback.intent.corrected">
                <option value="">Select correct intent</option>
                <option value="definition">Definition</option>
                <option value="distinction">Distinction</option>
                <option value="authority">Authority</option>
                <option value="organization">Organization</option>
                <option value="factual">Factual</option>
                <option value="relationship">Relationship</option>
                <option value="general">General</option>
              </select>
            </div>
            <div class="form-group">
              <label>Feedback Notes:</label>
              <textarea v-model="hilFeedback.intent.notes" placeholder="Why was the intent classification incorrect?"></textarea>
            </div>
          </div>
          
          <!-- Entity Correction Tab -->
          <div v-if="activeHilTab === 'entities'" class="feedback-tab-content">
            <div class="form-group">
              <label>Original Entities (comma-separated):</label>
              <input v-model="hilFeedback.entities.original" placeholder="e.g., DSCA, Security Cooperation" />
            </div>
            <div class="form-group">
              <label>Corrected Entities (comma-separated):</label>
              <input v-model="hilFeedback.entities.corrected" placeholder="e.g., DSCA, Defense Security Cooperation Agency, DoD" />
            </div>
            <div class="form-group">
              <label>Missing Entities:</label>
              <input v-model="hilFeedback.entities.missing" placeholder="Entities that should have been found" />
            </div>
            <div class="form-group">
              <label>Entity Definitions (JSON format):</label>
              <textarea v-model="hilFeedback.entities.definitions" placeholder='{"New Entity": "Definition of the new entity"}'></textarea>
            </div>
          </div>
          
          <!-- Answer Correction Tab -->
          <div v-if="activeHilTab === 'answer'" class="feedback-tab-content">
            <div class="form-group">
              <label>Corrected Answer:</label>
              <textarea v-model="hilFeedback.answer.corrected" placeholder="Provide the correct or improved answer" rows="4"></textarea>
            </div>
            <div class="form-group">
              <label>Improvement Notes:</label>
              <textarea v-model="hilFeedback.answer.notes" placeholder="What was wrong or could be improved?" rows="2"></textarea>
            </div>
            <div class="form-group">
              <label>Key Points to Include:</label>
              <input v-model="hilFeedback.answer.keyPoints" placeholder="e.g., section references, acronym expansions" />
            </div>
          </div>
          
          <div class="hil-actions">
            <button class="submit-feedback-btn" @click="submitHilFeedback" :disabled="isSubmittingHil">
              <i class="fas fa-paper-plane"></i> 
              {{ isSubmittingHil ? 'Submitting...' : 'Submit Feedback' }}
            </button>
            <button class="clear-feedback-btn" @click="clearHilFeedback">
              <i class="fas fa-eraser"></i> Clear
            </button>
          </div>
        </div>
      </div>
      
      <div v-if="chatStore.fileUploadError" class="upload-error-display">
        <small>File staging error: {{ chatStore.fileUploadError }}</small>
      </div>
      
      <div v-if="hilError" class="hil-error-display">
        <small>HIL Feedback Error: {{ hilError }}</small>
      </div>
      
      <div v-if="hilSuccess" class="hil-success-display">
        <small>{{ hilSuccess }}</small>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { useChatStore } from '@/stores/chatStore'; 
import { marked } from 'marked'; 

const props = defineProps({
  activeCaseId: {
    type: String,
    default: null
  }
});

const chatStore = useChatStore();

const userInput = ref('');
const chatMessagesContainerRef = ref(null);

// HIL Feedback State
const showHilFeedback = ref(false);
const activeHilTab = ref('intent');
const selectedMessage = ref(null);
const selectedMessageIndex = ref(-1);
const isSubmittingHil = ref(false);
const hilError = ref('');
const hilSuccess = ref('');

const hilFeedback = ref({
  intent: {
    original: '',
    corrected: '',
    notes: ''
  },
  entities: {
    original: '',
    corrected: '',
    missing: '',
    definitions: ''
  },
  answer: {
    corrected: '',
    notes: '',
    keyPoints: ''
  }
});

const renderMarkdown = (text) => {
  if (typeof text === 'string') {
    return marked.parse(text);
  }
  return '';
};

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + ['Bytes', 'KB', 'MB', 'GB', 'TB'][i];
};

const handleSessionSelectionChange = (event) => {
  const selectedId = event.target.value;
  if (selectedId === 'new-session-marker') {
    chatStore.createNewSession(false, props.activeCaseId); 
  } else {
    chatStore.setActiveSession(selectedId);
  }
};

const triggerRenameSession = () => {
  if (chatStore.activeSessionId && chatStore.activeSessionId !== 'new-session-marker') {
    const currentName = chatStore.chatSessions[chatStore.activeSessionId]?.name || '';
    const newName = prompt('Enter new name for this session:', currentName);
    if (newName) { 
        chatStore.renameActiveSession(newName);
    }
  }
};

const triggerDeleteSession = () => {
  chatStore.deleteActiveSession();
};

const submitMessage = async () => {
  const textToSend = userInput.value.trim();
  
  if (!textToSend && chatStore.currentSessionContextDocuments.length === 0) {
    if (chatStore.isUploadingToSession) {
        alert("A file is currently being added to the session context. Please wait.");
    } else {
        alert("Please type a message or add files to the session context via the Documents tab.");
    }
    return;
  }
  if (chatStore.isUploadingToSession) { 
      alert("A file is currently being added to the session context. Please wait.");
      return;
  }
  
  console.log('[ChatInterface] Submitting message. Text:', textToSend);
  await chatStore.sendMessageToBackend(textToSend, chatMessagesContainerRef); 
  
  userInput.value = ''; 
};

// HIL Feedback Functions
const openHilFeedback = (message, index) => {
  selectedMessage.value = message;
  selectedMessageIndex.value = index;
  showHilFeedback.value = true;
  activeHilTab.value = 'intent';
  
  // Pre-populate with metadata if available
  const metadata = message.metadata || {};
  hilFeedback.value.intent.original = metadata.intent || '';
  
  if (metadata.entities && Array.isArray(metadata.entities)) {
    hilFeedback.value.entities.original = metadata.entities.join(', ');
  }
  
  clearMessages();
  console.log('[HIL] Opened feedback for message:', index);
};

const closeHilFeedback = () => {
  showHilFeedback.value = false;
  selectedMessage.value = null;
  selectedMessageIndex.value = -1;
  clearHilFeedback();
  clearMessages();
};

const clearHilFeedback = () => {
  hilFeedback.value = {
    intent: { original: '', corrected: '', notes: '' },
    entities: { original: '', corrected: '', missing: '', definitions: '' },
    answer: { corrected: '', notes: '', keyPoints: '' }
  };
};

const clearMessages = () => {
  hilError.value = '';
  hilSuccess.value = '';
};

const submitHilFeedback = async () => {
  if (!selectedMessage.value) {
    hilError.value = 'No message selected for feedback';
    return;
  }
  
  isSubmittingHil.value = true;
  clearMessages();
  
  try {
    // Get the user's original query from the messages
    const userMessage = chatStore.currentSessionMessages[selectedMessageIndex.value - 1];
    const query = userMessage?.text || '';
    
    if (!query) {
      throw new Error('Could not find original user query');
    }
    
    // Prepare HIL update payload
    const hilPayload = {
      query: query
    };
    
    // Add intent correction if provided
    if (hilFeedback.value.intent.corrected) {
      hilPayload.intent_correction = {
        original_intent: hilFeedback.value.intent.original,
        corrected_intent: hilFeedback.value.intent.corrected,
        feedback_data: {
          notes: hilFeedback.value.intent.notes,
          user_feedback: true
        }
      };
    }
    
    // Add entity correction if provided
    if (hilFeedback.value.entities.corrected) {
      const originalEntities = hilFeedback.value.entities.original.split(',').map(e => e.trim()).filter(e => e);
      const correctedEntities = hilFeedback.value.entities.corrected.split(',').map(e => e.trim()).filter(e => e);
      
      let entityDefinitions = {};
      if (hilFeedback.value.entities.definitions) {
        try {
          entityDefinitions = JSON.parse(hilFeedback.value.entities.definitions);
        } catch (e) {
          console.warn('Invalid JSON in entity definitions, ignoring');
        }
      }
      
      hilPayload.entity_correction = {
        original_entities: originalEntities,
        corrected_entities: correctedEntities,
        feedback_data: {
          missing_entities: hilFeedback.value.entities.missing.split(',').map(e => e.trim()).filter(e => e),
          entity_definitions: entityDefinitions,
          user_feedback: true
        }
      };
    }
    
    // Add answer correction if provided
    if (hilFeedback.value.answer.corrected) {
      hilPayload.answer_correction = {
        original_answer: selectedMessage.value.text,
        corrected_answer: hilFeedback.value.answer.corrected,
        feedback_data: {
          improvement_notes: hilFeedback.value.answer.notes,
          key_points: hilFeedback.value.answer.keyPoints.split(',').map(p => p.trim()).filter(p => p),
          user_feedback: true,
          intent: hilFeedback.value.intent.corrected || hilFeedback.value.intent.original,
          entities: hilFeedback.value.entities.corrected.split(',').map(e => e.trim()).filter(e => e)
        }
      };
    }
    
    console.log('[HIL] Submitting feedback payload:', hilPayload);
    
    // Submit to HIL endpoint
    const response = await fetch('/api/agents/hil_update', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(hilPayload)
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP ${response.status}`);
    }
    
    const result = await response.json();
    console.log('[HIL] Feedback submitted successfully:', result);
    
    hilSuccess.value = 'Feedback submitted successfully! The AI will learn from your corrections.';
    
    // Close feedback after short delay
    setTimeout(() => {
      closeHilFeedback();
    }, 2000);
    
  } catch (error) {
    console.error('[HIL] Error submitting feedback:', error);
    hilError.value = `Failed to submit feedback: ${error.message}`;
  } finally {
    isSubmittingHil.value = false;
  }
};

onMounted(() => {
  chatStore._scrollToBottom(chatMessagesContainerRef); 
  console.log('[ChatInterface] Mounted. Active Case ID from prop:', props.activeCaseId);
});

watch(() => chatStore.currentSessionMessages, () => {
  chatStore._scrollToBottom(chatMessagesContainerRef);
}, { deep: true });

watch(() => chatStore.activeSessionId, (newId, oldId) => {
    chatStore._scrollToBottom(chatMessagesContainerRef);
    console.log(`[ChatInterface] Active session ID changed from ${oldId} to ${newId}`);
    // Close HIL feedback when switching sessions
    if (showHilFeedback.value) {
      closeHilFeedback();
    }
});

</script>

<style scoped>
/* Existing styles... */
.chat-interface-panel {
  flex: 1; 
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  padding: var(--space-md);
  display: flex;
  flex-direction: column;
  overflow: hidden; 
}

.section-title {
  margin-bottom: var(--space-sm);
  padding-bottom: var(--space-xs);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.section-title h3 {
  font-size: 1rem;
  color: var(--primary);
  margin: 0;
}

.session-selector {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-sm);
  padding-bottom: var(--space-sm);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.session-dropdown { flex: 1; margin-right: var(--space-sm); }
.session-dropdown select {
  width: 100%;
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  font-size: 0.85rem;
  background-color: white;
}
.session-controls { display: flex; gap: var(--space-xs); }
.session-btn {
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: white;
  cursor: pointer;
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  color: var(--text-dark);
}
.session-btn:hover { background-color: #f5f7fa; }
.session-btn .nav-icon { font-size: 0.8em; color: var(--text-dark); margin-right: 3px;}
.session-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.chat-messages { 
  flex-grow: 1;
  overflow-y: auto; 
  padding: var(--space-sm) 0; 
  margin-bottom: var(--space-sm);
}

.message { display: flex; margin-bottom: var(--space-sm); max-width: 95%; }
.message-avatar {
  width: 28px;
  height: 28px; 
  border-radius: 50%; 
  margin-right: var(--space-sm);
  display: flex; 
  align-items: center; 
  justify-content: center;
  font-weight: bold; 
  font-size: 0.75rem;
  flex-shrink: 0;
  color: white;
}
.user-message { align-self: flex-end; flex-direction: row-reverse; }
.user-message .message-avatar { background-color: var(--primary); margin-right: 0; margin-left: var(--space-sm); }
.ai-message .message-avatar { background-color: var(--accent); }

.message-content {
  background-color: #f1f1f1; 
  padding: var(--space-xs) var(--space-sm);
  border-radius: 15px; 
  max-width: calc(100% - 40px); 
  font-size: 0.9rem;
  word-wrap: break-word;
  line-height: 1.4;
}
.user-message .message-content { background-color: var(--accent); color: white; border-top-right-radius: 4px; }
.ai-message .message-content { background-color: #e9ecef; color: var(--text-dark); border-top-left-radius: 4px;}
.error-message-display .message-content {
    background-color: var(--danger-soft-bg, #ffebee); 
    color: var(--danger);
    border: 1px solid var(--danger);
}

/* Message Actions for HIL Feedback */
.message-actions {
  margin-top: var(--space-xs);
  padding-top: var(--space-xs);
  border-top: 1px dashed #ccc;
}

.hil-feedback-btn {
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  border: none;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 4px;
}

.hil-feedback-btn:hover {
  background: linear-gradient(135deg, #2980b9, #1f5f8b);
  transform: translateY(-1px);
}

.hil-feedback-btn i {
  font-size: 0.7rem;
}

/* Styles for displaying attachments within a message bubble */
.message-attachments {
  margin-top: var(--space-xs);
  font-size: 0.8rem;
  opacity: 0.9;
  border-top: 1px dashed var(--border);
  padding-top: var(--space-xs);
  margin-left: -5px;
  margin-right: -5px;
}
.message-attachments.sent-context {
  opacity: 0.7;
}
.message-attachments strong {
  display: block;
  margin-bottom: 3px;
  font-weight: 500;
  font-size: 0.75rem;
}
.message-attachments ul {
  list-style: none;
  padding-left: 0;
  margin: 0;
}
.message-attachments li {
  margin-bottom: 2px;
  display: flex;
  align-items: center;
  font-size: 0.75rem;
}
.message-attachments .attachment-icon {
    margin-right: var(--space-xs);
    font-size: 0.9em;
}
.message-attachments a {
  color: var(--accent); 
  text-decoration: none;
}
.message-attachments a:hover {
  text-decoration: underline;
}

.message-content :deep(p) { margin-bottom: var(--space-xs); }
.message-content :deep(ol),
.message-content :deep(ul) { padding-left: var(--space-md); margin-bottom: var(--space-xs); }
.message-content :deep(a):not(.message-attachments a) {
    color: var(--primary); text-decoration: underline; 
}

.message-content :deep(code) { 
  background-color: rgba(0,0,0,0.05);
  padding: 0.15em 0.4em;
  border-radius: 3px;
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.85em;
  color: var(--text-dark);
}

.message-content :deep(pre) { 
  background-color: rgba(0,0,0,0.07);
  padding: var(--space-sm);
  border-radius: 4px;
  overflow-x: auto;
  margin-bottom: var(--space-xs);
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.85em;
  line-height: 1.4;
  color: var(--text-dark);
}

.message-content :deep(pre code) { 
  background-color: transparent;
  padding: 0;
  border-radius: 0;
  font-size: 1em;
}

.chat-input-area {
  margin-top: auto; 
  padding-top: var(--space-sm); 
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}

.chat-input {
  display: flex; 
  align-items: center;
  border: 1px solid var(--border); 
  border-radius: 20px;
  padding: 5px var(--space-sm);
  background-color: white;
}
.chat-input input { 
  flex: 1; 
  border: none; 
  outline: none; 
  padding: var(--space-xs) 0; 
  font-size: 0.9rem; 
  background: transparent;
  color: var(--text-dark);
}
.chat-input button {
  background-color: var(--accent); 
  color: white; 
  border: none; 
  border-radius: 50%;
  width: 32px;
  height: 32px; 
  display: flex; 
  align-items: center; 
  justify-content: center; 
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.2s;
}
.chat-input button:hover {
    background-color: #2980b9; 
}
.chat-input button:disabled {
    background-color: var(--border);
    cursor: not-allowed;
}

/* HIL Feedback Section Styles */
.hil-feedback-section {
  margin-top: var(--space-sm);
  border: 2px solid #3498db;
  border-radius: 8px;
  background: linear-gradient(135deg, #f8fafe, #e3f2fd);
  padding: var(--space-sm);
  animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.hil-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-sm);
  padding-bottom: var(--space-xs);
  border-bottom: 1px solid #bbb;
}

.hil-header h4 {
  margin: 0;
  color: #2c3e50;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: var(--space-xs);
}

.hil-header h4 i {
  color: #3498db;
}

.close-hil-btn {
  background: none;
  border: none;
  color: #7f8c8d;
  cursor: pointer;
  font-size: 1rem;
  padding: 4px;
  border-radius: 4px;
  transition: all 0.2s;
}

.close-hil-btn:hover {
  background-color: #ecf0f1;
  color: #e74c3c;
}

.original-response {
  margin-bottom: var(--space-sm);
}

.original-response label {
  font-weight: 600;
  font-size: 0.8rem;
  color: #2c3e50;
  display: block;
  margin-bottom: 4px;
}

.response-preview {
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: var(--space-xs);
  font-size: 0.8rem;
  max-height: 80px;
  overflow-y: auto;
  color: #34495e;
}

.feedback-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: var(--space-sm);
}

.tab-btn {
  background: white;
  border: 1px solid #bdc3c7;
  border-radius: 4px 4px 0 0;
  padding: 6px 12px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
  color: #7f8c8d;
}

.tab-btn.active {
  background: #3498db;
  color: white;
  border-color: #3498db;
}

.tab-btn:hover:not(.active) {
  background: #ecf0f1;
}

.feedback-tab-content {
  background: white;
  border: 1px solid #bdc3c7;
  border-radius: 0 4px 4px 4px;
  padding: var(--space-sm);
}

.form-group {
  margin-bottom: var(--space-sm);
}

.form-group:last-child {
  margin-bottom: 0;
}

.form-group label {
  display: block;
  font-weight: 600;
  font-size: 0.8rem;
  color: #2c3e50;
  margin-bottom: 4px;
}

.form-group input,
.form-group select,
.form-group textarea {
  width: 100%;
  padding: 6px 8px;
  border: 1px solid #bdc3c7;
  border-radius: 4px;
  font-size: 0.8rem;
  font-family: inherit;
  transition: border-color 0.2s;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #3498db;
}

.form-group textarea {
  resize: vertical;
  min-height: 60px;
}

.hil-actions {
  margin-top: var(--space-sm);
  display: flex;
  gap: var(--space-xs);
}

.submit-feedback-btn {
  background: linear-gradient(135deg, #27ae60, #229954);
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: var(--space-xs);
}

.submit-feedback-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #229954, #1e8449);
  transform: translateY(-1px);
}

.submit-feedback-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.clear-feedback-btn {
  background: #95a5a6;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: var(--space-xs);
}

.clear-feedback-btn:hover {
  background: #7f8c8d;
  transform: translateY(-1px);
}

/* Error and Success Messages */
.upload-error-display,
.hil-error-display {
  font-size: 0.8rem;
  color: #e74c3c;
  margin-top: var(--space-xs);
  text-align: left;
  padding-left: var(--space-sm);
}

.hil-success-display {
  font-size: 0.8rem;
  color: #27ae60;
  margin-top: var(--space-xs);
  text-align: left;
  padding-left: var(--space-sm);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .feedback-tabs {
    flex-direction: column;
  }
  
  .tab-btn {
    border-radius: 4px;
    margin-bottom: 2px;
  }
  
  .feedback-tab-content {
    border-radius: 4px;
  }
  
  .hil-actions {
    flex-direction: column;
  }
}
</style>