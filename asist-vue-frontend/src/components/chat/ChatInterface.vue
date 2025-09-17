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
       <div v-if="chatStore.fileUploadError" class="upload-error-display"> <small>File staging error: {{ chatStore.fileUploadError }}</small>
        </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { useChatStore } from '@/stores/chatStore'; 
import { marked } from 'marked'; 

const props = defineProps({
  activeCaseId: { // This prop is passed from MainDashboardLayout
    type: String,
    default: null
  }
});

const chatStore = useChatStore();

const userInput = ref('');
const chatMessagesContainerRef = ref(null); 

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
    // Pass the current activeCaseId when creating a new session
    chatStore.createNewSession(false, props.activeCaseId); 
    // The store's createNewSession will set itself as active.
    // We might need to ensure the select dropdown updates if the store logic doesn't immediately reflect activeSessionId
    // For now, this relies on the store setting activeSessionId correctly.
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
  
  // The chatStore.sendMessageToBackend will now get context documents from the active session in the store
  if (!textToSend && chatStore.currentSessionContextDocuments.length === 0) {
    if (chatStore.isUploadingToSession) { // Check store's staging status
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

// File input related refs and methods are removed as staging is now in DocumentsContent.vue

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
    // If a new session was just created and it's now active,
    // ensure its caseId is set if props.activeCaseId is available
    if (newId && chatStore.chatSessions[newId] && !chatStore.chatSessions[newId].caseId && props.activeCaseId) {
        // This is a bit of a workaround. Ideally, createNewSession should robustly get the caseId.
        // chatStore.chatSessions[newId].caseId = props.activeCaseId;
        // chatStore.saveSessions(); // If you modify it directly
        console.log(`[ChatInterface] New session ${newId} might need its caseId associated.`);
    }
});

</script>

<style scoped>
/* Styles from vue_chat_interface_vue_with_store artifact */
/* Ensure all previous styles are present */
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

/* Styles for displaying attachments within a message bubble */
.message-attachments {
  margin-top: var(--space-xs);
  font-size: 0.8rem;
  opacity: 0.9;
  border-top: 1px dashed var(--border);
  padding-top: var(--space-xs);
  margin-left: -5px; /* Align with content above */
  margin-right: -5px;
}
.message-attachments.sent-context { /* Slightly different style for user's sent context */
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
.message-content :deep(a):not(.message-attachments a) { /* Avoid double styling links */
    color: var(--primary); text-decoration: underline; 
}

.message-content :deep(code) { 
  background-color: rgba(0,0,0,0.05); /* Light grey background */
  padding: 0.15em 0.4em; /* Small padding */
  border-radius: 3px;
  font-family: 'Courier New', Courier, monospace; /* Monospace font */
  font-size: 0.85em; /* Slightly smaller than surrounding text */
  color: var(--text-dark); /* Or a specific color for code */
}

/* Styling for code blocks (often wrapped in <pre><code>...</code></pre>) */
.message-content :deep(pre) { 
  background-color: rgba(0,0,0,0.07); /* Slightly darker background for blocks */
  padding: var(--space-sm); /* More padding for blocks */
  border-radius: 4px;
  overflow-x: auto; /* Allow horizontal scrolling for long lines */
  margin-bottom: var(--space-xs);
  font-family: 'Courier New', Courier, monospace; /* Monospace font */
  font-size: 0.85em;
  line-height: 1.4; /* Improve readability of multi-line code */
  color: var(--text-dark);
}

/* Styling for the <code> tag specifically inside a <pre> tag */
/* This is often used to reset any inline code styling if not desired for blocks */
.message-content :deep(pre code) { 
  background-color: transparent; /* No separate background from pre */
  padding: 0; /* No extra padding, pre handles it */
  border-radius: 0;
  font-size: 1em; /* Inherit font-size from pre or set to match */
  /* color can be inherited from pre or set specifically */
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
.upload-error-display { /* For staging errors from chatStore */
    font-size: 0.8rem;
    color: var(--danger);
    margin-top: var(--space-xs);
    text-align: left; /* Or center */
    padding-left: var(--space-sm);
}
</style>
