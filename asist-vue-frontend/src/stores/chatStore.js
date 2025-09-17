import { defineStore } from 'pinia';
import { ref, computed, nextTick } from 'vue';
import { useAuthStore } from './authStore'; 
import { marked } from 'marked'; 

// LocalStorage Keys for Chat
const LS_CHAT_SESSIONS = 'asist_vue_chatSessions_v2'; 
const LS_ACTIVE_CHAT_ID = 'asist_vue_activeChatId_v2';

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + ['Bytes', 'KB', 'MB', 'GB', 'TB'][i];
};

export const useChatStore = defineStore('chat', () => {
  // --- State ---
  const chatSessions = ref({}); 
  // Structure: { sessionId: { name, messages: [], contextDocuments: [], caseId? }, ... }
  const activeSessionId = ref(null);
  const isAiThinking = ref(false);
  const chatError = ref(null);
  const userInitialsForChat = ref('ME'); 
  
  const fileUploadError = ref(null); 
  const isUploadingToSession = ref(false); 

  // --- Getters ---
  const currentSessionMessages = computed(() => {
    if (activeSessionId.value && chatSessions.value[activeSessionId.value]) {
      return chatSessions.value[activeSessionId.value].messages || [];
    }
    return [];
  });

  // Getter for the current session's context documents
  const currentSessionContextDocuments = computed(() => {
    if (activeSessionId.value && chatSessions.value[activeSessionId.value]) {
      return chatSessions.value[activeSessionId.value].contextDocuments || [];
    }
    return [];
  });

  const activeSessionName = computed(() => {
    if (activeSessionId.value && chatSessions.value[activeSessionId.value]) {
      return chatSessions.value[activeSessionId.value].name;
    }
    return "No Active Session";
  });


  // --- Actions ---
  function _scrollToBottom(containerRef) {
    nextTick(() => {
      if (containerRef && containerRef.value) {
        containerRef.value.scrollTop = containerRef.value.scrollHeight;
      }
    });
  }
  
  function loadSessions() {
    const storedSessions = localStorage.getItem(LS_CHAT_SESSIONS);
    if (storedSessions) {
      try {
        const parsedSessions = JSON.parse(storedSessions);
        for (const sessionId in parsedSessions) {
          if (!parsedSessions[sessionId].messages) {
            parsedSessions[sessionId].messages = [];
          }
          if (!parsedSessions[sessionId].contextDocuments) { // Ensure contextDocuments array exists
            parsedSessions[sessionId].contextDocuments = [];
          }
        }
        chatSessions.value = parsedSessions;
      } catch (e) {
        console.error("Error parsing chat sessions from localStorage:", e);
        chatSessions.value = {}; 
      }
    }
    const storedActiveId = localStorage.getItem(LS_ACTIVE_CHAT_ID);
    if (storedActiveId && chatSessions.value[storedActiveId]) {
      activeSessionId.value = storedActiveId;
    } else if (Object.keys(chatSessions.value).length > 0) {
      activeSessionId.value = Object.keys(chatSessions.value)[0]; 
    } else {
      createNewSession(true, null); 
    }
  }

  function saveSessions() {
    localStorage.setItem(LS_CHAT_SESSIONS, JSON.stringify(chatSessions.value));
    if (activeSessionId.value) {
      localStorage.setItem(LS_ACTIVE_CHAT_ID, activeSessionId.value);
    }
  }

  // message.sentContextDocuments: metadata of files active in session *at the time this user message was sent*
  // message.aiResponseAttachments: metadata of *new* files the AI might have "attached" or referenced in its response
  function addMessageToCurrentSession(sender, text, messageType = 'text', aiResponseAttachments = null, sentContextDocuments = null) {
    if (activeSessionId.value && chatSessions.value[activeSessionId.value]) {
      const session = chatSessions.value[activeSessionId.value];
      if (!session.messages) session.messages = [];
      
      const message = { 
        sender, text, type: messageType, 
        timestamp: new Date().toISOString() 
      };

      if (sender === 'user' && sentContextDocuments && sentContextDocuments.length > 0) {
        message.sentContextDocuments = sentContextDocuments.map(att => ({ ...att })); 
        console.log('[ChatStore] Storing sentContextDocuments on USER message:', JSON.parse(JSON.stringify(message.sentContextDocuments)));
      }
      if (sender === 'ai' && aiResponseAttachments && aiResponseAttachments.length > 0) {
        message.aiResponseAttachments = aiResponseAttachments.map(att => ({ ...att })); 
        console.log('[ChatStore] Storing aiResponseAttachments on AI message:', JSON.parse(JSON.stringify(message.aiResponseAttachments)));
      }
      session.messages.push(message);
      saveSessions(); 
    } else {
      console.error("ChatStore: No active session to add message to.");
      if (!activeSessionId.value && Object.keys(chatSessions.value).length === 0) {
        const newId = createNewSession(true, null); 
        if (newId && chatSessions.value[newId]) { 
             const session = chatSessions.value[newId];
             if (!session.messages) session.messages = [];
             const message = { sender, text, type: messageType, timestamp: new Date().toISOString() };
             if (sender === 'user' && sentContextDocuments && sentContextDocuments.length > 0) message.sentContextDocuments = sentContextDocuments.map(att => ({ ...att }));
             if (sender === 'ai' && aiResponseAttachments && aiResponseAttachments.length > 0) message.aiResponseAttachments = aiResponseAttachments.map(att => ({ ...att }));
             session.messages.push(message);
             saveSessions();
        }
      }
    }
  }
  
  function createNewSession(isDefault = false, associatedCaseId = null) {
    const now = new Date();
    const defaultName = `Session ${now.toLocaleDateString()} ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    let sessionName = isDefault ? defaultName : prompt('Enter new session name:', defaultName);

    if (!sessionName || sessionName.trim() === "") {
      if (!isDefault) { alert("Session name cannot be empty."); return null; }
      sessionName = defaultName; 
    }
    
    sessionName = sessionName.trim();
    const newId = `session_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    chatSessions.value[newId] = {
      name: sessionName,
      messages: [{ sender: 'ai', text: `New session "${sessionName}" started. How can I assist?`, type: 'text', timestamp: new Date().toISOString() }],
      contextDocuments: [], // Initialize with empty context documents
      caseId: associatedCaseId 
    };
    activeSessionId.value = newId;
    saveSessions();
    return newId;
  }

  function setActiveSession(sessionId) {
    if (sessionId === 'new-session-marker') {
      const newId = createNewSession(false, /* TODO: pass current caseId from component if needed */);
      return newId; 
    } else if (chatSessions.value[sessionId]) {
      activeSessionId.value = sessionId;
      // Ensure the session has contextDocuments array when activated
      if (!chatSessions.value[activeSessionId.value].contextDocuments) {
        chatSessions.value[activeSessionId.value].contextDocuments = [];
      }
      saveSessions(); // Save the new activeSessionId
      return sessionId;
    }
    return null;
  }

  function renameActiveSession(newName) { /* ... same as before ... */ }
  
  function deleteActiveSession() { 
    if (activeSessionId.value && chatSessions.value[activeSessionId.value]) {
      if (Object.keys(chatSessions.value).length <= 1) { alert("Cannot delete the last session."); return false; }
      if (confirm(`Are you sure you want to delete session: "${chatSessions.value[activeSessionId.value].name}"? This will also remove its associated chat documents.`)) {
        const docsToDelete = chatSessions.value[activeSessionId.value].contextDocuments || [];
        delete chatSessions.value[activeSessionId.value];
        activeSessionId.value = Object.keys(chatSessions.value)[0] || null; 
        saveSessions();
        if (!activeSessionId.value && Object.keys(chatSessions.value).length > 0) { 
            // This case should not be hit if the length check above works
        } else if (!activeSessionId.value) { // If all sessions were deleted
             createNewSession(true, null);
        }
        
        if (docsToDelete.length > 0) {
          console.log("[ChatStore] TODO: Implement backend deletion for these chat documents:", docsToDelete);
        }
        return true;
      }
    }
    return false;
  }

  // Action to upload a file and add its metadata to the active session's contextDocuments
  async function addFileToSessionContext(fileObject) {
    const authStore = useAuthStore();
    if (!authStore.isLoggedIn) {
      fileUploadError.value = "User not authenticated to upload file.";
      return null;
    }
    if (!fileObject) {
      fileUploadError.value = "No file provided to add to session.";
      return null;
    }
    if (!activeSessionId.value || !chatSessions.value[activeSessionId.value]) {
      fileUploadError.value = "No active chat session to add file to.";
      // Optionally create a session here if desired, or let UI handle it
      const newSessionId = createNewSession(true, null); // Create a default session
      if (!newSessionId) { // If session creation failed (e.g., user cancelled prompt)
          return null;
      }
      // The activeSessionId is now set
    }

    isUploadingToSession.value = true;
    fileUploadError.value = null; 
    const formData = new FormData();
    formData.append('document', fileObject, fileObject.name); 

    console.log(`[ChatStore] Uploading file to session context: ${fileObject.name}`);
    try {
      const response = await fetch("/api/chat/stage_attachment", { 
        method: 'POST',
        body: formData,
      });

      const responseData = await response.json();
      if (!response.ok) {
        throw new Error(responseData.error || `Failed to upload file to session: ${response.statusText}`);
      }
      
      const stagedMetadata = responseData.stagedDocument;
      console.log('[ChatStore] File uploaded to session context, metadata:', stagedMetadata);
      
      const session = chatSessions.value[activeSessionId.value];
      if (!session.contextDocuments) { // Ensure array exists
        session.contextDocuments = [];
      }
      // Prevent duplicates based on blobName or a unique ID from backend
      if (!session.contextDocuments.some(doc => doc.blobName === stagedMetadata.blobName)) {
        session.contextDocuments.push(stagedMetadata);
        saveSessions(); 
      } else {
        console.log('[ChatStore] Document already in session context:', stagedMetadata.fileName);
      }
      return stagedMetadata; 

    } catch (e) {
      console.error('[ChatStore] Error uploading file to session context:', e);
      fileUploadError.value = e.message;
      return null;
    } finally {
      isUploadingToSession.value = false;
    }
  }

  // MODIFIED: removeFileFromSessionContext to call backend
  async function removeFileFromSessionContext(documentId) {
    if (activeSessionId.value && chatSessions.value[activeSessionId.value]) {
      const session = chatSessions.value[activeSessionId.value];
      const docIndex = session.contextDocuments.findIndex(doc => doc.documentId === documentId);
      
      if (docIndex > -1) {
        const docToRemove = session.contextDocuments[docIndex];
        
        // Optimistically remove from UI
        session.contextDocuments.splice(docIndex, 1);
        saveSessions(); // Save local state change immediately
        console.log(`[ChatStore] Optimistically removed context document ${docToRemove.fileName} (ID: ${documentId}) from session.`);

        // Call backend to delete the actual file from Azure Blob Storage
        try {
          console.log(`[ChatStore] Calling backend to delete blob: ${docToRemove.blobName} in container ${docToRemove.blobContainer}`);
          const response = await fetch(`/api/chat/attachments/delete`, { 
            method: 'POST', 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
              blobName: docToRemove.blobName, 
              blobContainer: docToRemove.blobContainer 
            })
          });
          
          const responseData = await response.json();
          if (!response.ok) {
            console.error(`[ChatStore] Backend error deleting blob ${docToRemove.blobName}:`, responseData.error || response.statusText);
            // Optionally, re-add the document to the UI if backend deletion failed, or show an error
            // For simplicity now, we'll assume it's deleted from UI even if backend fails.
            // A more robust solution would handle this failure.
            fileUploadError.value = `Failed to delete '${docToRemove.fileName}' from storage. Please try again.`;
          } else {
            console.log(`[ChatStore] Successfully deleted blob ${docToRemove.blobName} from backend.`);
            fileUploadError.value = null; // Clear any previous error
          }
        } catch (e) { 
          console.error("[ChatStore] Network error calling backend to delete blob:", e);
          fileUploadError.value = `Network error deleting '${docToRemove.fileName}'. Please try again.`;
          // Re-add to list if network error? Or let user retry.
        }
        return true;
      }
    }
    return false;
  }

  // Helper function for deleteActiveSession to call (not exposed directly)
  async function deleteStagedChatFileFromBackend(blobName, blobContainer) {
    try {
        console.log(`[ChatStore] Calling backend to delete blob: ${blobName} in container ${blobContainer} during session delete.`);
        const response = await fetch(`/api/chat/attachments/delete`, { 
            method: 'POST', 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ blobName, blobContainer })
        });
        const responseData = await response.json();
        if (!response.ok) {
            console.error(`[ChatStore] Backend error deleting blob ${blobName} during session delete:`, responseData.error || response.statusText);
        } else {
            console.log(`[ChatStore] Successfully deleted blob ${blobName} from backend during session delete.`);
        }
    } catch (e) {
        console.error("[ChatStore] Network error calling backend to delete blob during session delete:", e);
    }
  }

  async function sendMessageToBackend(messageText, currentChatContainerRef = null) {
    const authStore = useAuthStore();
    if (!authStore.isLoggedIn) { /* ... */ return; }
    if (!activeSessionId.value || !chatSessions.value[activeSessionId.value]) { /* ... */ return; }

    const session = chatSessions.value[activeSessionId.value];
    const contextDocsForThisMessage = session.contextDocuments ? [...session.contextDocuments] : []; // Get current context

    console.log('[ChatStore] sendMessageToBackend called.');
    console.log('[ChatStore] Message text:', messageText);
    console.log('[ChatStore] Context documents for this message:', JSON.parse(JSON.stringify(contextDocsForThisMessage)));

    isAiThinking.value = true;
    chatError.value = null;

    let userMessageContent = messageText;
    if (contextDocsForThisMessage.length > 0) {
        const fileNames = contextDocsForThisMessage.map(f => f.fileName).join(', ');
        userMessageContent = messageText ? `${messageText} (Context: ${fileNames})` : `(Context: ${fileNames})`;
    }
    // Add user message, also passing the context docs that were active for this specific message
    addMessageToCurrentSession('user', userMessageContent, 'text', null, contextDocsForThisMessage); 
    _scrollToBottom(currentChatContainerRef);

    const requestPayload = {
        question: messageText,
        chat_history: [], 
        staged_chat_documents: contextDocsForThisMessage // Send metadata of files already in blob
    };
    
    if (session.messages) {
      const messagesForHistory = session.messages.slice(0, -1); 
      for (let i = 0; i < messagesForHistory.length; i++) {
        if (messagesForHistory[i].sender === 'user') {
          const userInputHist = messagesForHistory[i].text;
          const aiOutputHist = (messagesForHistory[i+1] && messagesForHistory[i+1].sender === 'ai') ? messagesForHistory[i+1].text : "";
          // Include sentContextDocuments in history if available
          const historyEntry = { inputs: { question: userInputHist }, outputs: { answer: aiOutputHist } };
          if (messagesForHistory[i].sentContextDocuments) {
            historyEntry.inputs.staged_chat_documents = messagesForHistory[i].sentContextDocuments;
          }
          requestPayload.chat_history.push(historyEntry);
          if (messagesForHistory[i+1] && messagesForHistory[i+1].sender === 'ai') i++; 
        }
      }
    }

    try {
      const response = await fetch("/api/query", { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload) 
      });

      if (!response.ok) { /* ... error handling ... */ }

      const data = await response.json();
      const botText = data.response?.answer || data.answer || 'Sorry, I could not process that.';
      const aiResponseAttachments = data.uploadedChatDocuments || []; 

      addMessageToCurrentSession('ai', botText, 'text', aiResponseAttachments);
      
      // Context documents are NOT cleared here, they persist for the session.
      
      if (aiResponseAttachments.length > 0) { /* ... log ... */ }

    } catch (e) {
      console.error('[ChatStore] Error sending message to AI:', e);
      chatError.value = e.message;
      addMessageToCurrentSession('ai', `Error: ${e.message}. Please try again.`);
    } finally {
      isAiThinking.value = false;
      _scrollToBottom(currentChatContainerRef);
    }
  }

  loadSessions();

  return { 
    chatSessions, activeSessionId, isAiThinking, chatError, 
    fileUploadError, isUploadingToSession, 
    userInitialsForChat, currentSessionMessages, currentSessionContextDocuments, 
    activeSessionName,
    setActiveSession, createNewSession, renameActiveSession, deleteActiveSession,
    addFileToSessionContext, removeFileFromSessionContext, 
    sendMessageToBackend, addMessageToCurrentSession, 
    _scrollToBottom 
  };
});
