<template>
  <div id="asist-app-container">
    <div v-if="authStore.authIsLoading" class="loading-container">
      <p>Loading application...</p>
    </div>

    <template v-else-if="authStore.isLoggedIn">
      <TheSidebar 
        :is-collapsed="isSidebarCollapsed" 
        :current-active-case-id="caseStore.activeCaseDetails?.id || caseStore.activeCaseDetails?.caseNumber"
        :user-profile="authStore.userProfile"
        :user-initials="authStore.userInitials"
        @toggle-sidebar="toggleSidebar" 
        @open-settings="handleOpenSettings"
        @logout-user="handleLogout"
      />
      <RouterView 
        :is-sidebar-collapsed="isSidebarCollapsed" 
      />
    </template>

    <div v-else class="login-prompt-container">
      <h1>Welcome to ASIST</h1>
      <p>Please log in to continue.</p>
      <button @click="handleLogin" class="login-button">Log In with Auth0</button>
      <p v-if="authStore.authError" class="error-message">
        Authentication error: {{ authStore.authError }}
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'; 
import { RouterView } from 'vue-router'; 
import TheSidebar from './components/layout/TheSidebar.vue';
import { useAuthStore } from './stores/authStore';
import { useCaseStore } from './stores/caseStore'; 

// LocalStorage Helper for sidebar state
const LS_SIDEBAR_COLLAPSED = 'asist_vue_sidebarCollapsed';

function saveSidebarStateToLS(value) {
    try {
        localStorage.setItem(LS_SIDEBAR_COLLAPSED, JSON.stringify(value));
    } catch (e) {
        console.error("Error saving sidebar state to localStorage:", e);
    }
}

function loadSidebarStateFromLS(defaultValue = false) {
    try {
        const storedValue = localStorage.getItem(LS_SIDEBAR_COLLAPSED);
        return storedValue ? JSON.parse(storedValue) : defaultValue;
    } catch (e) {
        console.error("Error loading sidebar state from localStorage:", e);
        return defaultValue;
    }
}

const authStore = useAuthStore();
const caseStore = useCaseStore();

const isSidebarCollapsed = ref(loadSidebarStateFromLS(false));

const toggleSidebar = () => {
  isSidebarCollapsed.value = !isSidebarCollapsed.value;
  saveSidebarStateToLS(isSidebarCollapsed.value);
};

const handleOpenSettings = () => {
  console.log('[App.vue] Open settings triggered');
  alert("Settings would open here.");
};

const handleLogin = () => {
  authStore.login(); 
};

const handleLogout = () => {
  caseStore.setActiveCase(null); 
  authStore.logout(); 
};

onMounted(async () => {
  await authStore.fetchUser(); 
  if (authStore.isLoggedIn) {
    console.log('[App.vue] User is logged in, fetching cases...');
    await caseStore.fetchUserCases(); 
  } else {
    console.log('[App.vue] User not logged in. Cases will not be fetched.');
  }
});
</script>

<style>
/* Global styles */
#asist-app-container {
  display: flex;
  min-height: 100vh;
  background-color: #f5f7fa; 
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
  color: #2c3e50; 
  line-height: 1.6;
}

.loading-container, .login-prompt-container {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  text-align: center;
}

.login-button {
  background-color: var(--accent); 
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  font-size: 1rem;
  cursor: pointer;
  margin-top: 20px;
  transition: background-color 0.3s ease;
}
.login-button:hover {
  background-color: #2980b9; 
}
.error-message {
    margin-top: 15px;
    color: var(--danger); 
}
</style>
