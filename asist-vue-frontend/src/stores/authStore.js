import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

// Define the store
export const useAuthStore = defineStore('auth', () => {
  // --- State ---
  const user = ref(null); // Will hold user profile object from Auth0/Flask session
  const isLoading = ref(false); // To track loading state for user profile
  const error = ref(null); // To store any auth-related errors

  // --- Getters ---
  const isLoggedIn = computed(() => !!user.value);
  const userProfile = computed(() => user.value);
  const authIsLoading = computed(() => isLoading.value);
  const authError = computed(() => error.value);
  const userInitials = computed(() => {
    if (user.value && user.value.name) {
      const nameParts = user.value.name.split(' ');
      if (nameParts.length > 1) {
        return (nameParts[0][0] + nameParts[nameParts.length - 1][0]).toUpperCase();
      }
      return nameParts[0].substring(0, 2).toUpperCase();
    }
    return '??'; // Default if no name
  });


  // --- Actions ---
  
  // Action to fetch the current user's status/profile from our Flask backend
  // This endpoint needs to be created in Flask and should return user data if session exists
  async function fetchUser() {
    isLoading.value = true;
    error.value = null;
    try {
      // IMPORTANT: This API endpoint '/api/me' needs to be created in your Flask app.
      // It should check the Flask session and return user data if logged in, or 401 if not.
      const response = await fetch('/api/me'); // Relative path to your Flask backend
      
      if (response.ok) {
        const data = await response.json();
        if (data && data.userinfo) { // Assuming Flask returns { userinfo: { ... } } or similar
          user.value = data.userinfo; 
          console.log('[AuthStore] User fetched:', user.value);
        } else if (data && data.sub) { // If Flask returns userinfo directly
          user.value = data;
          console.log('[AuthStore] User fetched (direct):', user.value);
        }
         else {
          user.value = null; // No user data in response
        }
      } else if (response.status === 401) { // Unauthorized
        user.value = null;
        console.log('[AuthStore] User not authenticated (401).');
      } else {
        const errorData = await response.json().catch(() => ({ message: response.statusText }));
        throw new Error(errorData.message || `Failed to fetch user: ${response.status}`);
      }
    } catch (e) {
      console.error('[AuthStore] Error fetching user:', e);
      user.value = null;
      error.value = e.message;
    } finally {
      isLoading.value = false;
    }
  }

  // Action to initiate login - redirects to Flask's /login route
  function login() {
    // Flask will handle the redirect to Auth0
    window.location.href = '/login'; 
  }

  // Action to initiate logout - redirects to Flask's /logout route
  function logout() {
    user.value = null; // Optimistically clear user state
    // Flask will clear its session and handle Auth0 logout
    window.location.href = '/logout';
  }

  return { 
    user, 
    isLoading, 
    error, 
    isLoggedIn, 
    userProfile,
    userInitials,
    authIsLoading,
    authError,
    fetchUser, 
    login, 
    logout 
  };
});
