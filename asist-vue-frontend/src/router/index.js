import { createRouter, createWebHistory } from 'vue-router';
import MainDashboardLayout from '../components/layout/MainDashboardLayout.vue'; // Import MainDashboardLayout

const routes = [
  {
    path: '/',
    name: 'dashboard', // Changed name to 'dashboard' for clarity
    component: MainDashboardLayout,
    // Props can be passed to routed components here if needed,
    // but for isSidebarCollapsed and activeCase, MainDashboardLayout
    // will get them from Pinia stores or as direct props from <RouterView> in App.vue
  },
  // You can add other top-level routes here later, for example:
  // {
  //   path: '/settings-page',
  //   name: 'SettingsPage',
  //   component: () => import('../views/SettingsView.vue') // Example for a different page
  // }
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
});

export default router;
