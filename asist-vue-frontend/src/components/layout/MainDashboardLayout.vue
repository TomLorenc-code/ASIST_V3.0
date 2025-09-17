<template>
  <main 
    class="main-dashboard-layout" 
    :style="{ marginLeft: sidebarMarginLeft }"
  >
    <CaseDetailsPanel 
      :active-case-data="caseStore.activeCaseDetails" 
      @tab-selected="handleTabSelection" 
    />

    <div class="ai-assistant-area">
      <TabContentPanel 
        :active-case-id="caseStore.activeCaseDetails?.id || caseStore.activeCaseDetails?.caseNumber" 
        :active-tab-id="selectedTabId" 
        :active-case-data="caseStore.activeCaseDetails"
        :style="{ flexGrow: tabFlexGrow, transition: 'flex-grow 0.3s ease' }"
        class="content-panel"
        :class="{ 'panel-collapsed': isChatPanelExpanded }"
      />

      <div class="panel-resizer" @click="togglePanelExpansion" :title="isChatPanelExpanded ? 'Expand Info Panel' : 'Expand Chat Panel'">
        <i :class="isChatPanelExpanded ? 'fas fa-angle-right' : 'fas fa-angle-left'"></i>
      </div>

      <ChatInterface 
        :active-case-id="caseStore.activeCaseDetails?.id || caseStore.activeCaseDetails?.caseNumber" 
        :style="{ flexGrow: chatFlexGrow, transition: 'flex-grow 0.3s ease' }"
        class="content-panel"
        :class="{ 'panel-collapsed': !isChatPanelExpanded && totalFlexGrow > 2 }" 
      />
    </div>
  </main>
</template>

<script setup>
import { defineProps, ref, watch, computed } from 'vue';
import CaseDetailsPanel from '../dashboard/CaseDetailsPanel.vue'; 
import TabContentPanel from '../dashboard/TabContentPanel.vue';   
import ChatInterface from '../chat/ChatInterface.vue';         
import { useCaseStore } from '@/stores/caseStore'; 

const props = defineProps({
  isSidebarCollapsed: Boolean 
});

const caseStore = useCaseStore(); 
const selectedTabId = ref('overview'); 
const isChatPanelExpanded = ref(false); 

const baseTabFlex = 1.5; 
const baseChatFlex = 1;
const expandedFlex = 2.5; 
const collapsedFlex = 0.5; 

// Computed property for dynamic margin-left
const sidebarMarginLeft = computed(() => {
  // Assuming these values are consistent with TheSidebar.vue CSS
  const expandedSidebarWidth = "250px"; 
  const collapsedSidebarWidth = "70px"; // Or use var(--sidebar-collapsed-width) if accessible
  return props.isSidebarCollapsed ? collapsedSidebarWidth : expandedSidebarWidth;
});

const tabFlexGrow = computed(() => {
  return isChatPanelExpanded.value ? collapsedFlex : baseTabFlex;
});

const chatFlexGrow = computed(() => {
  return isChatPanelExpanded.value ? expandedFlex : baseChatFlex;
});

const totalFlexGrow = computed(() => tabFlexGrow.value + chatFlexGrow.value);

const handleTabSelection = (tabId) => {
  console.log(`[MainDashboardLayout] Received 'tab-selected' event with: ${tabId}`);
  selectedTabId.value = tabId;
  console.log(`[MainDashboardLayout] selectedTabId is now: ${selectedTabId.value}`);
};

const togglePanelExpansion = () => {
  isChatPanelExpanded.value = !isChatPanelExpanded.value;
  console.log(`[MainDashboardLayout] Chat panel expanded: ${isChatPanelExpanded.value}`);
};

watch(() => caseStore.activeCaseDetails?.id, (newCaseId, oldCaseId) => {
  if (newCaseId && newCaseId !== oldCaseId) {
    console.log("[MainDashboardLayout] Active case from store changed from", oldCaseId, "to", newCaseId, ". Resetting selectedTabId to overview.");
    selectedTabId.value = 'overview'; 
  } else if (newCaseId && !oldCaseId && selectedTabId.value !== 'overview') {
    selectedTabId.value = 'overview';
    console.log("[MainDashboardLayout] Initial case data from store, ensuring selectedTabId is overview.");
  }
}, {deep: true, immediate: true });

</script>

<style scoped>
.main-dashboard-layout {
  flex-grow: 1; 
  padding: var(--space-md); 
  display: flex;
  flex-direction: column;
  height: 100vh; 
  overflow-y: hidden; 
  transition: margin-left 0.3s ease; /* Keep transition for smooth shift */
}

.ai-assistant-area {
  display: flex;
  flex-grow: 1; 
  overflow: hidden; 
  margin-top: var(--space-sm); 
  align-items: stretch; 
}

.content-panel {
    min-width: 400px; 
    overflow: hidden; 
    display: flex; 
    flex-direction: column;
}

.panel-resizer {
  flex-shrink: 0;
  width: 12px; 
  background-color: var(--light); 
  cursor: pointer; /* Correct cursor for a clickable toggle */
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border);
  border-right: 1px solid var(--border);
  z-index: 10; 
  transition: background-color 0.2s ease;
}
.panel-resizer:hover {
  background-color: #e0e0e0; 
}
.panel-resizer i {
  color: var(--secondary);
  font-size: 0.9rem;
}
</style>
