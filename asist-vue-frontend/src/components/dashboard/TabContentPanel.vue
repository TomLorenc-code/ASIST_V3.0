<template>
  <section class="tab-content-panel">
    <OverviewContent v-if="activeTabId === 'overview'" :case-data="activeCaseData?.overview" />
    <TimelineContent v-if="activeTabId === 'timeline'" :case-data="activeCaseData?.timelineData" />
    <FinancialContent v-if="activeTabId === 'financial'" :case-data="activeCaseData?.financialData" />
    <LogisticsContent v-if="activeTabId === 'logistics'" :case-data="activeCaseData?.logisticsData" />
    <ReportsContent v-if="activeTabId === 'reports'" :case-data="activeCaseData?.reportsData" />
    <DocumentsContent v-if="activeTabId === 'documents'" :active-case-data="activeCaseData" :active-case-id="activeCaseId" />

    <div v-if="!isKnownTab" class="tab-content-section placeholder-content">
      <p>Content for '{{ activeTabId }}' is not yet implemented or tab ID is unknown.</p>
    </div>
  </section>
</template>

<script setup>
import { defineProps, computed, watch } from 'vue';

import OverviewContent from './tabs/OverviewContent.vue';
import TimelineContent from './tabs/TimelineContent.vue';
import FinancialContent from './tabs/FinancialContent.vue';
import LogisticsContent from './tabs/LogisticsContent.vue';
import ReportsContent from './tabs/ReportsContent.vue';
import DocumentsContent from './tabs/DocumentsContent.vue'; 

const props = defineProps({
  activeTabId: {
    type: String,
    required: true,
    default: 'overview'
  },
  activeCaseId: { 
    type: String,
    default: null
  },
  activeCaseData: { 
    type: Object,
    default: () => null
  }
});

// Log when activeTabId prop changes
watch(() => props.activeTabId, (newTabId, oldTabId) => {
  console.log(`[TabContentPanel] activeTabId prop changed from '${oldTabId}' to '${newTabId}'`);
}, { immediate: true }); // immediate: true to log initial value

const knownTabIds = ['overview', 'timeline', 'financial', 'logistics', 'reports', 'documents'];
const isKnownTab = computed(() => knownTabIds.includes(props.activeTabId));

</script>

<style scoped>
.tab-content-panel {
  flex: 1; 
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  display: flex; 
  flex-direction: column; 
  overflow-y: auto; 
}

.placeholder-content {
    padding: var(--space-md);
    text-align: center;
    color: #777;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
