<template>
  <section class="case-details-panel" v-if="activeCaseData">
    <div class="case-header">
      <div class="case-title-main-area"> 
        <div class="case-title-container">
          <h2 id="activeCaseTitle">
            <span id="activeCaseNumberDisplay" class="case-number-title">{{ activeCaseData.caseNumber || 'N/A' }}:</span>
            <span id="activeCaseDescriptionDisplay" class="case-description-title">{{ activeCaseData.caseDescription || 'No Description' }}</span>
          </h2>
          <div class="search-bar-case">
            <span class="nav-icon"><i class="fas fa-search"></i></span>
            <input type="text" placeholder="Search this case..." title="Search within this case">
          </div>
        </div>
        <div class="case-meta">
          <div class="case-meta-item" title="Creation Date">
            <span class="nav-icon"><i class="far fa-calendar-alt"></i></span>
            <span>Created: {{ formatDate(activeCaseData.createdAt) }}</span>
          </div>
          <div class="case-meta-item" title="Case Value">
            <span class="nav-icon"><i class="fas fa-dollar-sign"></i></span>
            <span>Value: {{ formatCurrency(activeCaseData.estimatedValue) }}</span>
          </div>
          <div class="case-meta-item" title="LOA Status">
            <span class="nav-icon"><i class="far fa-file-alt"></i></span>
            <span>LOA: {{ activeCaseData.loaStatus || 'N/A' }}</span>
          </div>
        </div>
      </div>
      <div class="case-status">
        <div class="status-badge" :class="statusBadgeClass">{{ activeCaseData.status || 'Unknown' }}</div>
      </div>
    </div>

    <nav class="tabs">
      <div 
        v-for="tab in tabs" 
        :key="tab.id" 
        class="tab" 
        :class="{ active: localActiveTab === tab.id }"
        @click="selectTab(tab.id)"
        role="tab"
        :aria-selected="localActiveTab === tab.id"
        :aria-controls="`${tab.id}ContentVue`" 
      >
        {{ tab.name }}
      </div>
    </nav>
  </section>
  <section v-else class="case-details-panel placeholder">
    <p>No case selected or case data not available.</p>
    <p v-if="caseStore.isLoadingCases">Loading cases...</p>
    <p v-if="caseStore.caseError">Error loading case data: {{ caseStore.caseError }}</p>
  </section>
</template>

<script setup>
import { ref, defineProps, defineEmits, computed, watch, onMounted } from 'vue';
import { useCaseStore } from '@/stores/caseStore'; // Import if needed for loading/error state display

const props = defineProps({
  activeCaseData: { // This prop should be the full case object from caseStore.activeCaseDetails
    type: Object,
    default: () => null
  }
});

const emit = defineEmits(['tab-selected']);
const caseStore = useCaseStore(); // For isLoadingCases and caseError display

const tabs = ref([
  { id: 'overview', name: 'Overview' },
  { id: 'timeline', name: 'Timeline' },
  { id: 'financial', name: 'Financial' },
  { id: 'logistics', name: 'Logistics' },
  { id: 'reports', name: 'Reports' },
  { id: 'documents', name: 'Documents' }
]);

const localActiveTab = ref('overview'); 

const selectTab = (tabId) => {
  localActiveTab.value = tabId;
  emit('tab-selected', tabId); 
};

const statusBadgeClass = computed(() => {
  if (!props.activeCaseData || !props.activeCaseData.status) return 'status-unknown';
  const status = props.activeCaseData.status.toLowerCase();
  if (status === 'active' || status === 'imported') return 'status-active'; // Treat 'imported' as active for now
  if (status === 'implemented') return 'status-implemented';
  if (status === 'closed') return 'status-closed';
  return 'status-other';
});

const formatCurrency = (value) => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return 'N/A';
  }
  const numberValue = Number(value);
  return `$${numberValue.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
};

const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });
    } catch (e) {
        return 'Invalid Date';
    }
};

watch(() => props.activeCaseData, (newCaseData, oldCaseData) => {
  console.log("[CaseDetailsPanel] activeCaseData prop updated:", newCaseData);
  if (newCaseData && (!oldCaseData || newCaseData.id !== oldCaseData.id || newCaseData.caseNumber !== oldCaseData.caseNumber)) {
    localActiveTab.value = 'overview'; 
    emit('tab-selected', 'overview'); 
  } else if (newCaseData && !oldCaseData) { 
    emit('tab-selected', localActiveTab.value); // Emit initial tab
  }
}, { immediate: true, deep: true });

onMounted(() => {
    if (props.activeCaseData) {
        console.log("[CaseDetailsPanel] Mounted with case:", props.activeCaseData.id || props.activeCaseData.caseNumber);
    } else {
        console.log("[CaseDetailsPanel] Mounted with no active case data.");
    }
});

</script>

<style scoped>
.case-details-panel {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  padding: var(--space-md);
  margin-bottom: var(--space-sm); 
  flex-shrink: 0; 
}
.case-details-panel.placeholder {
    text-align: center;
    color: #777;
    padding: var(--space-lg);
}

.case-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start; 
  margin-bottom: var(--space-sm);
  gap: var(--space-md);
}

.case-title-main-area {
    flex-grow: 1;
    min-width: 0; /* Allow shrinking for flex items */
}

.case-title-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-grow: 1;
  gap: var(--space-md);
  margin-bottom: var(--space-xs);
}

.case-title-container h2 {
  font-size: 1.3rem;
  display: flex;
  align-items: center;
  margin: 0;
  flex-shrink: 1; /* Allow title to shrink if search bar is wide */
  min-width: 0; /* Allow h2 to shrink */
}
.case-number-title {
    font-weight: 600; /* Make case number bold */
    margin-right: var(--space-xs);
    white-space: nowrap;
}
.case-description-title {
    font-weight: normal; /* Ensure description is normal weight */
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-dark); /* Ensure it's not too light */
}


.search-bar-case {
  display: flex;
  align-items: center;
  background-color: #f0f0f0;
  border-radius: 4px;
  padding: var(--space-xs) var(--space-sm);
  width: 250px;
  max-width: 300px;
  border: 1px solid var(--border);
  flex-shrink: 0;
}
.search-bar-case .nav-icon { 
  color: var(--text-dark);
  margin-right: var(--space-xs);
  font-size: 0.9rem;
}
.search-bar-case input {
  border: none;
  outline: none;
  flex: 1;
  font-size: 0.85rem;
  background: transparent;
}

.case-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  font-size: 0.85rem;
  color: #666;
}
.case-meta-item {
  display: flex;
  align-items: center;
}
.case-meta-item .nav-icon { 
  margin-right: var(--space-xs);
  color: var(--accent);
  font-size: 0.9rem;
}

.case-status {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}
.status-badge {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  color: white;
}
.status-badge.status-active { background-color: var(--success); }
.status-badge.status-implemented { background-color: var(--accent); } 
.status-badge.status-closed { background-color: var(--secondary); } 
.status-badge.status-unknown, .status-badge.status-other { background-color: var(--border); color: var(--text-dark); }


.tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0; 
}
.tab {
  padding: var(--space-sm) var(--space-md);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-weight: 500;
  font-size: 0.9rem;
  color: var(--text-dark);
  transition: color 0.2s ease, border-bottom-color 0.2s ease;
}
.tab:hover {
  color: var(--accent);
}
.tab.active {
  border-bottom-color: var(--accent);
  color: var(--accent);
}
</style>
