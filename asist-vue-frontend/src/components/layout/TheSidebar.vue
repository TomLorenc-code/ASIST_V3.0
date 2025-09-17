<template>
  <aside class="sidebar" :class="{ collapsed: isCollapsed }">
    <div class="sidebar-header">
      <div class="logo">
        <button class="sidebar-toggle-btn" @click="toggleSidebarState" :title="isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'">
          <i class="fas fa-bars"></i>
        </button>
        <div class="logo-text" v-if="!isCollapsed">
          <h1>ASIST</h1>
        </div>
      </div>
      <div class="user-info">
        <div class="user-avatar">{{ userInitials || '??' }}</div>
        <div class="user-details" v-if="!isCollapsed && userProfile">
          <div class="name">{{ userProfile.name || 'User' }}</div>
          <div class="role">{{ userProfile.role || 'Case Manager' }}</div> 
        </div>
         <div class="user-details" v-else-if="!isCollapsed && !userProfile">
            <div class="name">Not Logged In</div>
        </div>
      </div>
      <div class="case-search" :class="{ 'collapsed-search-clickable': isCollapsed }" @click="handleCollapsedSearchClick">
        <span class="nav-icon-search"><i class="fas fa-search"></i></span>
        <input 
          type="text" 
          placeholder="Search cases..." 
          id="caseSearchInputSidebar" 
          title="Search cases in sidebar"
          v-if="!isCollapsed"
          v-model="sidebarSearchTerm"
          @input="filterCasesInSidebar" 
          ref="sidebarSearchInputRef"
        >
      </div>
    </div>

    <div class="cases-scroll-area">
      <div class="quick-access-section" id="pinnedCasesSectionVue">
        <div class="quick-access-header">
            <span v-if="!isCollapsed">Pinned Cases</span>
            <span v-else class="nav-icon header-icon" title="Pinned Cases">
                <i class="fas fa-thumbtack"></i>
            </span>
        </div>
        <div 
          v-for="pinnedCase in caseStore.renderedPinnedCases" 
          :key="pinnedCase.id" 
          class="quick-access-item" 
          :title="pinnedCase.name" 
          @click="handleCaseSelection(pinnedCase.id)"
          :class="{ active: localSelectedCaseId === pinnedCase.id }"
        >
          <span class="nav-icon item-icon">
            <span v-if="isCollapsed" class="collapsed-text-icon">{{ pinnedCase.iconText }}</span>
            <i v-else class="fas fa-thumbtack"></i>
          </span>
          <div class="quick-access-text" v-if="!isCollapsed">{{ pinnedCase.id }}</div>
          <button 
            class="pin-button always-visible-pin" 
            :class="{ pinned: true }" 
            @click.stop="callTogglePinCase(pinnedCase.id)" 
            :title="'Unpin ' + pinnedCase.id"
            v-if="!isCollapsed" 
          >
            <i class="fas fa-star"></i>
          </button>
        </div>
        <div v-if="!isCollapsed && caseStore.renderedPinnedCases.length === 0" class="empty-state-message">No pinned cases.</div>
      </div>

      <div class="quick-access-section" id="recentCasesSectionVue">
        <div class="quick-access-header" v-if="!isCollapsed || (isCollapsed && caseStore.recentCasesData.length > 0)">
            <span v-if="!isCollapsed">Recent Cases</span>
            <span v-else-if="isCollapsed && caseStore.recentCasesData.length > 0" class="nav-icon header-icon" title="Recent Cases">
                <i class="fas fa-history"></i>
            </span>
        </div>
        <div 
          v-for="recentCaseItem in caseStore.recentCasesData" 
          :key="recentCaseItem.id" 
          class="quick-access-item" 
          :title="recentCaseItem.name" 
          @click="handleCaseSelection(recentCaseItem.id)"
          :class="{ active: localSelectedCaseId === recentCaseItem.id }"
        >
          <span class="nav-icon item-icon">
            <span v-if="isCollapsed" class="collapsed-text-icon">{{ recentCaseItem.iconText }}</span>
            <i v-else class="fas fa-history"></i>
          </span>
          <div class="quick-access-text" v-if="!isCollapsed">{{ recentCaseItem.id }}</div>
           <button 
            v-if="!isCollapsed"
            class="pin-button" 
            :class="{ pinned: caseStore.isCasePinned(recentCaseItem.id) }" 
            @click.stop="callTogglePinCase(recentCaseItem.id)" 
            :title="caseStore.isCasePinned(recentCaseItem.id) ? 'Unpin case' : 'Pin case'"
          >
            <i :class="caseStore.isCasePinned(recentCaseItem.id) ? 'fas fa-star' : 'far fa-star'"></i>
          </button>
        </div>
        <div v-if="!isCollapsed && caseStore.recentCasesData.length === 0" class="empty-state-message">No recent cases.</div>
      </div>

      <div class="countries-container">
        <div class="countries-list" id="countriesListVue">
          <div class="countries-header" v-if="!isCollapsed">
            <h2>Your Assigned Countries</h2>
          </div>
          <div class="country-with-cases" v-for="country in filteredCasesByCountry" :key="country.code">
            <div class="country-item" @click="toggleCountryExpansion(country.code)" :title="country.name">
              <span v-if="isCollapsed" class="nav-icon item-icon collapsed-text-icon country-code-icon">{{ country.code.toUpperCase() }}</span>
              <span v-else :class="['nav-icon', 'item-icon', 'country-code-icon']">{{ country.code.toUpperCase() }}</span>
              
              <div class="country-name" v-if="!isCollapsed">{{ country.name }}</div>
              <div class="country-toggle" v-if="!isCollapsed">
                <i :class="caseStore.expandedCountriesInSidebar.includes(country.code) ? 'fas fa-chevron-up' : 'fas fa-chevron-down'"></i>
              </div>
            </div>
            <div class="country-cases" :class="{ collapsed: !caseStore.expandedCountriesInSidebar.includes(country.code) }">
              <div 
                v-for="caseItem in country.cases" 
                :key="caseItem.id"
                class="case-item"
                :data-case-id="caseItem.id" 
                :data-case-name="caseItem.name"
                :data-case-icon-text="caseItem.iconText"
                :data-country-code="country.code" 
                @click="handleCaseSelection(caseItem.id)"
                :class="{ active: localSelectedCaseId === caseItem.id }"
                :title="caseItem.name"
              >
                <span class="nav-icon item-icon case-item-identifier-icon">
                  <span v-if="isCollapsed" class="collapsed-text-icon">{{ caseItem.iconText }}</span>
                  <i v-else class="fas fa-book"></i> 
                </span>
                <div class="case-text" v-if="!isCollapsed">{{ caseItem.id }}</div>
                <button 
                  v-if="!isCollapsed"
                  class="pin-button" 
                  :class="{ pinned: caseStore.isCasePinned(caseItem.id) }" 
                  @click.stop="callTogglePinCase(caseItem.id)" 
                  :title="caseStore.isCasePinned(caseItem.id) ? 'Unpin case' : 'Pin case'"
                >
                  <i :class="caseStore.isCasePinned(caseItem.id) ? 'fas fa-star' : 'far fa-star'"></i>
                </button>
              </div>
               <div v-if="!isCollapsed && country.cases.length === 0 && sidebarSearchTerm" class="empty-state-message">
                No cases match search in {{country.name}}.
              </div>
            </div>
          </div>
           <div v-if="caseStore.isLoadingCases" class="loading-sidebar-cases">Loading cases...</div>
           <div v-if="!caseStore.isLoadingCases && caseStore.casesByCountry.length === 0 && !caseStore.caseError" class="empty-state-message">
             No cases found for this user.
           </div>
           <div v-if="caseStore.caseError" class="error-sidebar-cases">
             Error loading cases: {{ caseStore.caseError }}
           </div>
        </div>
      </div>
    </div>

    <div class="sidebar-footer">
      <div class="settings-container" title="Settings" @click="openSettings">
        <span class="nav-icon item-icon"><i class="fas fa-cog"></i></span>
        <div class="settings-text" v-if="!isCollapsed">Settings</div>
      </div>
      <div class="settings-container logout-item" title="Logout" @click="triggerLogout" v-if="userProfile">
        <span class="nav-icon item-icon"><i class="fas fa-sign-out-alt"></i></span>
        <div class="settings-text" v-if="!isCollapsed">Logout</div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, defineProps, defineEmits, computed, onMounted, nextTick, watch } from 'vue';
import { useCaseStore } from '@/stores/caseStore'; 

const props = defineProps({
  isCollapsed: Boolean,
  userProfile: { 
    type: Object,
    default: () => null
  },
  userInitials: { 
    type: String,
    default: '??'
  },
  currentActiveCaseId: { 
    type: String,
    default: null
  }
});
const emit = defineEmits(['toggleSidebar', 'caseSelected', 'openSettings', 'logoutUser']);

const caseStore = useCaseStore();

const sidebarSearchTerm = ref('');
const sidebarSearchInputRef = ref(null); 
const localSelectedCaseId = ref(props.currentActiveCaseId); 

watch(() => props.currentActiveCaseId, (newId) => {
    localSelectedCaseId.value = newId;
});

const filteredCasesByCountry = computed(() => {
    if (!sidebarSearchTerm.value) {
        return caseStore.casesByCountry;
    }
    const term = sidebarSearchTerm.value.toLowerCase();
    return caseStore.casesByCountry.map(country => ({
        ...country,
        cases: country.cases.filter(caseItem =>
            (caseItem.id || '').toLowerCase().includes(term) ||
            (caseItem.name || '').toLowerCase().includes(term)
        )
    })).filter(country => country.cases.length > 0); 
});

const toggleSidebarState = () => {
  emit('toggleSidebar');
};

const handleCollapsedSearchClick = async () => {
  if (props.isCollapsed) {
    emit('toggleSidebar'); 
    await nextTick(); 
    if (sidebarSearchInputRef.value) {
        sidebarSearchInputRef.value.focus();
    }
  }
};

const filterCasesInSidebar = () => {
  // Handled by computed property
};

const toggleCountryExpansion = (countryCode) => {
  const currentExpanded = [...caseStore.expandedCountriesInSidebar];
  const index = currentExpanded.indexOf(countryCode);
  if (index > -1) {
    currentExpanded.splice(index, 1);
  } else {
    currentExpanded.push(countryCode);
  }
  caseStore.saveExpandedCountries(currentExpanded); 
};

const handleCaseSelection = (caseId) => {
  caseStore.setActiveCase(caseId); 
};

const callTogglePinCase = (caseId) => { 
  caseStore.togglePinCase(caseId);
};

const openSettings = () => {
  emit('openSettings');
};

const triggerLogout = () => {
  emit('logoutUser');
};

onMounted(() => {
  if (props.currentActiveCaseId) {
    localSelectedCaseId.value = props.currentActiveCaseId;
  } else {
      console.log('[TheSidebar] No active case prop, store will handle default selection on case fetch.');
  }
});

</script>

<style scoped>
/* Re-applied and verified styles for the sidebar */
.sidebar {
  width: 250px; 
  background-color: var(--primary); 
  color: var(--text-light);
  height: 100vh;
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  overflow-x: hidden; 
  position: fixed; 
  top: 0;
  left: 0;
  z-index: 1000;
}
.sidebar.collapsed {
  width: var(--sidebar-collapsed-width);
}

.sidebar-header {
  flex-shrink: 0;
  padding: var(--sidebar-header-padding) var(--sidebar-header-padding) 0 var(--sidebar-header-padding);
}
.logo {
  padding-bottom: var(--space-md);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  min-height: 40px; 
}
.sidebar-toggle-btn {
  background: transparent;
  border: none;
  color: var(--text-light); 
  cursor: pointer;
  font-size: 1.2rem;
  padding: var(--space-xs);
  display: flex;
  align-items: center;
  justify-content: center; 
  width: 30px; 
  height: 30px;
  border-radius: 4px;
  flex-shrink: 0; 
}
.sidebar:not(.collapsed) .sidebar-toggle-btn {
    margin-right: var(--space-sm);
}
.sidebar.collapsed .sidebar-toggle-btn {
  margin-right: 0; 
  width: 100%; 
  justify-content: flex-start; 
  padding-left: calc((var(--sidebar-collapsed-width) - 20px) / 2); 
}
.sidebar-toggle-btn:hover {
  background-color: rgba(255, 255, 255, 0.1);
}
.logo-text h1 {
  font-size: 1.6rem;
  font-weight: 600;
  white-space: nowrap;
  margin: 0; 
  color: var(--text-light); 
}

.user-info {
  padding: var(--space-sm) var(--sidebar-header-padding); 
  display: flex;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--text-light); 
}
.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: var(--space-sm);
  font-weight: bold;
  flex-shrink: 0;
  font-size: 0.8rem;
  color: white; 
}
.user-details .name, .user-details .role {
  font-size: 0.8rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text-light); 
}
.user-details .role {
  opacity: 0.7;
  font-size: 0.75rem;
}
.sidebar.collapsed .user-avatar {
  margin-right: 0; 
}

.case-search {
  padding: var(--space-sm) var(--sidebar-header-padding);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
}
.sidebar.collapsed .case-search.collapsed-search-clickable {
  padding: var(--space-sm) var(--sidebar-header-padding); 
  justify-content: flex-start; 
  cursor: pointer;
}
.nav-icon-search {
  color: rgba(255,255,255,0.6);
  font-size: 0.9rem;
  margin-right: var(--space-xs);
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
}
.sidebar.collapsed .nav-icon-search {
  font-size: 1.2rem; 
  margin-right: 0;
}
.case-search input {
  width: 100%;
  padding: var(--space-xs) var(--space-sm);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.1);
  color: var(--text-light); 
  font-size: 0.8rem;
  flex-grow: 1;
}
.case-search input::placeholder {
  color: rgba(255, 255, 255, 0.6);
}

.cases-scroll-area {
  flex-grow: 1;
  overflow-y: auto;
  padding-bottom: var(--space-md);
  color: var(--text-light); 
}
.cases-scroll-area::-webkit-scrollbar { width: 6px; }
.cases-scroll-area::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
.cases-scroll-area::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 3px;}

.quick-access-section {
  padding: var(--space-sm) 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.quick-access-header {
  padding: 0 var(--sidebar-header-padding) var(--space-xs);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  opacity: 0.6;
  display: flex; 
  align-items: center;
  min-height: 20px; 
  color: var(--text-light); 
}
.sidebar.collapsed .quick-access-header {
    padding: var(--space-xs) var(--sidebar-header-padding); 
}
.quick-access-header .nav-icon.header-icon { 
    margin-right: 0; 
    font-size: 1.2rem; 
    color: var(--text-light); 
    opacity: 0.8;
}
.empty-state-message {
    padding: var(--space-xs) var(--sidebar-header-padding);
    font-size: 0.8rem;
    opacity: 0.7;
    color: var(--text-light);
}

.quick-access-item, .country-item, .settings-container {
  padding: var(--space-xs) var(--sidebar-header-padding);
  display: flex;
  align-items: center;
  cursor: pointer;
  transition: background-color 0.2s;
  font-size: 0.85rem;
  min-height: 30px;
  color: var(--text-light); 
}
.quick-access-item:hover, .country-item:hover, .settings-container:hover {
  background-color: rgba(46, 7, 221, 0.423);
}
.quick-access-item.active, .case-item.active { 
    background-color: var(--accent) !important; 
    color: white !important; 
}
.quick-access-item.active .nav-icon, .case-item.active .nav-icon,
.quick-access-item.active .case-item-identifier-icon, .case-item.active .case-item-identifier-icon,
.quick-access-item.active .collapsed-text-icon, .case-item.active .collapsed-text-icon {
    color: white !important; 
}


.nav-icon.item-icon { 
  width: 20px;
  min-width: 20px;
  margin-right: var(--space-sm);
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center; 
  font-size: 0.9em; 
  color: var(--accent); 
}
.sidebar.collapsed .nav-icon.item-icon {
  margin-right: 0;
  font-size: 1.2rem; 
  width: auto; 
  justify-content: flex-start; 
  color: var(--text-light); 
}
.collapsed-text-icon { 
    font-weight: 600;
    color: var(--text-light); 
    opacity: 0.9;
    font-size: 0.9rem; 
    line-height: 1.2rem; 
    display: inline-block;
    width: auto; 
    text-align: left; 
    padding-left: 0; 
}
.sidebar.collapsed .country-item .country-code-icon {
    font-size: 0.9rem; 
    color: var(--text-light); 
}

.quick-access-text, .country-name, .settings-text, .case-text {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-grow: 1;
  color: inherit; 
}

.pin-button {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--text-light);
  cursor: pointer;
  opacity: 0.7;
  font-size: 0.9rem;
  padding: var(--space-xs);
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.pin-button:hover { opacity: 1; }
.pin-button.pinned .fa-star { color: var(--warning); }
.pin-button .fa-star { font-weight: 400; } 
.pin-button.pinned .fa-star { font-weight: 900; } 

.sidebar.collapsed .pin-button.always-visible-pin {
    display: flex; 
    width: auto; 
    padding-left: 0; 
}

.countries-list { padding: var(--space-md) 0; }
.countries-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 var(--sidebar-header-padding) var(--space-sm);
}
.countries-header h2 {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  opacity: 0.6;
  color: var(--text-light); 
}

.country-item.active { background-color: var(--accent); }

.country-code-icon { 
    margin-right: var(--space-sm); 
    color: var(--text-light); 
    font-weight: 600;
}
.sidebar.collapsed .country-item .country-code-icon {
    margin-right: 0;
}

.country-toggle {
  margin-left: auto;
  font-size: 0.8rem;
  opacity: 0.7;
  padding: var(--space-xs);
  width: 20px;
  text-align: center;
  color: var(--text-light); 
}

.country-cases {
  max-height: 500px;
  overflow: hidden;
  transition: max-height 0.3s ease-out;
}
.country-cases.collapsed {
  max-height: 0;
}

.case-item {
  padding: 3px var(--sidebar-header-padding) 3px var(--sidebar-header-padding); 
  font-size: 0.8rem;
  line-height: 1.3;
  transition: background-color 0.2s;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: flex;
  align-items: center;
  position: relative; 
  color: var(--text-light); 
}
.sidebar:not(.collapsed) .case-item .case-item-identifier-icon + .case-text {
  margin-left: calc(20px + var(--space-sm)); 
}
.sidebar.collapsed .case-item {
  justify-content: flex-start; 
}
.case-item:hover { background-color: rgba(255, 255, 255, 0.1); }

.case-item-identifier-icon { 
    width: 20px; 
    min-width: 20px;
    margin-right: var(--space-sm);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: var(--accent); 
}
.sidebar:not(.collapsed) .case-item-identifier-icon {
    position: absolute;
    left: var(--sidebar-header-padding);
    top: 50%;
    transform: translateY(-50%);
}
.sidebar.collapsed .case-item-identifier-icon {
    margin-right: 0; 
}
.case-item-identifier-icon .fa-book {
    font-size: 0.9em; 
}
.sidebar.collapsed .case-item .pin-button {
    display: none; 
}

.sidebar-footer {
  margin-top: auto; 
  flex-shrink: 0; 
  border-top: 1px solid rgba(255, 255, 255, 0.1); 
}

.settings-container { 
  border-top: none; 
  color: var(--text-light); 
}
.sidebar.collapsed .settings-container .nav-icon {
    padding-left: 0; 
    justify-content: flex-start; 
    color: var(--text-light); 
}
.logout-item:hover {
  background-color: rgba(231, 76, 60, 0.2); 
}
.logout-item .nav-icon i {
  color: var(--danger); 
}
.sidebar.collapsed .logout-item .nav-icon i {
    color: var(--danger); 
}

.loading-sidebar-cases, .error-sidebar-cases {
    padding: var(--space-md) var(--sidebar-header-padding);
    color: var(--text-light);
    opacity: 0.8;
}
.error-sidebar-cases {
    color: var(--danger);
}
</style>
