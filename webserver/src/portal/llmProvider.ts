import type {
  LlmModuleResponse,
  LlmModuleSettings,
  LlmProviderId,
  StatusType,
  ValidatableField,
} from './types';

type KeyInput = ValidatableField & { readOnly: boolean };

interface ProviderKeyState {
  displayName: string;
  hasKey: boolean;
  last4: string;
}

interface LlmProviderDeps {
  apiKeyInput: KeyInput;
  fetchLlmModuleJson: (path: string, init?: RequestInit) => Promise<LlmModuleResponse>;
  notify: (message: string, type?: StatusType) => void;
  providerSelect: ValidatableField;
  updateUi: () => void;
}

const SETTINGS_API = '/api/settings/llm';
const RESET_API = '/api/settings/llm/reset';
const PROVIDER_IDS: LlmProviderId[] = ['muse', 'gemini'];
const DEFAULT_DISPLAY_NAMES: Record<LlmProviderId, string> = {
  gemini: 'Gemini',
  muse: 'Muse',
};

function isProviderId(value: unknown): value is LlmProviderId {
  return value === 'muse' || value === 'gemini';
}

function emptyKeyState(id: LlmProviderId): ProviderKeyState {
  return { displayName: DEFAULT_DISPLAY_NAMES[id], hasKey: false, last4: '' };
}

/**
 * Drives the "AI Provider" card: a Muse/Gemini picker plus the API key for the picked provider.
 * The firmware stores one key per provider, so switching never discards a saved key. Saving a
 * key also makes that provider the active one.
 */
export function createLlmProviderController(deps: LlmProviderDeps) {
  let activeProvider: LlmProviderId = 'muse'; // what the firmware is using
  let selectedProvider: LlmProviderId = 'muse'; // what the card is showing
  let isBusy = false;
  let loaded = false;
  const keys: Record<LlmProviderId, ProviderKeyState> = {
    gemini: emptyKeyState('gemini'),
    muse: emptyKeyState('muse'),
  };

  function displayName(id: LlmProviderId = selectedProvider) {
    return keys[id].displayName;
  }

  function render() {
    const state = keys[selectedProvider];
    deps.providerSelect.value = selectedProvider;
    deps.apiKeyInput.value = state.hasKey ? (state.last4 ? `******${state.last4}` : '******') : '';
  }

  function applySettings(settings?: LlmModuleSettings) {
    if (isProviderId(settings?.provider)) {
      activeProvider = settings.provider;
      selectedProvider = activeProvider;
    }
    for (const id of PROVIDER_IDS) {
      const state = settings?.providers?.[id];
      keys[id] = {
        displayName: state?.display_name || DEFAULT_DISPLAY_NAMES[id],
        hasKey: state?.has_key === true,
        last4: typeof state?.last4 === 'string' ? state.last4 : '',
      };
    }
    loaded = true;
    render();
  }

  async function load() {
    try {
      const data = await deps.fetchLlmModuleJson(SETTINGS_API);
      applySettings(data.settings);
    } catch (error) {
      console.error('AI provider settings load failed:', error);
    } finally {
      deps.updateUi();
    }
  }

  async function handleProviderChange() {
    const next = deps.providerSelect.value;
    if (!isProviderId(next) || next === selectedProvider || isBusy) {
      render();
      return;
    }
    selectedProvider = next;
    render();
    if (!loaded) {
      deps.updateUi();
      return;
    }

    isBusy = true;
    deps.notify(`Switching to ${displayName()}...`, 'info');
    deps.updateUi();
    try {
      const data = await deps.fetchLlmModuleJson(SETTINGS_API, {
        method: 'PATCH',
        body: JSON.stringify({ provider: next }),
      });
      applySettings(data.settings);
      deps.notify(data.message || `Using ${displayName()}.`, 'success');
    } catch (error) {
      console.error('AI provider switch failed:', error);
      deps.notify(
        error instanceof Error ? error.message : 'Failed to switch AI provider.',
        'error'
      );
      selectedProvider = activeProvider;
      render();
    } finally {
      isBusy = false;
      deps.updateUi();
    }
  }

  async function saveKey() {
    if (isBusy || keys[selectedProvider].hasKey) {
      return;
    }
    const apiKey = deps.apiKeyInput.value.trim();
    if (!apiKey) {
      deps.notify(`${displayName()} API key is required.`, 'warning');
      return;
    }

    isBusy = true;
    deps.notify(`Saving ${displayName()} API key...`, 'info');
    deps.updateUi();
    try {
      const data = await deps.fetchLlmModuleJson(SETTINGS_API, {
        method: 'PATCH',
        body: JSON.stringify({
          api_key: apiKey,
          key_provider: selectedProvider,
          provider: selectedProvider,
        }),
      });
      applySettings(data.settings);
      deps.notify(data.message || `${displayName()} API key stored.`, 'success');
    } catch (error) {
      console.error('AI provider key save failed:', error);
      deps.notify(
        error instanceof Error ? error.message : `Failed to store ${displayName()} API key.`,
        'error'
      );
    } finally {
      isBusy = false;
      deps.updateUi();
    }
  }

  async function clearKey() {
    if (isBusy || !keys[selectedProvider].hasKey) {
      return;
    }

    isBusy = true;
    deps.notify(`Clearing ${displayName()} API key...`, 'info');
    deps.updateUi();
    try {
      const data = await deps.fetchLlmModuleJson(RESET_API, {
        method: 'POST',
        body: JSON.stringify({ provider: selectedProvider }),
      });
      applySettings(data.settings);
      deps.notify(data.message || `${displayName()} API key cleared.`, 'success');
    } catch (error) {
      console.error('AI provider key clear failed:', error);
      deps.notify(
        error instanceof Error ? error.message : `Failed to clear ${displayName()} API key.`,
        'error'
      );
    } finally {
      isBusy = false;
      deps.updateUi();
    }
  }

  return {
    applySettings,
    clearKey,
    getSelectedProvider: () => selectedProvider,
    handleProviderChange,
    hasKey: () => keys[selectedProvider].hasKey,
    isBusy: () => isBusy,
    load,
    saveKey,
  };
}
