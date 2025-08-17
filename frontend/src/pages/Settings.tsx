import React, { useState, useEffect } from 'react';
import { saveSettings, fetchSettings, type AppSettings } from '../services/apiService';

interface Settings extends AppSettings {
  apiKeys: {
    openai: string;
    anthropic: string;
    google: string;
    cohere: string;
    openrouter: string;
  };
  defaults: {
    defaultLLM: string;
    defaultScenario: string;
    defaultAgent: string;
    defaultMetrics: string[];
    autoSave: boolean;
    notifications: boolean;
  };
  ui: {
    theme: 'light' | 'dark' | 'system';
    language: string;
    timezone: string;
  };
}

const Settings: React.FC = () => {
  const [settings, setSettings] = useState<Settings>({
    apiKeys: {
      openai: '',
      anthropic: '',
      google: '',
      cohere: '',
      openrouter: '',
    },
    defaults: {
      defaultLLM: 'gpt-4',
      defaultScenario: 'standard',
      defaultAgent: 'basic',
      defaultMetrics: ['revenue', 'profit', 'costs'],
      autoSave: true,
      notifications: true,
    },
    ui: {
      theme: 'system',
      language: 'en',
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    },
  });
  
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'api' | 'defaults' | 'ui'>('api');

  useEffect(() => {
    const loadSettings = async () => {
      try {
        setLoading(true);
        const savedSettings = await fetchSettings();
        if (savedSettings) {
          setSettings(savedSettings);
        }
      } catch (error) {
        console.error('Failed to load settings:', error);
        setMessage({ type: 'error', text: 'Failed to load settings. Using defaults.' });
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, []);

  const handleSave = async () => {
    try {
      setSaving(true);
      await saveSettings(settings);
      setMessage({ type: 'success', text: 'Settings saved successfully!' });
    } catch (error) {
      console.error('Failed to save settings:', error);
      setMessage({ type: 'error', text: 'Failed to save settings. Please try again.' });
    } finally {
      setSaving(false);
    }
  };

  const handleApiKeyChange = (provider: keyof Settings['apiKeys'], value: string) => {
    setSettings(prev => ({
      ...prev,
      apiKeys: {
        ...prev.apiKeys,
        [provider]: value,
      },
    }));
  };

  const handleDefaultChange = (key: keyof Settings['defaults'], value: string | boolean | string[]) => {
    setSettings(prev => ({
      ...prev,
      defaults: {
        ...prev.defaults,
        [key]: value,
      },
    }));
  };

  const handleUiChange = (key: keyof Settings['ui'], value: string) => {
    setSettings(prev => ({
      ...prev,
      ui: {
        ...prev.ui,
        [key]: value,
      },
    }));
  };

  const handleMetricToggle = (metric: string) => {
    setSettings(prev => {
      const metrics = prev.defaults.defaultMetrics.includes(metric)
        ? prev.defaults.defaultMetrics.filter(m => m !== metric)
        : [...prev.defaults.defaultMetrics, metric];
      
      return {
        ...prev,
        defaults: {
          ...prev.defaults,
          defaultMetrics: metrics,
        },
      };
    });
  };

  if (loading) {
    return (
      <div className="p-6 bg-gray-100 min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Settings & Administration</h1>
          <p className="text-gray-600">Configure API keys, default settings, and UI preferences</p>
        </div>

        {message && (
          <div className={`mb-6 p-4 rounded-md ${message.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
            {message.text}
          </div>
        )}

        <div className="bg-white rounded-lg shadow">
          {/* Tab Navigation */}
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('api')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'api'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                API Keys
              </button>
              <button
                onClick={() => setActiveTab('defaults')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'defaults'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Default Settings
              </button>
              <button
                onClick={() => setActiveTab('ui')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'ui'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                UI Preferences
              </button>
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {/* API Keys Tab */}
            {activeTab === 'api' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-4">LLM API Keys</h2>
                  <p className="text-gray-600 mb-6">
                    Enter your API keys for different LLM providers. These keys are stored securely and used only for benchmarking.
                  </p>
                  
                  <div className="space-y-4">
                    {Object.entries(settings.apiKeys).map(([provider, key]) => (
                      <div key={provider}>
                        <label htmlFor={`${provider}-api-key`} className="block text-sm font-medium text-gray-700 mb-1">
                          {provider.charAt(0).toUpperCase() + provider.slice(1)} API Key
                        </label>
                        <input
                          type="password"
                          id={`${provider}-api-key`}
                          value={key}
                          onChange={(e) => handleApiKeyChange(provider as keyof Settings['apiKeys'], e.target.value)}
                          className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                          placeholder={`Enter your ${provider} API key`}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Default Settings Tab */}
            {activeTab === 'defaults' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-4">Benchmark Defaults</h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="default-llm" className="block text-sm font-medium text-gray-700 mb-1">
                        Default LLM Model
                      </label>
                      <select
                        id="default-llm"
                        value={settings.defaults.defaultLLM}
                        onChange={(e) => handleDefaultChange('defaultLLM', e.target.value)}
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      >
                        <option value="gpt-4">GPT-4</option>
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                        <option value="claude-3-opus">Claude 3 Opus</option>
                        <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                        <option value="gemini-pro">Gemini Pro</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="default-scenario" className="block text-sm font-medium text-gray-700 mb-1">
                        Default Scenario
                      </label>
                      <select
                        id="default-scenario"
                        value={settings.defaults.defaultScenario}
                        onChange={(e) => handleDefaultChange('defaultScenario', e.target.value)}
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      >
                        <option value="standard">Standard Business</option>
                        <option value="competitive">Competitive Market</option>
                        <option value="volatile">Volatile Market</option>
                        <option value="resource-constrained">Resource Constrained</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="default-agent" className="block text-sm font-medium text-gray-700 mb-1">
                        Default Agent Type
                      </label>
                      <select
                        id="default-agent"
                        value={settings.defaults.defaultAgent}
                        onChange={(e) => handleDefaultChange('defaultAgent', e.target.value)}
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      >
                        <option value="basic">Basic Agent</option>
                        <option value="advanced">Advanced Agent</option>
                        <option value="adaptive">Adaptive Agent</option>
                        <option value="specialized">Specialized Agent</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Default Metrics
                      </label>
                      <div className="space-y-2 mt-2">
                        {['revenue', 'profit', 'costs', 'decisionsMade', 'accuracy', 'efficiency'].map((metric) => (
                          <label key={metric} className="inline-flex items-center mr-4">
                            <input
                              type="checkbox"
                              checked={settings.defaults.defaultMetrics.includes(metric)}
                              onChange={() => handleMetricToggle(metric)}
                              className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                            />
                            <span className="ml-2 text-sm text-gray-700 capitalize">{metric}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 space-y-4">
                    <div className="flex items-center">
                      <input
                        id="auto-save"
                        type="checkbox"
                        checked={settings.defaults.autoSave}
                        onChange={(e) => handleDefaultChange('autoSave', e.target.checked)}
                        className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                      />
                      <label htmlFor="auto-save" className="ml-2 block text-sm text-gray-700">
                        Auto-save configurations
                      </label>
                    </div>

                    <div className="flex items-center">
                      <input
                        id="notifications"
                        type="checkbox"
                        checked={settings.defaults.notifications}
                        onChange={(e) => handleDefaultChange('notifications', e.target.checked)}
                        className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notifications" className="ml-2 block text-sm text-gray-700">
                        Enable notifications
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* UI Preferences Tab */}
            {activeTab === 'ui' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-4">User Interface</h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="theme" className="block text-sm font-medium text-gray-700 mb-1">
                        Theme
                      </label>
                      <select
                        id="theme"
                        value={settings.ui.theme}
                        onChange={(e) => handleUiChange('theme', e.target.value as 'light' | 'dark' | 'system')}
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      >
                        <option value="light">Light</option>
                        <option value="dark">Dark</option>
                        <option value="system">System Default</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
                        Language
                      </label>
                      <select
                        id="language"
                        value={settings.ui.language}
                        onChange={(e) => handleUiChange('language', e.target.value)}
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      >
                        <option value="en">English</option>
                        <option value="es">Spanish</option>
                        <option value="fr">French</option>
                        <option value="de">German</option>
                        <option value="zh">Chinese</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="timezone" className="block text-sm font-medium text-gray-700 mb-1">
                        Timezone
                      </label>
                      <select
                        id="timezone"
                        value={settings.ui.timezone}
                        onChange={(e) => handleUiChange('timezone', e.target.value)}
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                      >
                        <option value="UTC">UTC</option>
                        <option value="America/New_York">Eastern Time</option>
                        <option value="America/Chicago">Central Time</option>
                        <option value="America/Denver">Mountain Time</option>
                        <option value="America/Los_Angeles">Pacific Time</option>
                        <option value="Europe/London">London</option>
                        <option value="Europe/Paris">Paris</option>
                        <option value="Asia/Tokyo">Tokyo</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Save Button */}
            <div className="mt-8 pt-6 border-t border-gray-200">
              <div className="flex justify-end">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                >
                  {saving ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Saving...
                    </>
                  ) : (
                    'Save Settings'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;