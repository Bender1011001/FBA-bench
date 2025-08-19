import React, { useEffect, useState } from 'react';
import Modal from '../ui/Modal';
import Spinner from '../ui/Spinner';
import type { Agent } from '../../types/api';

export interface AgentFormValues {
  name: string;
  runner: string;
  description?: string;
  config?: Record<string, unknown>;
}

export interface AgentFormProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (values: AgentFormValues) => Promise<void> | void;
  initial?: Partial<Agent>;
  title?: string;
}

const AgentForm: React.FC<AgentFormProps> = ({ open, onClose, onSubmit, initial, title }) => {
  const [name, setName] = useState(initial?.name ?? '');
  const [runner, setRunner] = useState(initial?.runner ?? '');
  const [description, setDescription] = useState(initial?.description ?? '');
  const [configText, setConfigText] = useState<string>(() => {
    if (initial?.config && typeof initial.config === 'object') {
      try {
        return JSON.stringify(initial.config, null, 2);
      } catch {
        return '';
      }
    }
    return '';
  });

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // reset fields when opening with new initial
    if (open) {
      setName(initial?.name ?? '');
      setRunner(initial?.runner ?? '');
      setDescription(initial?.description ?? '');
      setConfigText(
        initial?.config && typeof initial.config === 'object' ? JSON.stringify(initial.config, null, 2) : ''
      );
      setError(null);
      setSubmitting(false);
    }
  }, [open, initial?.name, initial?.runner, initial?.description, initial?.config]);

  const parseConfig = (text: string): Record<string, unknown> | undefined => {
    const trimmed = text.trim();
    if (!trimmed) return undefined;
    try {
      const obj = JSON.parse(trimmed) as unknown;
      if (obj && typeof obj === 'object') {
        return obj as Record<string, unknown>;
      }
      setError('Config must be a JSON object.');
      return undefined;
    } catch {
      setError('Invalid JSON in config.');
      return undefined;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Name is required.');
      return;
    }
    if (!runner.trim()) {
      setError('Runner is required.');
      return;
    }

    const cfg = parseConfig(configText);
    if (configText.trim() && !cfg) {
      return;
    }

    try {
      setSubmitting(true);
      await onSubmit({
        name: name.trim(),
        runner: runner.trim(),
        description: description.trim() || undefined,
        config: cfg,
      });
      setSubmitting(false);
      onClose();
    } catch (err) {
      setSubmitting(false);
      setError(err instanceof Error ? err.message : 'Failed to submit the form.');
    }
  };

  return (
    <Modal
      open={open}
      onClose={() => {
        if (!submitting) onClose();
      }}
      title={title ?? (initial?.id ? 'Edit Agent' : 'Create Agent')}
    >
      <form onSubmit={handleSubmit} className="space-y-4" aria-label="Agent Form">
        <div>
          <label htmlFor="agent-name" className="block text-sm font-medium text-gray-700">
            Name <span className="text-red-600" aria-hidden="true">*</span>
          </label>
          <input
            id="agent-name"
            name="name"
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </div>

        <div>
          <label htmlFor="agent-runner" className="block text-sm font-medium text-gray-700">
            Runner <span className="text-red-600" aria-hidden="true">*</span>
          </label>
          <input
            id="agent-runner"
            name="runner"
            type="text"
            required
            value={runner}
            onChange={(e) => setRunner(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            placeholder="e.g., python, node, docker-image, custom"
          />
        </div>

        <div>
          <label htmlFor="agent-description" className="block text-sm font-medium text-gray-700">
            Description
          </label>
          <textarea
            id="agent-description"
            name="description"
            rows={3}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </div>

        <div>
          <div className="flex items-center justify-between">
            <label htmlFor="agent-config" className="block text-sm font-medium text-gray-700">
              Config (JSON)
            </label>
            <button
              type="button"
              className="text-xs text-blue-700 hover:underline"
              onClick={() => setConfigText('{\n  \n}')}
              aria-label="Insert JSON template"
            >
              Insert template
            </button>
          </div>
          <textarea
            id="agent-config"
            name="config"
            rows={6}
            value={configText}
            onChange={(e) => setConfigText(e.target.value)}
            placeholder='{"param": "value"}'
            className="mt-1 font-mono text-sm block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
          <p className="mt-1 text-xs text-gray-500">Leave blank for no config.</p>
        </div>

        {error ? (
          <div role="alert" className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
            {error}
          </div>
        ) : null}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="inline-flex items-center rounded-md border px-3 py-2 text-sm hover:bg-gray-50 focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting}
            className="inline-flex items-center rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            {submitting ? <Spinner size="sm" label="Saving…" /> : 'Save'}
          </button>
        </div>
      </form>
    </Modal>
  );
};

export default AgentForm;