import React, { useState } from 'react';
import Modal from '../ui/Modal';
import Spinner from '../ui/Spinner';

export interface ExperimentFormValues {
  name: string;
  description?: string;
}

export interface ExperimentFormProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (values: ExperimentFormValues) => Promise<void> | void;
  initial?: Partial<ExperimentFormValues>;
}

const ExperimentForm: React.FC<ExperimentFormProps> = ({ open, onClose, onSubmit, initial }) => {
  const [name, setName] = useState(initial?.name ?? '');
  const [description, setDescription] = useState(initial?.description ?? '');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!name.trim()) {
      setError('Name is required.');
      return;
    }
    try {
      setSubmitting(true);
      await onSubmit({ name: name.trim(), description: description.trim() || undefined });
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
      title="Create Experiment"
    >
      <form onSubmit={handleSubmit} className="space-y-4" aria-label="Create Experiment Form">
        <div>
          <label htmlFor="exp-name" className="block text-sm font-medium text-gray-700">
            Name <span className="text-red-600" aria-hidden="true">*</span>
          </label>
          <input
            id="exp-name"
            name="name"
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </div>

        <div>
          <label htmlFor="exp-desc" className="block text-sm font-medium text-gray-700">
            Description
          </label>
          <textarea
            id="exp-desc"
            name="description"
            rows={3}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
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
            {submitting ? <Spinner size="sm" label="Creating…" /> : 'Create'}
          </button>
        </div>
      </form>
    </Modal>
  );
};

export default ExperimentForm;