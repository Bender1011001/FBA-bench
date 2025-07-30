import React from 'react';
import type { Template, Configuration } from '../types';
import { predefinedTemplates } from '../data/templates';

interface TemplateSelectionFormProps {
  onSelectTemplate: (config: Configuration) => void;
  onNext: () => void;
}

const TemplateSelectionForm: React.FC<TemplateSelectionFormProps> = ({ onSelectTemplate, onNext }) => {
  const handleSelectTemplate = (template: Template) => {
    onSelectTemplate(template.configuration);
    onNext();
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold text-gray-900">Choose a Configuration Template</h2>
      <p className="text-gray-600">Select a predefined template to quickly set up your simulation, or choose 'Custom Configuration' to start from scratch.</p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {predefinedTemplates.map((template) => (
          <div
            key={template.id}
            className="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer p-6 flex flex-col"
            onClick={() => handleSelectTemplate(template)}
          >
            <h3 className="text-xl font-medium text-gray-900 mb-2">{template.name}</h3>
            <p className="text-gray-700 text-sm mb-4 flex-grow">{template.description}</p>
            <div className="mt-auto">
              <span className="inline-flex items-center rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-700/10">
                Use Case: {template.useCase}
              </span>
              <button
                type="button"
                className="mt-4 w-full inline-flex justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-base font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 sm:text-sm"
                onClick={(e) => {
                  e.stopPropagation(); // Prevent card onClick from firing
                  handleSelectTemplate(template);
                }}
              >
                Select Template
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TemplateSelectionForm;