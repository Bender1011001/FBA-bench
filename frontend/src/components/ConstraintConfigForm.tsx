import React, { useState, useEffect, useCallback } from 'react';
import type { ConstraintConfig, TierConfig } from '../types';
import { constraintTiers, defaultConstraintConfig } from '../data/constraintTiers';
import HelpTooltip from './HelpTooltip'; // Assuming HelpTooltip is in the same directory or adjust path

interface ConstraintConfigFormProps {
  initialConfig?: ConstraintConfig;
  onConfigChange: (config: ConstraintConfig, isValid: boolean) => void;
}

const ConstraintConfigForm: React.FC<ConstraintConfigFormProps> = ({
  initialConfig,
  onConfigChange,
}) => {
  const [config, setConfig] = useState<ConstraintConfig>(
    initialConfig || {
      ...defaultConstraintConfig,
      tier: defaultConstraintConfig.tier as ConstraintConfig['tier'],
    }
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [warnings, setWarnings] = useState<Record<string, string>>({});

  const selectedTier = constraintTiers.find(tier => tier.name === config.tier) || constraintTiers[0];

  const validate = useCallback(() => {
    const newErrors: Record<string, string> = {};
    const newWarnings: Record<string, string> = {};
    let isValid = true;

    // Budget Limit validation
    if (config.budgetLimitUSD <= 0) {
      newErrors.budgetLimitUSD = 'Budget limit must be a positive number.';
      isValid = false;
    } else if (config.budgetLimitUSD > selectedTier.budgetLimit) {
      newWarnings.budgetLimitUSD = `Budget limit exceeds selected tier (${selectedTier.name}) limit of $${selectedTier.budgetLimit}.`;
    }

    // Token Limit validation
    if (config.tokenLimit <= 0 || !Number.isInteger(config.tokenLimit)) {
      newErrors.tokenLimit = 'Token limit must be an integer greater than 0.';
      isValid = false;
    } else if (config.tokenLimit > selectedTier.tokenLimit) {
      newWarnings.tokenLimit = `Token limit exceeds selected tier (${selectedTier.name}) limit of ${selectedTier.tokenLimit} tokens.`;
    }

    // Rate Limit validation
    if (config.rateLimitPerMinute < 1 || config.rateLimitPerMinute > 1000 || !Number.isInteger(config.rateLimitPerMinute)) {
      newErrors.rateLimitPerMinute = 'Rate limit must be an integer between 1 and 1000 calls per minute.';
      isValid = false;
    } else if (config.rateLimitPerMinute > selectedTier.rateLimit) {
      newWarnings.rateLimitPerMinute = `Rate limit exceeds selected tier (${selectedTier.name}) limit of ${selectedTier.rateLimit} calls/min.`;
    }

    // Memory Constraint validation
    if (config.memoryLimitMB < 1 || config.memoryLimitMB > 1024 || !Number.isInteger(config.memoryLimitMB)) {
      newErrors.memoryLimitMB = 'Memory limit must be an integer between 1 and 1024 MB.';
      isValid = false;
    } else if (config.memoryLimitMB > selectedTier.memoryLimit) {
      newWarnings.memoryLimitMB = `Memory limit exceeds selected tier (${selectedTier.name}) limit of ${selectedTier.memoryLimit} MB.`;
    }

    setErrors(newErrors);
    setWarnings(newWarnings);
    return isValid && Object.keys(newWarnings).length === 0; // Consider warnings as invalid for full submission, but show them
  }, [config, selectedTier]);

  useEffect(() => {
    const isValid = validate();
    onConfigChange(config, isValid);
  }, [config, onConfigChange, validate]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    let newValue: string | number = value;

    if (['budgetLimitUSD', 'tokenLimit', 'rateLimitPerMinute', 'memoryLimitMB'].includes(name)) {
      newValue = parseFloat(value);
      if (isNaN(newValue)) {
        newValue = 0; // Or handle as empty string for immediate validation feedback
      }
    }

    setConfig(prev => {
      const updatedConfig = { ...prev, [name]: newValue };
      if (name === 'tier') {
        const newTier = constraintTiers.find(t => t.name === (value as 'T0' | 'T1' | 'T2' | 'T3'));
        if (newTier) {
          // When tier changes, suggest updating limits to match tier defaults
          updatedConfig.budgetLimitUSD = newTier.budgetLimit;
          updatedConfig.tokenLimit = newTier.tokenLimit;
          updatedConfig.rateLimitPerMinute = newTier.rateLimit;
          updatedConfig.memoryLimitMB = newTier.memoryLimit;
        }
      }
      return updatedConfig;
    });
  };

  const InputField: React.FC<{
    id: string;
    label: string;
    type: string;
    value: number;
    name: string;
    min?: number;
    max?: number;
    helpText: string;
  }> = ({ id, label, type, value, name, min, max, helpText }) => (
    <div className="mb-4">
      <label htmlFor={id} className="block text-sm font-medium text-gray-700">
        {label}
        <HelpTooltip content={helpText} />
      </label>
      <input
        type={type}
        id={id}
        name={name}
        value={value}
        onChange={handleChange}
        min={min}
        max={max}
        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
      />
      {errors[name] && <p className="mt-1 text-sm text-red-600">{errors[name]}</p>}
      {warnings[name] && <p className="mt-1 text-sm text-yellow-600">{warnings[name]}</p>}
    </div>
  );

  return (
    <div className="p-4 bg-white shadow rounded-lg">
      <h2 className="text-xl font-semibold mb-4">Constraint Configuration</h2>

      <div className="mb-4">
        <label htmlFor="tier" className="block text-sm font-medium text-gray-700">
          Tier Selection
          <HelpTooltip content="Select a pre-defined tier to set baseline limits for your simulation. You can customize limits after selecting a tier." />
        </label>
        <select
          id="tier"
          name="tier"
          value={config.tier}
          onChange={handleChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        >
          {constraintTiers.map(tier => (
            <option key={tier.name} value={tier.name}>
              {tier.name} - {tier.description}
            </option>
          ))}
        </select>
        <p className="mt-2 text-sm text-gray-500">
          Selected Tier Limits: Budget: ${selectedTier.budgetLimit}, Tokens: {selectedTier.tokenLimit},
          Rate: {selectedTier.rateLimit}/min, Memory: {selectedTier.memoryLimit}MB
        </p>
      </div>

      <InputField
        id="budgetLimitUSD"
        label="Budget Limit (USD)"
        type="number"
        name="budgetLimitUSD"
        value={config.budgetLimitUSD}
        min={0}
        helpText="Set the maximum budget (in USD) for your simulation. Exceeding this limit will trigger warnings or stop the simulation."
      />
      <InputField
        id="tokenLimit"
        label="Token Limit (Max tokens per simulation)"
        type="number"
        name="tokenLimit"
        value={config.tokenLimit}
        min={1}
        helpText="Define the maximum number of tokens your simulation can consume. High token usage can lead to higher costs and longer simulation times."
      />
      <InputField
        id="rateLimitPerMinute"
        label="Rate Limit (API calls per minute)"
        type="number"
        name="rateLimitPerMinute"
        value={config.rateLimitPerMinute}
        min={1}
        max={1000}
        helpText="Control the maximum number of API calls per minute to prevent hitting provider rate limits. Lower values can slow down simulation."
      />
      <InputField
        id="memoryLimitMB"
        label="Memory Constraint (Agent memory size in MB)"
        type="number"
        name="memoryLimitMB"
        value={config.memoryLimitMB}
        min={1}
        max={1024}
        helpText="Specify the maximum memory allocated per agent in megabytes. Higher memory can improve agent performance but consumes more resources."
      />
    </div>
  );
};

export default ConstraintConfigForm;