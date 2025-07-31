import React from 'react';
import type { TierConfig, ConstraintConfig } from '../types';
import { constraintTiers } from '../data/constraintTiers';
import HelpTooltip from './HelpTooltip';

interface TierSelectorProps {
  currentConfig: ConstraintConfig;
  onTierSelect: (tierName: string) => void;
}

const TierSelector: React.FC<TierSelectorProps> = ({ currentConfig, onTierSelect }) => {
  const currentTierName = currentConfig.tier;
  const currentIndex = constraintTiers.findIndex(tier => tier.name === currentTierName);

  // Simple recommendation logic:
  // If current config exceeds T0, recommend T1. If T1, recommend T2, etc.
  // This is a placeholder and can be refined based on actual usage/performance data.
  const recommendedTier = constraintTiers.find(tier => {
    return (
      currentConfig.budgetLimitUSD > tier.budgetLimit ||
      currentConfig.tokenLimit > tier.tokenLimit ||
      currentConfig.rateLimitPerMinute > tier.rateLimit ||
      currentConfig.memoryLimitMB > tier.memoryLimit
    );
  }) || constraintTiers[currentIndex + 1]; // Recommend next tier if current config exceeds current tier
  
  const recommendedTierName = recommendedTier ? recommendedTier.name : 'None';


  return (
    <div className="p-4 bg-white shadow rounded-lg">
      <h2 className="text-xl font-semibold mb-4">Tier Management Interface</h2>
      <p className="mb-4 text-gray-700">
        Compare the available tiers and select the one that best suits your simulation needs.
        <HelpTooltip content="Different tiers offer varying resource limits and features. Choose a tier that aligns with the scale and complexity of your experiments." />
      </p>

      <div className="overflow-x-auto mb-6">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Tier Name
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Budget Limit ($)
                <HelpTooltip content="Maximum estimated cost in USD for simulations within this tier." />
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Token Limit
                <HelpTooltip content="Maximum total tokens allowed per simulation run." />
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Rate Limit (calls/min)
                <HelpTooltip content="Maximum API calls per minute to external LLMs." />
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Memory Limit (MB)
                <HelpTooltip content="Maximum memory allocated per agent during simulation." />
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Features Available
                <HelpTooltip content="Key features unlocked or enhanced at this tier level." />
              </th>
              <th scope="col" className="relative px-6 py-3">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {constraintTiers.map((tier) => (
              <tr key={tier.name} className={tier.name === currentTierName ? 'bg-indigo-50' : ''}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {tier.name} {tier.name === currentTierName && <span className="text-indigo-600 font-semibold">(Current)</span>}
                  {tier.name === recommendedTierName && <span className="ml-2 text-green-600 font-semibold">(Recommended)</span>}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  ${tier.budgetLimit.toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {tier.tokenLimit.toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {tier.rateLimit.toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {tier.memoryLimit.toLocaleString()}
                </td>
                <td className="px-6 py-4 text-sm text-gray-500">
                  <ul className="list-disc list-inside">
                    {tier.features.map((feature, idx) => (
                      <li key={idx}>{feature}</li>
                    ))}
                  </ul>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  {tier.name !== currentTierName ? (
                    <button
                      onClick={() => onTierSelect(tier.name as TierConfig['name'])}
                      className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      {currentIndex > constraintTiers.indexOf(tier) ? 'Downgrade' : 'Upgrade'}
                    </button>
                  ) : (
                    <span className="text-gray-400">Selected</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-sm text-gray-600">
        <span className="font-semibold">Note on Recommendation:</span> The recommended tier is a suggestion based on your current configuration settings compared to tier limits. For optimal performance and cost, always review your simulation needs.
      </p>
    </div>
  );
};

export default TierSelector;