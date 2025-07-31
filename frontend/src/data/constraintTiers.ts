import type { TierConfig } from '../types';

export const constraintTiers: TierConfig[] = [
  {
    name: 'T0',
    budgetLimit: 7.5, // (250000 / 1000) * 0.03 = 7.5 USD
    tokenLimit: 250000,
    rateLimit: 10, // Placeholder
    memoryLimit: 128, // Placeholder
    features: ['Basic Simulations', 'Limited Analysis'],
    description: 'Entry-level tier for small, basic simulations. Ideal for quick tests and concept validation.'
  },
  {
    name: 'T1',
    budgetLimit: 30, // (500000 / 1000) * 0.06 = 30 USD
    tokenLimit: 500000,
    rateLimit: 50, // Placeholder
    memoryLimit: 256, // Placeholder
    features: ['Standard Simulations', 'Basic Reporting', 'Token Efficiency Tracking'],
    description: 'Standard tier for medium-sized simulations with more detailed reporting and cost tracking.'
  },
  {
    name: 'T2',
    budgetLimit: 120, // (1000000 / 1000) * 0.12 = 120 USD
    tokenLimit: 1000000,
    rateLimit: 100, // Placeholder
    memoryLimit: 512, // Placeholder
    features: ['Advanced Simulations', 'Comprehensive Analysis', 'Multi-Agent Support'],
    description: 'Advanced tier for complex simulations, higher agent count, and in-depth analytical capabilities.'
  },
  {
    name: 'T3',
    budgetLimit: 1000, // (5000000 / 1000) * 0.20 = 1000 USD
    tokenLimit: 5000000,
    rateLimit: 200, // Placeholder
    memoryLimit: 1024, // Placeholder
    features: ['Enterprise-Grade Simulations', 'Custom Integrations', 'Dedicated Support', 'Large-Scale Data Handling'],
    description: 'Premium tier designed for enterprise-level simulations requiring extensive resources and custom solution integration.'
  }
];

export const defaultConstraintConfig = {
  tier: 'T0',
  budgetLimitUSD: constraintTiers[0].budgetLimit,
  tokenLimit: constraintTiers[0].tokenLimit,
  rateLimitPerMinute: constraintTiers[0].rateLimit,
  memoryLimitMB: constraintTiers[0].memoryLimit,
};