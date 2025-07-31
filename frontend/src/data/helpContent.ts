// Help content for all configuration forms in FBA-Bench
// Provides business-friendly explanations with practical examples and recommended values

export interface HelpContentItem {
  title?: string;
  content: string;
}

export interface HelpContent {
  simulation: Record<string, HelpContentItem>;
  agent: Record<string, HelpContentItem>;
  experiment: Record<string, HelpContentItem>;
  constraints: Record<string, HelpContentItem>; // New category
}

export const helpContent: HelpContent = {
  simulation: {
    simulationName: {
      title: "Simulation Name",
      content: "A unique identifier for your simulation run. Use descriptive names like 'PricingStrategy_Q4_2024' or 'InventoryOptimization_Test'. This helps you identify and compare different simulation runs in your results."
    },
    description: {
      title: "Simulation Description", 
      content: "Detailed explanation of what this simulation aims to test or achieve. Include your hypothesis, key variables, and expected outcomes. Example: 'Testing the impact of dynamic pricing on profit margins during peak season with 20% demand increase.'"
    },
    duration: {
      title: "Duration (Ticks)",
      content: "How many simulation steps to run. Each tick represents one time interval. Typical ranges: 1,440 ticks (24 hours with 1-minute intervals), 720 ticks (12 hours with 1-minute intervals). Longer simulations provide more data but take more computational time."
    },
    duration_hours: {
      title: "Duration (Hours)",
      content: "How long the simulation runs in simulated time. Typical ranges: 24-168 hours (1-7 days). Longer simulations capture more market dynamics and seasonal patterns but require more computational resources. Start with 24 hours for initial testing."
    },
    tick_interval_seconds: {
      title: "Tick Interval",
      content: "How often the simulation updates in simulated seconds. Lower values = higher precision but slower performance. Recommended: 60 seconds for most experiments, 300 seconds (5 minutes) for quick tests, 10-30 seconds for high-frequency trading analysis. Values below 10 seconds may significantly impact simulation speed."
    },
    max_ticks: {
      title: "Maximum Ticks",
      content: "Safety limit to prevent runaway simulations. Set this higher than your expected duration to allow for buffer. Example: For a 24-hour simulation with 60-second intervals, set max_ticks to 1,500+ (1,440 ticks needed + buffer)."
    },
    time_acceleration: {
      title: "Time Acceleration",
      content: "Speed multiplier for simulation time vs real time. 1.0 = real-time, 10.0 = 10x faster than real-time. Higher values complete simulations faster but may stress your system. Recommended: 5.0-20.0 for most experiments, 1.0-2.0 for detailed analysis."
    },
    initial_price: {
      title: "Initial Price",
      content: "Starting price for your product in dollars. This should reflect your actual Amazon listing price or target price point. Example: $29.99 for consumer electronics, $149.99 for premium items. This affects agent pricing decisions and profit calculations."
    },
    cost_basis: {
      title: "Cost Basis",
      content: "Your cost to acquire/manufacture the product, including shipping to Amazon warehouses. Used to calculate profit margins. Should be lower than initial_price. Example: If selling at $29.99, cost_basis might be $15.00-20.00 for healthy margins."
    },
    initial_inventory: {
      title: "Initial Inventory",
      content: "Units available in Amazon warehouses at simulation start. Represents your FBA inventory levels. Typical ranges: 100-1000 units for new products, 500-5000 for established products. Affects stockout risk and storage fees in the simulation."
    },
    randomSeed: {
      title: "Random Seed",
      content: "Controls randomness in the simulation for reproducibility. Use the same seed to get identical results across runs. Leave empty for random behavior. Use specific numbers (e.g., 12345) when comparing different strategies or debugging. Essential for scientific reproducibility."
    },
    seed: {
      title: "Random Seed",
      content: "Controls randomness in the simulation for reproducibility. Use the same seed to get identical results across runs. Leave empty for random behavior. Use specific numbers (e.g., 12345) when comparing different strategies or debugging. Essential for scientific reproducibility."
    },
    metricsInterval: {
      title: "Metrics Collection Interval",
      content: "How often to collect and record performance metrics (in ticks). Lower values provide more granular data but increase storage requirements. Recommended: Every 10-60 ticks. For 60-second tick intervals, collecting every 10 ticks gives you metrics every 10 minutes."
    },
    snapshotInterval: {
      title: "Snapshot Interval",
      content: "How often to save complete simulation state (in ticks). Snapshots enable resuming simulations and detailed analysis but consume significant storage. Recommended: Every 60-300 ticks. For debugging, use smaller intervals (10-30 ticks)."
    }
  },

  agent: {
    framework: {
      title: "Agent Framework",
      content: "Choose how agents are built and managed:\n\n• **DIY**: Custom-coded agents with full control over logic and behavior. Best for specific business rules and custom strategies.\n\n• **CrewAI**: Team-based agents that collaborate on decisions. Ideal for complex scenarios requiring multiple perspectives (pricing, inventory, marketing).\n\n• **LangChain**: Chain-based agents with structured workflows. Good for sequential decision-making and integration with external data sources."
    },
    agentType: {
      title: "Agent Type",
      content: "Specific implementation within the chosen framework. Each type has different capabilities and behaviors:\n\n• **pricing_agent**: Focuses on dynamic pricing decisions\n• **inventory_agent**: Manages stock levels and reordering\n• **marketing_agent**: Handles advertising and promotions\n• **competitive_agent**: Monitors and responds to competitor actions\n\nChoose based on the primary decision-making focus of your simulation."
    },
    llmProvider: {
      title: "LLM Provider",
      content: "The AI service powering agent decisions. Options include:\n\n• **OpenRouter**: Access to multiple models (GPT-4, Claude, etc.)\n• **OpenAI**: Direct access to GPT models\n• **Anthropic**: Claude models for reasoning-heavy tasks\n• **Local**: Self-hosted models for privacy/cost control\n\nOpenRouter is recommended for flexibility and cost optimization."
    },
    llmModel: {
      title: "LLM Model",
      content: "Specific AI model for decision-making. Performance vs cost considerations:\n\n• **GPT-4**: Highest reasoning quality, most expensive\n• **GPT-3.5-turbo**: Good balance of performance and cost\n• **Claude-3-haiku**: Fast and cost-effective for simple decisions\n• **Claude-3-sonnet**: Better reasoning for complex scenarios\n\nStart with GPT-3.5-turbo for initial testing, upgrade to GPT-4 for production scenarios."
    },
    temperature: {
      title: "Temperature",
      content: "Controls AI creativity and randomness in decisions (0.0 to 1.0):\n\n• **0.0-0.3**: Deterministic, conservative decisions. Good for pricing and inventory management.\n• **0.4-0.7**: Balanced creativity and consistency. Recommended for most business scenarios.\n• **0.8-1.0**: High creativity but inconsistent. Use for brainstorming or experimental strategies.\n\nRecommended: 0.3 for financial decisions, 0.7 for marketing strategies."
    },
    max_tokens: {
      title: "Max Tokens",
      content: "Maximum length of AI responses in tokens (~4 characters). Controls response detail and cost:\n\n• **500-1000**: Short, focused decisions. Cost-effective for simple scenarios.\n• **1000-2000**: Detailed reasoning with explanations. Good for complex business logic.\n• **2000+**: Comprehensive analysis. Use for strategic planning or debugging.\n\nRecommended: 1000 for most scenarios, 500 for high-frequency decisions."
    },
    top_p: {
      title: "Top P (Nucleus Sampling)",
      content: "Alternative creativity control to temperature. Controls the diversity of token selection (0.0 to 1.0):\n\n• **0.1-0.5**: Conservative, focused responses\n• **0.6-0.9**: Balanced diversity (recommended range)\n• **0.95-1.0**: Maximum diversity\n\nMost users should leave this at default (0.9) and adjust temperature instead. Use lower values (0.1-0.3) for financial calculations where precision is critical."
    },
    memoryType: {
      title: "Memory Configuration",
      content: "How agents remember and learn from past decisions:\n\n• **short_term**: Remember only recent decisions (last 10-50 actions). Fast and efficient.\n• **long_term**: Persistent memory across entire simulation. Better learning but slower.\n• **episodic**: Remember specific events and outcomes. Good for pattern recognition.\n• **none**: No memory between decisions. Fastest but agents don't learn.\n\nRecommended: short_term for most scenarios, long_term for strategic simulations."
    },
    role: {
      title: "Agent Role",
      content: "The agent's primary responsibility and decision-making focus. This defines what aspects of the business the agent will prioritize and how it interprets market conditions. Examples: 'Pricing Strategist', 'Inventory Manager', 'Marketing Coordinator'."
    },
    behavior: {
      title: "Agent Behavior",
      content: "Specific behavioral traits and decision-making style. This shapes how the agent approaches problems and makes trade-offs. Examples: 'Conservative pricing with focus on margin protection', 'Aggressive growth-oriented strategy', 'Data-driven decisions with risk management'."
    }
  },

  experiment: {
    experiment_name: {
      title: "Experiment Name",
      content: "Unique identifier for this parameter sweep experiment. Use descriptive names that indicate what you're testing. Examples: 'PricingStrategy_Comparison_2024', 'InventoryLevel_Optimization', 'SeasonalDemand_Analysis'. Helps organize and compare multiple experiments."
    },
    description: {
      title: "Experiment Description",
      content: "Detailed explanation of your experiment's hypothesis, methodology, and expected outcomes. Include what business question you're trying to answer and how the results will inform decisions. Example: 'Testing optimal inventory levels to minimize stockouts while reducing storage costs during Q4 peak season.'"
    },
    parameter_sweep: {
      title: "Parameter Sweep Configuration",
      content: "Define which simulation parameters to vary across experiment runs. Each parameter gets multiple values to test systematically. Example: initial_price: [24.99, 29.99, 34.99] will run simulations at each price point. This enables statistical analysis of parameter impact on business outcomes."
    },
    parallel_workers: {
      title: "Parallel Workers",
      content: "Number of simulations to run simultaneously. More workers = faster completion but higher resource usage. Recommended: 2-4 workers for desktop computers, 8-16 for servers. Consider your CPU cores and available RAM. Each worker needs ~1-2GB RAM for typical simulations."
    },
    max_runs: {
      title: "Maximum Runs",
      content: "Optional limit on total simulation runs for testing. Useful when experimenting with large parameter sweeps. Leave empty for full parameter sweep. Example: Set to 10 to test your experiment setup quickly before running the full 100+ parameter combinations."
    },
    save_events: {
      title: "Save Full Event Stream",
      content: "Whether to record every event during simulations. Provides complete audit trail and enables detailed analysis but significantly increases storage requirements. Enable for debugging or when you need to analyze specific decision sequences. Disable for large-scale experiments to save space."
    },
    save_snapshots: {
      title: "Save Periodic Snapshots",
      content: "Whether to save complete simulation state at regular intervals. Enables resuming interrupted simulations and analyzing state evolution over time. Requires substantial storage (10-100MB per snapshot). Recommended for long-running experiments or when simulation stability is a concern."
    },
    snapshot_interval_hours: {
      title: "Snapshot Interval (Hours)",
      content: "How often to save simulation snapshots in simulated time. Shorter intervals provide more recovery points but use more storage. Recommended: 4-12 hours for most experiments, 1-2 hours for critical production scenarios. Balance between data granularity and storage costs."
    },
    statistical_significance: {
      title: "Statistical Significance",
      content: "Parameter sweeps automatically calculate statistical significance of results. Experiments with multiple parameter values enable confidence testing of performance differences. Minimum 3 runs per parameter combination recommended for basic statistics, 10+ runs for robust statistical analysis."
    },
    output_directory: {
      title: "Output Directory",
      content: "Where experiment results are saved. Default is 'experiments/[experiment_name]_[timestamp]'. Each run creates subdirectories with metrics, logs, and snapshots. Ensure sufficient disk space - experiments can generate 1-10GB of data depending on duration and snapshot settings."
    }
  },

  constraints: {
    tierSelection: {
      title: "Tier Selection",
      content: "Different tiers (T0, T1, T2, T3) offer varying levels of resources and features. T0 is for basic testing, while higher tiers provide larger budgets, more tokens, faster API rates, and greater memory for complex and large-scale simulations. Choose a tier that matches your expected resource consumption."
    },
    budgetLimitUSD: {
      title: "Budget Limit (USD)",
      content: "Set a cap on the estimated monetary cost of your simulation. This is calculated based on token usage and LLM costs. Exceeding this limit will trigger warnings or stop the simulation. Estimate your needs based on simulation duration, number of agents, and expected LLM interactions. For example, a budget of $5-10 for small tests, $50-100 for medium experiments."
    },
    tokenLimit: {
      title: "Token Limit (Max tokens per simulation)",
      content: "The maximum number of tokens (pieces of text) your simulation can process. Higher token limits allow for more detailed agent responses, longer agent memories, and more complex prompts, which can improve simulation accuracy but lead to higher costs. Ensure your token limit aligns with the verbosity and complexity of your agents' communication."
    },
    rateLimitPerMinute: {
      title: "Rate Limit (API calls per minute)",
      content: "Controls the maximum number of requests your simulation can send to external LLM APIs per minute. Setting an appropriate rate limit prevents hitting API provider limits and ensures smooth simulation execution. Lower limits prolong simulation time but reduce risk of rate limiting. Balance this with your required simulation speed."
    },
    memoryLimitMB: {
      title: "Memory Constraint (Agent memory size in MB)",
      content: "Defines the maximum memory, in megabytes, allocated to each individual agent within the simulation. More memory allows agents to retain more context, learn from longer histories, and process more complex information, potentially leading to more sophisticated behaviors and better performance. This directly impacts the scale and intelligence of your agents."
    }
  }
};

// Helper function to get help content for a specific field
export function getHelpContent(category: keyof HelpContent, field: string): HelpContentItem | undefined {
  return helpContent[category]?.[field];
}

// Helper function to get just the content string (for simple usage)
export function getHelpText(category: keyof HelpContent, field: string): string {
  return helpContent[category]?.[field]?.content || '';
}