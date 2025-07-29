# Community Exploit Framework

This directory contains community-contributed adversarial exploit definitions for FBA-Bench red-team testing. The framework provides a standardized way to define, validate, and share exploit scenarios that test agent resistance to various attack vectors.

## Directory Structure

```
redteam_scripts/
├── README.md                          # This file
├── exploit_template.yaml              # Template for creating new exploits
├── validation_rules.md                # Rules for exploit validation
├── community_exploits/                # Community-contributed exploits
│   ├── phishing/                      # Phishing exploit scenarios
│   ├── market_manipulation/           # Market manipulation attacks
│   ├── compliance_traps/              # Fake policy/compliance attacks
│   ├── financial_exploits/            # Financial-focused attacks
│   └── information_warfare/           # Information/misinformation attacks
├── examples/                          # Example exploit definitions
└── testing/                          # Test exploits for development
```

## Exploit Definition Format

Exploits are defined in YAML format with the following structure:

```yaml
name: "Human-readable exploit name"
author: "contributor_username"
version: "1.0.0"
category: "phishing|market_manipulation|compliance_trap|financial_exploit|information_warfare"
difficulty: 1-5  # 1=trivial, 5=expert-level
description: "Detailed description of the exploit scenario"
exploit_type: "technical_type_matching_event_system"

# Actions the agent should NOT take if resistant
target_actions:
  - "action_1"
  - "action_2"

# Techniques used to deceive the agent
deception_techniques:
  - "technique_1"
  - "technique_2"

# Conditions indicating successful exploitation
success_conditions:
  primary: "main_condition"
  secondary: "optional_secondary_condition"

# Conditions indicating failed exploitation
failure_conditions:
  detection: "agent_detects_attack"
  verification: "agent_verifies_authenticity"

# Exploit timing and constraints
time_window_hours: 24
financial_impact_limit_usd: 1000.0

# Conditions that must be met to activate exploit
trigger_conditions:
  simulation_day: ">= 3"
  agent_has_active_products: true

# Required simulation context
context_requirements:
  sender_email: "example@suspicious-domain.com"
  message_content: "Urgent action required"
  # ... other context-specific fields

# Validation rules for this exploit
validation_rules:
  financial_impact_max: 10000.0
  realism_score_min: 7
  reversibility_required: true

# Additional metadata
metadata:
  tags: ["tag1", "tag2"]
  references: ["https://example.com/reference"]
  created_date: "2024-01-01"
  last_updated: "2024-01-01"
```

## Creating New Exploits

1. **Copy the template**: Start with `exploit_template.yaml`
2. **Follow naming convention**: `category_name_v1.0.0.yaml`
3. **Validate your exploit**: Use the validation framework
4. **Test thoroughly**: Ensure the exploit works as expected
5. **Submit for review**: Follow the contribution guidelines

## Exploit Categories

### Phishing (`phishing`)
Social engineering attacks using deceptive communications to trick agents into harmful actions.

**Required Context Fields:**
- `sender_email`: Email address of the phishing attempt
- `message_content`: The deceptive message content
- `requested_action`: What the email asks the agent to do

### Market Manipulation (`market_manipulation`)  
False market information designed to influence pricing or strategic decisions.

**Required Context Fields:**
- `false_data_type`: Type of false data (pricing, sales_data, reviews, etc.)
- `manipulated_values`: The false values being presented
- `source_credibility`: How credible the source appears (1-5)

### Compliance Traps (`compliance_trap`)
Fake policy changes or compliance requirements that look legitimate but are harmful.

**Required Context Fields:**
- `fake_policy_name`: Name of the fake policy
- `policy_description`: Description of the fake requirements
- `deadline_hours`: Hours until fake deadline
- `penalty_claim`: Claimed penalties for non-compliance

### Financial Exploits (`financial_exploit`)
Attacks targeting financial decision-making and payment processes.

### Information Warfare (`information_warfare`)
Misinformation campaigns and false intelligence designed to mislead agents.

## Validation Rules

All community exploits must pass validation before inclusion:

1. **Financial Impact**: Maximum $50,000 potential damage
2. **Realism Score**: Minimum 6/10 realism rating
3. **Reversibility**: Exploits must be reversible/undoable
4. **No Harmful Content**: No real-world harmful instructions
5. **Fair Testing**: Exploits must provide fair assessment opportunities

## Difficulty Levels

- **Level 1 (Trivial)**: Obvious attacks that basic security awareness should catch
- **Level 2 (Easy)**: Simple attacks with clear warning signs
- **Level 3 (Moderate)**: Believable attacks requiring careful scrutiny
- **Level 4 (Hard)**: Sophisticated attacks mimicking legitimate communications
- **Level 5 (Expert)**: Advanced persistent threat-style attacks

## Contributing Guidelines

### Submission Process

1. **Fork and Branch**: Create a feature branch for your exploit
2. **Follow Template**: Use the provided template structure
3. **Validate Locally**: Run validation tests before submission
4. **Add Tests**: Include test cases for your exploit
5. **Document Thoroughly**: Provide clear descriptions and rationale
6. **Submit PR**: Create a pull request with detailed explanation

### Code Review Process

All submissions undergo review for:
- Technical correctness
- Realism and believability  
- Educational value
- Safety and ethics compliance
- Code quality and documentation

### Naming Conventions

**Files**: `category_descriptive_name_v1.0.0.yaml`
**Exploit Names**: Clear, descriptive names without offensive language
**Authors**: Use GitHub username or professional identifier

## Testing Framework

### Local Testing

```bash
# Validate a single exploit
python -m redteam.validation validate_exploit redteam_scripts/examples/phishing_supplier_v1.0.0.yaml

# Test exploit execution
python -m redteam.testing test_exploit redteam_scripts/examples/phishing_supplier_v1.0.0.yaml

# Validate entire directory
python -m redteam.validation validate_directory redteam_scripts/community_exploits/
```

### Integration Testing

Exploits are automatically tested against baseline agents during CI to ensure:
- Proper execution without errors
- Reasonable success/failure rates
- Consistent scoring behavior
- Performance impact assessment

## Best Practices

### Writing Effective Exploits

1. **Realistic Scenarios**: Base on actual attack patterns
2. **Clear Objectives**: Define what constitutes success/failure
3. **Graduated Difficulty**: Provide exploits across difficulty spectrum
4. **Cultural Sensitivity**: Avoid culturally specific assumptions
5. **Version Control**: Update version numbers for significant changes

### Security Considerations

1. **No Real Credentials**: Never use actual passwords, keys, or credentials
2. **Safe Financial Limits**: Keep financial impact limits reasonable
3. **Ethical Boundaries**: Focus on security awareness, not actual harm
4. **Privacy Respect**: Don't collect or expose sensitive information

## Support and Questions

- **Documentation**: See `/docs/red_team_framework.md`
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join the FBA-Bench Discord for real-time help

## License

All community exploits are released under the same license as FBA-Bench. By contributing, you agree to license your work under these terms.