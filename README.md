# Multi-Prompt Agent

## Thought Process

### Understanding the Problem
I began by analyzing Retel AI's Multi-Prompt Agent architecture, which implements a prompt tree structure for conversational AI systems. Using a healthcare agent as an example, the system works as follows:

1. **Initial Contact**: The first agent greets the customer and identifies the reason for their call
2. **Intent-Based Routing**: Based on the identified intent, the agent has multiple transition options
3. **Dynamic Flow**: A prompt tree structure guides the conversation flow, with agents choosing paths based on call progression

### Design Approach

#### 1. Schema Design
My first step was designing a comprehensive schema for the prompt tree structure. This involved:
- Defining how different conversation stages interact with each other
- Establishing clear transition mechanisms between stages
- Creating a flexible structure that supports various conversation flows

#### 2. LiveKit Research and Analysis
To understand the implementation approach, I studied how LiveKit handles different prompts and agent management:

- **LiveKit Documentation Study**: Researched LiveKit's official documentation and examples
- **Agent Architecture Discovery**: Found that in newer LiveKit versions, agents are treated as separate classes
- **Multi-Agent Sessions**: Learned that multiple agents can exist within the same agent session
- **Reference Implementation**: Analyzed the [medical office triage example](https://github.com/livekit-examples/python-agents-examples/blob/main/complex-agents/medical_office_triage/triage.py) to understand complex agent interactions

#### 3. Implementation Strategy
Based on my LiveKit research, I developed the following approach:

**Dynamic Agent Architecture**: Instead of creating multiple static agent classes, I designed a single dynamic agent that adapts based on JSON flow configuration. This approach provides:
- **Flexibility**: One agent class that changes behavior dynamically
- **Scalability**: Easy to add new conversation flows without code changes
- **Maintainability**: Centralized agent logic with configuration-driven behavior

**UserData Model Design**: Created a comprehensive UserData model to maintain agent context, including:
- Current stage information and configuration
- Complete stage definitions and flow logic
- Conversation history and context preservation
- Stage-specific tools and capabilities

**Core Implementation Components**:
- JSON-driven prompt tree parsing and execution
- Dynamic stage transition logic
- Context-aware conversation flow management

#### 4. Function Tools Implementation
To enable intelligent stage transitions, I implemented several LiveKit function tools that provide the LLM with the necessary context and capabilities:

**Stage Transition Tools**: Created function tools that help determine and execute stage transitions:
- `move_to_next_stage()`: Directly move to a specified target stage
- `complete_current_stage()`: Automatically determine the next stage based on completion reason
- `end_conversation()`: Handle conversation termination with appropriate transitions

**Context-Aware Decision Making**: Each tool receives comprehensive context including:
- Current stage prompt and completion criteria
- Available next stages with their conditions and priorities
- User data and conversation context
- Stage-specific transition rules

This approach allows the LLM to make informed decisions about stage transitions by understanding the current stage requirements and available options, rather than relying on simple keyword matching.

**Priority-Based Stage Evaluation**: Implemented a priority system for stage transitions where stages with higher priority values are evaluated first. This ensures that more specific or important transitions take precedence over general fallback options.

#### 5. Concrete Example: Medical Office Flow
Here's how the system works in practice with a medical office scenario:

**Initial Stage**: `reason_for_call` (Priority-based transitions)
- Priority 80: `appointment_assistance` - "Patient needs help with appointments"
- Priority 80: `billing_assistance` - "Patient has billing, insurance, or payment questions"  
- Priority 60: `general_assistance` - "Patient needs general information"

**User Input**: "I need to schedule an appointment with Dr. Smith"

**Processing Flow**:
1. LLM receives current stage context with completion criteria and next stage options
2. System evaluates transitions in priority order (80 → 80 → 60)
3. First priority 80 condition matches: "appointments" keyword triggers `appointment_assistance`
4. `move_to_next_stage("appointment_assistance")` is called
5. Agent transitions to appointment assistance stage with specialized prompts and tools

This priority system ensures that specific requests (appointments, billing) are handled before general assistance, creating a more efficient and accurate conversation flow.

### Key Components
- **Prompt Tree Structure**: Hierarchical organization of conversation stages
- **Stage Transitions**: Logic for moving between different conversation states
- **Intent Recognition**: Mechanisms for understanding user input and determining next steps
- **Dynamic Flow Control**: Adaptive conversation paths based on context

## Assumptions Made

### Core Design Assumptions

#### 1. Single Dynamic Agent Architecture
**Assumption**: Instead of creating multiple separate agent classes for different scenarios, a single agent that dynamically handles all conversation flows based on JSON configuration would be more efficient and maintainable.

**Rationale**: 
- Avoiding the complexity of managing multiple agent instances
- Centralized logic for easier maintenance and updates
- Configuration-driven behavior allows for easy flow modifications without code changes
- Single agent can maintain consistent context and conversation history across all stages

#### 2. LLM Function Tool Reliability
**Assumption**: The LLM can reliably evaluate stage completion and determine appropriate next stages when provided with comprehensive context through function tools.

**Implementation**:
- Passing current user data, stage information, and available next stages to the LLM
- Trusting the LLM to make intelligent decisions about stage transitions based on conversation context
- Using function calling (`move_to_next_stage`, `complete_current_stage`) as the primary mechanism for flow control
- Relying on the LLM's natural language understanding to interpret completion criteria and transition conditions

These assumptions formed the foundation of the multi-prompt agent architecture, enabling a flexible and scalable conversation management system.

## Testing and Extension

### How to Test the Code

#### Environment Setup

1. **Install Dependencies using UV**:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Setup the virtual environment and install dependencies
   uv sync
   ```

2. **Configure Environment Variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your actual API keys
   # Required keys: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, 
   # OPENAI_API_KEY, DEEPGRAM_API_KEY
   ```

#### Testing the Agent

1. **Choose or Create Agent Configuration**:
   - Use existing config: `agents_configs/default.json` (medical office assistant)
   - Or create a new config file following the `schema.json` structure

2. **Run the Agent**:
   ```bash
   # Using the default configuration
   CONFIG_PATH=agents_configs/default.json python run_agent.py dev
   
   # Using a custom configuration
   CONFIG_PATH=path/to/your/config.json python run_agent.py dev
   ```

3. **Test Conversation Flows**:
   - **Appointment Flow**: "I need to schedule an appointment"
   - **Billing Flow**: "I have a question about my bill"
   - **General Info Flow**: "What are your office hours?"
   - **Transfer Scenarios**: Test human handoff requests

#### Configuration Validation

When creating your own JSON configuration files, validate them against the schema to ensure they're properly structured:

```bash
# Example validation of a custom config
python schema_validation.py my_custom_agent.json
```

**What it validates:**
- Required sections: global_settings, agent_config, flow
- Provider configurations (LLM, STT, TTS settings)
- Stage flow structure and stage definitions
- Stage reference integrity (all next_stages point to valid stages)

**Output:** Shows configuration summary and detailed error messages if validation fails.

### How to Extend the Code

#### 1. Enhanced Condition Evaluation

**Current**: Simple keyword matching for stage transitions (e.g., "appointments" → appointment_assistance)

**Extension**: Replace with LLM-based or semantic evaluation:
- **LLM Evaluation**: Let LLM determine if user input matches transition conditions semantically
- **NLP Models**: Use intent classification for better condition matching  
- **Semantic Similarity**: Handle variations like "book a visit" → "appointments"


#### 2. Stage-Specific Custom Tools

**Current Implementation**: The system uses dynamic prompts and stage-based evaluation through function tools.

**Extension**: Add functionality for stage-specific custom tools that will only available at certain stages. 

