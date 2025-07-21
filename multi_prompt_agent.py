# Multi-Prompt Agent Implementation
# This module implements a dynamic, stage-based conversational AI agent using LiveKit
# The agent can handle complex conversation flows defined through JSON configuration

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# LiveKit imports for voice agent functionality
from livekit.agents import JobContext
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import cartesia, deepgram, openai, silero, elevenlabs

# Initialize logger for tracking agent behavior and debugging
logger = logging.getLogger("multi-prompt-agent-logger")

@dataclass
class UserData:
    """
    Data structure to maintain conversation state across the session
    This stores all the information needed to track user progress through stages
    """
    # Current stage the user is in (e.g., "reason_for_call", "appointment_assistance")
    current_stage_id: str = ""
    
    # Dynamic context variables that can be updated by stages (e.g., user preferences, collected data)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Complete conversation history with timestamps and stage information
    conversation_history: list = field(default_factory=list)
    
    # Reference to the full agent configuration loaded from JSON
    config: Dict[str, Any] = field(default_factory=dict)
    
    # All available stages and their definitions from the configuration
    stages: Dict[str, Any] = field(default_factory=dict)
    
    # LiveKit job context for room and participant information
    ctx: Optional[JobContext] = None

    def summarize(self) -> str:
        """Generate a summary of current session state for logging"""
        return f"Dynamic agent in stage: {self.current_stage_id}, context: {self.context}"

# Type alias for RunContext with UserData for cleaner function signatures
RunContext_T = RunContext[UserData]

class MultiPromptAgent(Agent):
    """
    Dynamic Stage-Based Conversational Agent
    
    This agent can handle any conversation flow defined through JSON configuration.
    It dynamically adapts its behavior based on the current stage and uses function tools
    to transition between stages intelligently.
    
    Key Features:
    - JSON-driven conversation flow definition
    - Dynamic stage transitions with priority-based routing
    - Context-aware variable substitution in prompts
    - Comprehensive conversation history tracking
    - Extensible function tool system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent with JSON configuration
        
        Args:
            config: Complete configuration dict containing:
                - global_settings: LLM, STT, TTS provider configurations
                - agent_config: Agent name, description, base instructions
                - flow: Stage definitions and conversation flow logic
        """
        # Store configuration sections for easy access
        self.config = config
        self.global_settings = config["global_settings"]
        self.agent_config = config["agent_config"]
        self.flow = config["flow"]
        
        # Extract stage information from flow configuration
        self.start_stage_id = self.flow["start_stage"]  # Which stage to start conversation with
        self.stages = self.flow["stages"]  # All available stages and their definitions
        
        logger.info(f"📋 Loaded {len(self.stages)} stages: {list(self.stages.keys())}")
        
        # Build dynamic base instructions that inform the LLM about available stages
        stage_list = ", ".join(self.stages.keys())
        # Create comprehensive base instructions that combine agent config with stage information
        base_instructions = f"""
{self.agent_config["base_instructions"]}

DYNAMIC STAGE-BASED BEHAVIOR:
- You operate in stages: {stage_list}
- Each stage has specific objectives and completion criteria
- Stay in current stage until completion criteria is met
- Use the move_to_next_stage function ONLY when current stage is complete
- Follow stage instructions carefully and ask follow-up questions as needed
"""
        
        # Initialize the LiveKit Agent with all necessary providers
        # This sets up the core voice agent functionality
        super().__init__(
            instructions=base_instructions,  # System prompt with stage information
            stt=self._create_stt_provider(),  # Speech-to-text provider
            llm=self._create_llm_provider(),  # Large language model provider
            tts=self._create_tts_provider(),  # Text-to-speech provider
            vad=self._create_vad_provider()   # Voice activity detection
        )
        
        logger.info(f"🤖 Initialized {self.agent_config['name']}")
    
    def _create_stt_provider(self):
        """
        Create Speech-to-Text provider based on configuration
        
        Supports multiple STT providers (Deepgram, OpenAI) with fallback to Deepgram
        
        Returns:
            STT provider instance configured according to global settings
        """
        stt_settings = self.global_settings["stt_settings"]
        provider = stt_settings["provider"]
        
        # Configure provider based on settings
        if provider == "deepgram":
            return deepgram.STT(
                model=stt_settings["model"],
                language=stt_settings["language"]
            )
        elif provider == "openai":
            return openai.STT(
                model=stt_settings["model"],
                language=stt_settings["language"]
            )
        else:
            # Fallback to Deepgram if provider not recognized
            logger.warning(f"Unknown STT provider: {provider}, using Deepgram")
            return deepgram.STT()
    
    def _create_llm_provider(self):
        """
        Create Large Language Model provider based on configuration
        
        Currently supports OpenAI models with configurable temperature
        
        Returns:
            LLM provider instance configured for conversation management
        """
        llm_settings = self.global_settings["llm_settings"]
        
        # Initialize OpenAI LLM with specified model and creativity settings
        return openai.LLM(
            model=llm_settings["model"],        # e.g., "gpt-4o-mini"
            temperature=llm_settings["temperature"]  # Controls response creativity (0.0-2.0)
        )
    
    def _create_tts_provider(self):
        """
        Create Text-to-Speech provider based on configuration
        
        Supports multiple TTS providers (ElevenLabs, Cartesia, OpenAI) with fallback
        
        Returns:
            TTS provider instance configured with specified voice and model
        """
        tts_settings = self.global_settings["tts_settings"]
        provider = tts_settings["provider"]
        
        # Configure TTS provider based on settings
        if provider == "elevenlabs":
            return elevenlabs.TTS(
                voice_id=tts_settings["voice"],  # ElevenLabs voice ID
                model=tts_settings["model"]
            )
        elif provider == "cartesia":
            return cartesia.TTS(
                model=tts_settings["model"],
                voice=tts_settings["voice"]      # Cartesia voice identifier
            )
        elif provider == "openai":
            return openai.TTS(
                model=tts_settings["model"],
                voice=tts_settings["voice"]      # OpenAI voice (e.g., "ash", "nova")
            )
        else:
            # Fallback to Cartesia if provider not recognized
            logger.warning(f"Unknown TTS provider: {provider}, using cartesia")
            return cartesia.TTS()
    
    def _create_vad_provider(self):
        """
        Create Voice Activity Detection provider
        
        Uses Silero VAD to detect when user is speaking vs silence
        This is crucial for real-time voice conversations
        
        Returns:
            VAD provider instance for voice activity detection
        """
        return silero.VAD.load()
    
    async def on_enter(self) -> None:
        """
        Called when agent starts a new conversation session
        
        This initializes the session state and begins the conversation
        by entering the configured starting stage.
        """
        logger.info("🚀 Dynamic Agent entering conversation")
        
        # Get user session data to track conversation state
        userdata: UserData = self.session.userdata
        
        # Set room participant attributes for identification (if in a room)
        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes({"agent": "MultiPromptAgent"})
        
        # Initialize session state with configuration data
        userdata.current_stage_id = self.start_stage_id  # Start with the configured first stage
        userdata.config = self.config                    # Store full configuration for access
        userdata.stages = self.stages                    # Store stage definitions for reference
        
        # Log initial session state for debugging
        logger.info(f"📋 Initial context: {userdata.context}")
        logger.info(f"🎯 Starting stage: {self.start_stage_id}")
        
        # Begin conversation by entering the starting stage
        await self._enter_stage(userdata.current_stage_id)
        
        logger.info(f"✅ Started conversation in stage: {userdata.current_stage_id}")
    
    async def on_user_speech_committed(self, user_msg: str) -> None:
        """
        Called when user speech is processed and committed by STT
        
        This captures user input and adds it to conversation history
        with context about the current stage.
        
        Args:
            user_msg: The transcribed user speech as text
        """
        userdata: UserData = self.session.userdata
        logger.info(f"👤 USER SAID (in stage '{userdata.current_stage_id}'): '{user_msg}'")
        
        # Record user message in conversation history with metadata
        userdata.conversation_history.append({
            "role": "user",                           # Identifies this as user input
            "content": user_msg,                       # The actual spoken content
            "stage_id": userdata.current_stage_id,     # Which stage user was in when speaking
            "timestamp": datetime.now().isoformat()    # When this was said
        })
        
        # Log current context state for debugging conversation flow
        logger.info(f"📋 Context before processing: {userdata.context}")
    
    async def _enter_stage(self, stage_id: str) -> None:
        """
        Enter and activate a specific conversation stage
        
        This is the core method for stage transitions. It:
        1. Updates the current stage ID
        2. Applies any context updates defined for the stage
        3. Processes and speaks the stage greeting
        4. Updates the LLM's chat context with stage information
        
        Args:
            stage_id: The ID of the stage to enter (must exist in self.stages)
        """
        userdata: UserData = self.session.userdata
        userdata.current_stage_id = stage_id
        
        # Validate that the requested stage exists
        if stage_id not in self.stages:
            logger.error(f"❌ Stage '{stage_id}' not found!")
            return
        
        stage = self.stages[stage_id]
        
        # Apply any context updates defined for this stage
        # These can set variables like department, user_type, etc.
        if "context_updates" in stage:
            userdata.context.update(stage["context_updates"])
            logger.info(f"📝 Context updated: {stage['context_updates']}")
        
        # Log the complete context state after updates
        logger.info(f"📋 Current full context: {userdata.context}")
        
        # Process stage messages with variable substitution
        # Variables like {{user_name}} get replaced with actual context values
        greeting = self._substitute_variables(stage["greeting"], userdata.context)
        prompt = self._substitute_variables(stage["prompt"], userdata.context)
        
        # Log stage entry for debugging and monitoring
        logger.info(f"🎭 ENTERING STAGE '{stage_id}' ({stage['name']})")
        logger.info(f"📍 Stage Description: {stage['description']}")
        logger.info(f"🎯 Completion Criteria: {stage['completion_criteria']}")
        
        # Speak the greeting to the user (this is what they hear)
        logger.info(f"🎤 Speaking: {greeting}")
        await self.session.say(greeting)
        
        # Record the assistant's message in conversation history
        # Note: We store the prompt (instructions) rather than greeting (what was spoken)
        userdata.conversation_history.append({
            "role": "assistant",                    # This is the agent speaking
            "content": prompt,                      # Store the instructions, not the greeting
            "stage_id": stage_id,                   # Which stage this message belongs to
            "timestamp": datetime.now().isoformat() # When this stage was entered
        })
        
        # Update the LLM's chat context with comprehensive stage information
        # This tells the LLM what stage it's in and what it should do
        await self._update_stage_context(stage, prompt)
    
    def _substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """
        Replace variables in text with actual values from context
        
        Variables are defined using double braces: {{variable_name}}
        This allows dynamic content in prompts and greetings.
        
        Example:
            text: "Hello {{user_name}}, welcome to {{department}}"
            context: {"user_name": "John", "department": "billing"}
            result: "Hello John, welcome to billing"
        
        Args:
            text: Text containing variables to substitute
            context: Dictionary of variable names and their values
            
        Returns:
            Text with all variables replaced with their context values
        """
        # Replace each context variable found in the text
        for key, value in context.items():
            # Pattern: {{key}} gets replaced with str(value)
            text = text.replace(f"{{{{{key}}}}}", str(value))
        return text
    
    async def _update_stage_context(self, stage: Dict[str, Any], prompt: str) -> None:
        """
        Update the LLM's chat context with comprehensive stage information
        
        This gives the LLM complete awareness of:
        - What stage it's currently in
        - What the completion criteria are
        - What the next possible stages are
        - Current conversation context
        
        Args:
            stage: The current stage configuration dictionary
            prompt: The processed prompt for this stage
        """
        userdata: UserData = self.session.userdata
        
        # Build information about possible next stages for the LLM
        next_stages_info = ""
        if "next_stages" in stage:
            next_stages_info = "\\nPossible next stages:\\n"
            for next_stage in stage["next_stages"]:
                # Include stage ID, condition, and priority for LLM decision-making
                next_stages_info += f"- {next_stage['stage_id']}: {next_stage['condition']} (Priority: {next_stage.get('priority', 50)})\\n"
        
        # Create comprehensive system message that informs the LLM about current state
        stage_system_msg = f"""
{self.agent_config['base_instructions']}

CURRENT STAGE: {stage['id']} ({stage['name']})
STAGE DESCRIPTION: {stage['description']}
COMPLETION CRITERIA: {stage['completion_criteria']}
CURRENT CONTEXT: {userdata.context}

PROMPT: "{prompt}"

{next_stages_info}

IMPORTANT BEHAVIOR:
- Stay in this stage until completion criteria is fully met
- Follow the stage instructions carefully
- Ask follow-up questions as needed within this stage
- Only call move_to_next_stage when completion criteria is satisfied
- Be thorough and don't rush to next stage
- Consider the priority of next stages when making transition decisions
"""
        
        # Update the LLM's chat context with this comprehensive stage information
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(role="system", content=stage_system_msg)  # System instructions
        chat_ctx.add_message(role="assistant", content=prompt)         # What the agent should do
        await self.update_chat_ctx(chat_ctx)
    
    def _find_next_stage(self, user_input: str, current_stage_id: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Find the appropriate next stage using keyword-based matching
        
        This method implements the current keyword-based transition logic.
        It evaluates conditions in priority order and returns the first match.
        
        Note: This is a candidate for enhancement with LLM-based evaluation
        for more sophisticated condition matching.
        
        Args:
            user_input: What the user said (used for keyword matching)
            current_stage_id: ID of the current stage
            context: Current conversation context
            
        Returns:
            Stage ID to transition to, or None if no conditions match
        """
        current_stage = self.stages[current_stage_id]
        
        # Check if current stage has any next stage definitions
        if "next_stages" not in current_stage:
            logger.info(f"🚫 No next stages defined for '{current_stage_id}'")
            return None
        
        next_stages = current_stage["next_stages"]
        
        # Sort next stages by priority (highest first) for evaluation order
        # Higher priority stages are checked first for matches
        sorted_next_stages = sorted(next_stages, key=lambda s: s.get("priority", 50), reverse=True)
        
        user_lower = user_input.lower()
        
        # Evaluate each possible next stage in priority order
        for next_stage in sorted_next_stages:
            # Simple keyword-based condition matching
            # TODO: This could be enhanced with LLM-based evaluation for better understanding
            condition = next_stage["condition"].lower()
            
            # Split condition into keywords and check if any appear in user input
            condition_keywords = condition.split()
            if any(keyword in user_lower for keyword in condition_keywords):
                target_stage = next_stage["stage_id"]
                logger.info(f"🎯 Next stage found: '{current_stage_id}' → '{target_stage}'")
                logger.info(f"📝 Condition matched: {next_stage['condition']}")
                return target_stage
        
        # No conditions matched - stay in current stage
        logger.info(f"🔄 No next stage condition matched, staying in '{current_stage_id}'")
        return None
    
    # === FUNCTION TOOLS ===
    # These are the tools available to the LLM for stage management
    
    @function_tool
    async def move_to_next_stage(self, target_stage: str, context: RunContext_T) -> None:
        """
        Move directly to a specified target stage
        
        This is the primary function tool for stage transitions. The LLM calls this
        when it determines that the current stage is complete and knows which
        specific stage to move to next.
        
        The function validates:
        1. Target stage exists
        2. Transition is allowed from current stage
        3. Performs the actual stage transition
        
        Args:
            target_stage: The ID of the stage to move to (or "END" to terminate)
        """
        logger.info(f"🎯 FUNCTION CALLED: move_to_next_stage(target_stage='{target_stage}')")
        userdata = context.userdata
        
        # Handle conversation termination
        if target_stage == "END":
            logger.info("🏁 Ending conversation")
            await self.session.say("Thank you for contacting us. Have a great day!")
            return
        
        # Validate that the target stage exists in configuration
        if target_stage not in userdata.stages:
            logger.error(f"❌ Target stage '{target_stage}' not found!")
            return
        
        # Validate that the transition is allowed from current stage
        current_stage = userdata.stages[userdata.current_stage_id]
        valid_transitions = []
        
        if "next_stages" in current_stage:
            valid_transitions = [ns["stage_id"] for ns in current_stage["next_stages"]]
        
        # Check transition validity (END is always allowed)
        if target_stage not in valid_transitions and target_stage != "END":
            logger.warning(f"⚠️  Invalid transition from '{userdata.current_stage_id}' to '{target_stage}'")
            logger.info(f"Valid transitions: {valid_transitions}")
            return
        
        # Perform the validated stage transition
        logger.info(f"✅ Stage transition: '{userdata.current_stage_id}' → '{target_stage}'")
        await self._enter_stage(target_stage)
    
    @function_tool
    async def complete_current_stage(self, next_stage_reason: str, context: RunContext_T) -> None:
        """
        Complete current stage and automatically determine the next stage
        
        This function tool is used when the LLM knows the current stage is complete
        but isn't sure which specific stage to move to next. It uses the reason
        to automatically determine the best next stage using keyword matching.
        
        This is useful when the completion reason contains keywords that match
        the conditions of available next stages.
        
        Args:
            next_stage_reason: Description of why stage is complete (used for automatic stage selection)
        """
        logger.info(f"🎯 FUNCTION CALLED: complete_current_stage(reason='{next_stage_reason}')")
        userdata = context.userdata
        
        # Use keyword-based matching to find the appropriate next stage
        # This leverages the existing _find_next_stage logic
        next_stage = self._find_next_stage(next_stage_reason, userdata.current_stage_id, userdata.context)
        
        if next_stage:
            # Found a matching next stage - transition to it
            await self.move_to_next_stage(next_stage, context)
        else:
            # No matching stage found - stay in current stage
            logger.info("🔄 No appropriate next stage found, staying in current stage")
    
    @function_tool
    async def end_conversation(self, context: RunContext_T) -> None:
        """
        End the conversation gracefully
        
        This function tool handles conversation termination by:
        1. Looking for a completion stage first (allows for wrap-up)
        2. Falling back to a goodbye stage if available
        3. Providing a default goodbye message if no stages are configured
        
        The order of preference ensures a natural conversation ending.
        """
        logger.info("🎯 FUNCTION CALLED: end_conversation")
        
        userdata = context.userdata
        current_stage = userdata.stages[userdata.current_stage_id]
        
        # Get list of valid transitions from current stage
        valid_transitions = []
        if "next_stages" in current_stage:
            valid_transitions = [ns["stage_id"] for ns in current_stage["next_stages"]]
        
        # Try to end conversation through proper stages (in order of preference)
        if "completion" in valid_transitions:
            # Best option: go through completion stage for proper wrap-up
            await self.move_to_next_stage("completion", context)
        elif "goodbye" in valid_transitions:
            # Second option: direct goodbye stage
            await self.move_to_next_stage("goodbye", context)
        elif "goodbye" in userdata.stages:
            # Fallback: force goodbye stage even if not in valid transitions
            logger.info("🔄 Forcing transition to goodbye stage")
            await self._enter_stage("goodbye")
        else:
            # Last resort: direct goodbye message with no stage transition
            await self.session.say("Thank you for contacting us. Have a great day!")
            logger.info("🏁 Conversation ended")

