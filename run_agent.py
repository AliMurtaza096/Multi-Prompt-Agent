#!/usr/bin/env python3
"""
Multi-Prompt Agent Runner

This is the main entry point for running the Multi-Prompt Agent with LiveKit.
It handles:
- Environment setup and configuration loading
- Schema validation of agent configurations
- LiveKit session initialization and management
- Error handling and logging

Usage:
    CONFIG_PATH=agents_configs/default.json python run_agent.py dev
"""

import logging
from dotenv import load_dotenv
import os

# LiveKit imports for voice agent infrastructure
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe
from livekit.agents.voice import AgentSession

# Import our validation logic for JSON schema compliance
from schema_validation import SchemaValidator

# Import our dynamic multi-prompt agent and data models
from multi_prompt_agent import MultiPromptAgent, UserData

# Import logging utilities
from utils.logger import setup_logging

# Initialize logger for this module
logger = logging.getLogger("multi-prompt-agent")

# Load environment variables from .env file
load_dotenv()

# Configuration file path - can be set via environment variable
# Default fallback is for backward compatibility
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/default.json")

async def entrypoint(ctx: JobContext):
    """
    Main entrypoint function called by LiveKit when a new session starts
    
    This function:
    1. Connects to the LiveKit room
    2. Waits for a participant to join
    3. Loads and validates the agent configuration
    4. Initializes the multi-prompt agent
    5. Starts the conversation session
    
    Args:
        ctx: LiveKit JobContext containing room and session information
    """
    # Establish connection to LiveKit room (audio only for voice agent)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Wait for a participant to join the room before starting
    participant = await ctx.wait_for_participant()
    
    # Initialize logging system and get log file path
    log_filename = setup_logging()
    logger.info(f"📝 Log file created: {log_filename}")
    logger.info("🚀 Starting Dynamic Multi-Stage Agent")
    logger.info(f"👤 Participant connected: {participant.identity}")
    
    try:
        # Load configuration file and validate against JSON schema
        logger.info(f"📄 Loading configuration from: {CONFIG_PATH}")
        validator = SchemaValidator(CONFIG_PATH)
        config = validator.config
        
        logger.info(f"✅ Config loaded: {config['agent_config']['name']}")
        
        # Perform additional validation of flow structure
        # This ensures the configuration has the required conversation flow elements
        if "flow" not in config:
            raise ValueError("Missing 'flow' section in configuration")
        
        flow = config["flow"]
        if "start_stage" not in flow or "stages" not in flow:
            raise ValueError("Flow must contain both 'start_stage' and 'stages' definitions")
        
        start_stage = flow["start_stage"]
        stages = flow["stages"]
        
        # Validate that the specified start stage actually exists
        if start_stage not in stages:
            raise ValueError(f"Start stage '{start_stage}' not found in defined stages")
        
        logger.info(f"✅ Flow validation passed - Start stage: {start_stage}")
        logger.info(f"📋 Available stages: {list(stages.keys())}")
        
        # Create user data object to track conversation state
        # This will be passed to the agent and persist throughout the session
        userdata = UserData(ctx=ctx)
        
        # Initialize the multi-prompt agent with validated configuration
        agent = MultiPromptAgent(config)
        
        # Create LiveKit agent session with our user data model
        session = AgentSession[UserData](userdata=userdata)
        
        logger.info("✅ Starting LiveKit agent session")
        # Start the session - this begins the conversation
        await session.start(agent=agent, room=ctx.room)
        
    except Exception as e:
        # Log detailed error information for debugging
        logger.error(f"❌ Failed to start agent: {str(e)}")
        logger.error(f"❌ Exception type: {type(e)}")
        
        # Print full stack trace for development debugging
        import traceback
        traceback.print_exc()
        
        # Re-raise the exception so LiveKit can handle it appropriately
        raise


if __name__ == "__main__":
    # Start the LiveKit CLI application with our entrypoint function
    # This handles command line arguments and LiveKit server connection
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))