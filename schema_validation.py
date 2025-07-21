# schema_validation.py - Updated for Dynamic Stage Schema
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaValidator:
    """
    A dynamic multi-stage agent that uses JSON configuration.
    
    This class will:
    1. Load and validate JSON configuration
    2. Validate the JSON Config
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the agent with a configuration file.
        
        Args:
            config_path (str): Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config = None
        
        # Load and validate configuration
        self._load_config()
        self._validate_config()
        
        logger.info(f"MultiPromptAgent initialized with config: {config_path}")
    
    def _load_config(self) -> None:
        """
        Load the JSON configuration from file.
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = json.load(file)
            logger.info("Configuration loaded successfully")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration against the detected schema.
        """
        logger.info("Starting configuration validation...")
        
        # Check required top-level keys
        required_keys = ["global_settings", "agent_config", "flow"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Validate global_settings
        self._validate_global_settings()
        
        # Validate agent_config
        self._validate_agent_config()
        
        #Valdate Stage Flow
        self._validate_stage_flow()
        
        logger.info("Configuration validation completed successfully")
    
    def _validate_global_settings(self) -> None:
        """Validate global_settings section"""
        global_settings = self.config["global_settings"]
        
        # Check required subsections
        required_sections = ["llm_settings", "stt_settings", "tts_settings"]
        for section in required_sections:
            if section not in global_settings:
                raise ValueError(f"Missing {section} in global_settings")
        
        # Validate LLM settings
        llm = global_settings["llm_settings"]
        if not all(key in llm for key in ["provider", "model", "temperature"]):
            raise ValueError("LLM settings missing required fields")
        
        # Validate STT settings
        stt = global_settings["stt_settings"]
        if not all(key in stt for key in ["provider", "model", "language"]):
            raise ValueError("STT settings missing required fields")
        
        # Validate TTS settings
        tts = global_settings["tts_settings"]
        if not all(key in tts for key in ["provider", "model", "voice"]):
            raise ValueError("TTS settings missing required fields")
        
        logger.info("Global settings validation passed")
    
    def _validate_agent_config(self) -> None:
        """Validate agent_config section"""
        agent_config = self.config["agent_config"]
        
        required_fields = ["name", "base_instructions"]
        for field in required_fields:
            if field not in agent_config:
                raise ValueError(f"Missing {field} in agent_config")
        
        logger.info("Agent config validation passed")
    
    def _validate_stage_flow(self) -> None:
        """Validate stage-based flow section"""
        flow = self.config["flow"]
        
        # Check required fields for stage-based flow
        if "start_stage" not in flow:
            raise ValueError("Missing start_stage in flow")
        if "stages" not in flow:
            raise ValueError("Missing stages in flow")
        
        # Check that start_stage exists in stages
        start_stage = flow["start_stage"]
        stages = flow["stages"]
        
        if start_stage not in stages:
            raise ValueError(f"start_stage '{start_stage}' not found in stages")
        
        # Validate each stage
        for stage_id, stage in stages.items():
            self._validate_stage(stage_id, stage, stages)
        
        logger.info("Stage flow validation passed")
    

    
    def _validate_stage(self, stage_id: str, stage: Dict[str, Any], all_stages: Dict[str, Any]) -> None:
        """Validate individual stage"""
        # Check required fields for stages
        required_fields = ["id", "name", "prompt", "completion_criteria"]
        for field in required_fields:
            if field not in stage:
                raise ValueError(f"Stage '{stage_id}' missing required field: {field}")
        
        # Validate next_stages if they exist
        if "next_stages" in stage:
            for next_stage in stage["next_stages"]:
                if "stage_id" not in next_stage:
                    raise ValueError(f"next_stage in stage '{stage_id}' missing stage_id")
                if "condition" not in next_stage:
                    raise ValueError(f"next_stage in stage '{stage_id}' missing condition")
                
                target_stage = next_stage["stage_id"]
                # Check that target stage exists (unless it's END)
                if target_stage != "END" and target_stage not in all_stages:
                    raise ValueError(f"Stage '{stage_id}' references non-existent stage: {target_stage}")
        
        logger.debug(f"Stage '{stage_id}' validation passed")
    
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded configuration.
        """
        if not self.config:
            return {}
        
        flow = self.config["flow"]
        
        return {
                "agent_name": self.config["agent_config"]["name"],
                "schema_type": "stages",
                "start_stage": flow["start_stage"],
                "total_stages": len(flow["stages"]),
                "stage_list": list(flow["stages"].keys()),
                "llm_model": self.config["global_settings"]["llm_settings"]["model"],
                "stt_provider": self.config["global_settings"]["stt_settings"]["provider"],
                "tts_provider": self.config["global_settings"]["tts_settings"]["provider"]
                 }

# Test the validation setup
if __name__ == "__main__":
    import sys
    
    # Check if config file path is provided as argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Default to the standard config file
        config_file = "agents_configs/default.json"
        print(f"📋 No config file specified, using default: {config_file}")
        print("💡 Usage: python schema_validation.py <path/to/config.json>")
    
    try:
        print(f"🔍 Validating configuration file: {config_file}")
        
        # Test with the specified config file
        validator = SchemaValidator(config_file)
        
        # Print configuration summary
        summary = validator.get_config_summary()
        print("\n=== Configuration Summary ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        import traceback
        print("\n--- Full Error Details ---")
        traceback.print_exc()
        sys.exit(1)