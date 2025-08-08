"""Configuration loading and management for Bear MCP Server."""

import os
from pathlib import Path
from typing import Optional

import structlog
import yaml
from pydantic import ValidationError

from bear_mcp.config.models import BearMCPConfig

logger = structlog.get_logger()


def get_default_bear_db_path() -> Optional[Path]:
    """Auto-detect the Bear database path on macOS."""
    # Standard Bear 2 database locations
    possible_paths = [
        Path.home() / "Library" / "Group Containers" / "9K33E3U3T4.net.shinyfrog.bear" / "Application Data" / "database.sqlite",
        Path.home() / "Library" / "Containers" / "net.shinyfrog.bear" / "Data" / "Documents" / "Application Data" / "database.sqlite",
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info("Found Bear database", path=str(path))
            return path
    
    logger.warning("Bear database not found in standard locations", paths=[str(p) for p in possible_paths])
    return None


def load_config(config_path: Optional[Path] = None) -> BearMCPConfig:
    """Load configuration from YAML file with environment variable overrides.
    
    Args:
        config_path: Optional path to configuration file. If None, will look for:
                    - ./config/config.yaml
                    - ./config.yaml
                    - ~/.bear-mcp/config.yaml
    
    Returns:
        BearMCPConfig instance
        
    Raises:
        FileNotFoundError: If no configuration file is found
        ValidationError: If configuration is invalid
    """
    if config_path is None:
        # Try standard locations
        candidates = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path.home() / ".bear-mcp" / "config.yaml",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
        
        if config_path is None:
            logger.info("No configuration file found, using defaults")
            config_data = {}
        else:
            logger.info("Loading configuration", path=str(config_path))
            config_data = _load_yaml_file(config_path)
    else:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        config_data = _load_yaml_file(config_path)
    
    # Apply environment variable overrides
    config_data = _apply_env_overrides(config_data)
    
    # Auto-detect Bear database path if not provided
    if not config_data.get("bear_db", {}).get("path"):
        bear_db_path = get_default_bear_db_path()
        if bear_db_path:
            if "bear_db" not in config_data:
                config_data["bear_db"] = {}
            config_data["bear_db"]["path"] = str(bear_db_path)
    
    try:
        config = BearMCPConfig(**config_data)
        logger.info("Configuration loaded successfully")
        return config
    except ValidationError as e:
        logger.error("Configuration validation failed", errors=e.errors())
        raise


def _load_yaml_file(path: Path) -> dict:
    """Load YAML file safely."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in configuration file", path=str(path), error=str(e))
        raise ValueError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        logger.error("Error reading configuration file", path=str(path), error=str(e))
        raise


def _apply_env_overrides(config_data: dict) -> dict:
    """Apply environment variable overrides to configuration.
    
    Environment variables should be prefixed with BEAR_MCP_ and use underscores
    to separate nested keys. For example:
    - BEAR_MCP_BEAR_DB_PATH=/path/to/db.sqlite
    - BEAR_MCP_OLLAMA_MODEL=llama2:13b
    - BEAR_MCP_LOGGING_LEVEL=DEBUG
    """
    env_prefix = "BEAR_MCP_"
    
    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue
            
        # Remove prefix and convert to lowercase
        config_key = key[len(env_prefix):].lower()
        
        # Split by underscores to get nested path
        key_parts = config_key.split("_")
        
        # Navigate/create nested dictionary structure
        current = config_data
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value, attempting type conversion
        final_key = key_parts[-1]
        current[final_key] = _convert_env_value(value)
        
        logger.debug("Applied environment override", key=key, value=value)
    
    return config_data


def _convert_env_value(value: str) -> any:
    """Convert environment variable string to appropriate type."""
    # Boolean conversion
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    # Integer conversion
    if value.isdigit():
        return int(value)
    
    # Float conversion
    try:
        if "." in value:
            return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def create_default_config_file(path: Path) -> None:
    """Create a default configuration file.
    
    Args:
        path: Path where to create the configuration file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = BearMCPConfig()
    
    # Convert to dictionary for YAML serialization
    config_dict = default_config.model_dump()
    
    # Convert Path objects to strings for YAML serialization
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    config_dict = convert_paths(config_dict)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info("Created default configuration file", path=str(path))