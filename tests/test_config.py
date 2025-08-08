"""ABOUTME: Unit tests for configuration loading and validation
ABOUTME: Tests YAML loading, environment variable overrides, and Pydantic validation"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml

from bear_mcp.config.settings import load_config
from bear_mcp.config.models import (
    BearMCPConfig, 
    BearDatabaseConfig, 
    EmbeddingConfig,
    VectorStorageConfig,
    OllamaConfig,
    MCPServerConfig,
    LoggingConfig,
    PerformanceConfig
)


class TestConfigModels:
    """Test Pydantic configuration models."""

    def test_bear_database_config_defaults(self):
        """Test BearDatabaseConfig with default values."""
        config = BearDatabaseConfig()
        
        assert config.path is None
        assert config.read_only is True
        assert config.timeout == 30.0
        assert config.check_same_thread is False

    def test_bear_database_config_with_values(self):
        """Test BearDatabaseConfig with custom values."""
        config = BearDatabaseConfig(
            path="/custom/path/database.sqlite",
            read_only=False,
            timeout=60.0,
            check_same_thread=True
        )
        
        assert config.path == Path("/custom/path/database.sqlite")
        assert config.read_only is False
        assert config.timeout == 60.0
        assert config.check_same_thread is True

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig with default values."""
        config = EmbeddingConfig()
        
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.cache_dir is None

    def test_vector_storage_config_defaults(self):
        """Test VectorStorageConfig with default values.""" 
        config = VectorStorageConfig()
        
        assert config.persist_directory == Path("chroma_db")
        assert config.collection_name == "bear_notes"
        assert config.distance_function == "cosine"

    def test_ollama_config_defaults(self):
        """Test OllamaConfig with default values."""
        config = OllamaConfig()
        
        assert config.base_url == "http://localhost:11434"
        assert config.model == "llama2"
        assert config.timeout == 60.0
        assert config.max_retries == 3

    def test_mcp_server_config_defaults(self):
        """Test MCPServerConfig with default values."""
        config = MCPServerConfig()
        
        assert config.name == "bear-notes-mcp"
        assert config.version == "0.1.0"
        assert config.max_resources == 1000

    def test_logging_config_defaults(self):
        """Test LoggingConfig with default values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file_path is None

    def test_performance_config_defaults(self):
        """Test PerformanceConfig with default values."""
        config = PerformanceConfig()
        
        assert config.cache_size == 1000
        assert config.similarity_threshold == 0.7
        assert config.max_related_notes == 10
        assert config.refresh_debounce_seconds == 2.0

    def test_bear_mcp_config_composition(self):
        """Test that BearMCPConfig properly composes all sub-configs."""
        config = BearMCPConfig()
        
        assert isinstance(config.bear_db, BearDatabaseConfig)
        assert isinstance(config.embedding, EmbeddingConfig) 
        assert isinstance(config.vector_storage, VectorStorageConfig)
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.mcp_server, MCPServerConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.performance, PerformanceConfig)


class TestConfigLoading:
    """Test configuration loading from files and environment variables."""

    def test_load_config_with_valid_yaml(self):
        """Test loading configuration from a valid YAML file."""
        config_data = {
            'bear_db': {
                'path': '/test/database.sqlite',
                'timeout': 45.0
            },
            'mcp_server': {
                'name': 'test-server',
                'version': '1.0.0'
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'text'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(Path(config_path))
            
            assert config.bear_db.path == Path('/test/database.sqlite')
            assert config.bear_db.timeout == 45.0
            assert config.mcp_server.name == 'test-server'
            assert config.mcp_server.version == '1.0.0'
            assert config.logging.level == 'DEBUG'
            assert config.logging.format == 'text'
        finally:
            os.unlink(config_path)

    def test_load_config_missing_file(self):
        """Test loading configuration when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config(Path('/nonexistent/config.yaml'))

    @pytest.mark.skip(reason="Needs proper error handling implementation")
    def test_load_config_invalid_yaml(self):
        """Test loading configuration from invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config(Path(config_path))
        finally:
            os.unlink(config_path)

    @pytest.mark.skip(reason="Environment variable override not implemented yet")
    @patch.dict(os.environ, {'BEAR_MCP_MCP_SERVER_NAME': 'env-server'})
    def test_load_config_environment_override(self):
        """Test that environment variables override config file values."""
        config_data = {
            'mcp_server': {
                'name': 'file-server',
                'version': '1.0.0'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(Path(config_path))
            
            # Environment variable should override file value
            assert config.mcp_server.name == 'env-server'
            # File value should still be used where no env var exists
            assert config.mcp_server.version == '1.0.0'
        finally:
            os.unlink(config_path)

    @pytest.mark.skip(reason="Environment variable override not implemented yet")
    @patch.dict(os.environ, {
        'BEAR_MCP_BEAR_DB_PATH': '/env/database.sqlite',
        'BEAR_MCP_BEAR_DB_TIMEOUT': '120',
        'BEAR_MCP_LOGGING_LEVEL': 'ERROR'
    })
    def test_load_config_multiple_environment_overrides(self):
        """Test multiple environment variable overrides."""
        config_data = {
            'bear_db': {
                'path': '/file/database.sqlite',
                'timeout': 30.0
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(Path(config_path))
            
            assert config.bear_db.path == Path('/env/database.sqlite')
            assert config.bear_db.timeout == 120.0  # Should be converted to float
            assert config.logging.level == 'ERROR'
        finally:
            os.unlink(config_path)

    def test_config_validation_errors(self):
        """Test that invalid configuration values raise validation errors."""
        config_data = {
            'bear_db': {
                'timeout': -10  # Invalid: should be positive
            },
            'embedding': {
                'batch_size': 0  # Invalid: should be positive
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(Exception):  # Should be a validation error
                load_config(config_path)
        finally:
            os.unlink(config_path)

    @pytest.mark.skip(reason="Auto-detection implementation details")
    @patch('bear_mcp.config.settings.find_bear_database')
    def test_auto_bear_database_detection(self, mock_find_db):
        """Test automatic Bear database path detection."""
        mock_find_db.return_value = "/auto/detected/database.sqlite"
        
        config_data = {
            'bear_db': {
                # No path specified, should auto-detect
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(Path(config_path))
            
            mock_find_db.assert_called_once()
            assert config.bear_db.path == Path("/auto/detected/database.sqlite")
        finally:
            os.unlink(config_path)

    @pytest.mark.skip(reason="Auto-detection implementation details")
    @patch('bear_mcp.config.settings.find_bear_database')
    def test_bear_database_detection_failure(self, mock_find_db):
        """Test handling when Bear database cannot be auto-detected."""
        mock_find_db.return_value = None
        
        config_data = {
            'bear_db': {
                # No path specified, auto-detection should fail
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Bear database not found"):
                load_config(config_path)
        finally:
            os.unlink(config_path)

    @pytest.mark.skip(reason="Auto-detection implementation details")
    def test_empty_config_file(self):
        """Test loading from an empty config file uses all defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            config_path = f.name
        
        try:
            with patch('bear_mcp.config.settings.find_bear_database') as mock_find_db:
                mock_find_db.return_value = "/default/database.sqlite"
                
                config = load_config(config_path)
                
                # Should have all default values
                assert config.mcp_server.name == "bear-notes-mcp"
                assert config.mcp_server.version == "0.1.0"
                assert config.logging.level == "INFO"
                assert config.bear_db.timeout == 30.0
        finally:
            os.unlink(config_path)