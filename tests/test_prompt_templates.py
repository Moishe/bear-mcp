"""ABOUTME: Unit tests for prompt template system and management
ABOUTME: Tests template loading, rendering, and context management"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from bear_mcp.ai.prompt_templates import (
    PromptTemplate,
    PromptTemplateManager,
    SummarizationPrompts,
    TemplateNotFoundError,
    TemplateRenderError
)
from bear_mcp.bear_db.models import BearNote


@pytest.fixture
def sample_note():
    """Create sample Bear note for testing."""
    return BearNote(
        z_pk=1,
        zuniqueidentifier="NOTE-123",
        ztitle="Machine Learning Introduction",
        ztext="# Machine Learning\n\nMachine learning is a field of AI that focuses on algorithms. #ml #ai\n\nKey concepts:\n- Supervised learning\n- Unsupervised learning\n- Deep learning",
        zcreationdate=725846400.0,
        zmodificationdate=725846400.0
    )


class TestPromptTemplate:
    """Test prompt template functionality."""

    def test_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="test_template",
            template="Hello {name}, you have {count} messages.",
            description="Test template"
        )
        
        assert template.name == "test_template"
        assert template.template == "Hello {name}, you have {count} messages."
        assert template.description == "Test template"

    def test_template_render_success(self):
        """Test successful template rendering."""
        template = PromptTemplate(
            name="greeting",
            template="Hello {name}! Today is {day}.",
            description="Greeting template"
        )
        
        result = template.render(name="Alice", day="Monday")
        assert result == "Hello Alice! Today is Monday."

    def test_template_render_missing_variable(self):
        """Test template rendering with missing variable."""
        template = PromptTemplate(
            name="greeting", 
            template="Hello {name}! You have {count} items.",
            description="Test template"
        )
        
        with pytest.raises(TemplateRenderError, match="Missing template variable"):
            template.render(name="Bob")

    def test_template_render_extra_variables(self):
        """Test template rendering with extra variables (should ignore them)."""
        template = PromptTemplate(
            name="simple",
            template="Hello {name}!",
            description="Simple template"
        )
        
        result = template.render(name="Charlie", extra="ignored")
        assert result == "Hello Charlie!"

    def test_template_get_variables(self):
        """Test extracting template variables."""
        template = PromptTemplate(
            name="complex",
            template="Dear {name}, your {item} is ready. Please arrive at {time}.",
            description="Complex template"
        )
        
        variables = template.get_variables()
        assert set(variables) == {"name", "item", "time"}

    def test_template_validate_context(self):
        """Test context validation."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}, your score is {score}.",
            description="Test template"
        )
        
        # Valid context
        assert template.validate_context({"name": "Alice", "score": "95"}) == True
        
        # Missing variable
        assert template.validate_context({"name": "Bob"}) == False
        
        # Extra variables (should still be valid)
        assert template.validate_context({"name": "Charlie", "score": "88", "extra": "data"}) == True


class TestSummarizationPrompts:
    """Test summarization prompt templates."""

    def test_brief_summary_template(self, sample_note):
        """Test brief summary template."""
        prompts = SummarizationPrompts()
        
        result = prompts.render_brief_summary(
            title=sample_note.ztitle,
            content=sample_note.ztext,
            max_words=50
        )
        
        assert "Machine Learning Introduction" in result
        assert "50 words" in result
        assert sample_note.ztext in result

    def test_detailed_summary_template(self, sample_note):
        """Test detailed summary template."""
        prompts = SummarizationPrompts()
        
        result = prompts.render_detailed_summary(
            title=sample_note.ztitle,
            content=sample_note.ztext,
            max_sentences=3
        )
        
        assert "Machine Learning Introduction" in result
        assert "3 sentences" in result
        assert sample_note.ztext in result

    def test_structured_summary_template(self, sample_note):
        """Test structured summary template."""
        prompts = SummarizationPrompts()
        
        result = prompts.render_structured_summary(
            title=sample_note.ztitle,
            content=sample_note.ztext,
            sections=["Main Topic", "Key Points", "Tags"]
        )
        
        assert "Machine Learning Introduction" in result
        assert "Main Topic" in result
        assert "Key Points" in result
        assert "Tags" in result

    def test_multi_note_summary_template(self, sample_note):
        """Test multi-note summary template."""
        prompts = SummarizationPrompts()
        
        notes = [sample_note]
        result = prompts.render_multi_note_summary(
            notes=notes,
            theme="AI and Machine Learning",
            max_paragraphs=2
        )
        
        assert "AI and Machine Learning" in result
        assert "2 paragraphs" in result
        assert sample_note.ztitle in result

    def test_keyword_extraction_template(self, sample_note):
        """Test keyword extraction template."""
        prompts = SummarizationPrompts()
        
        result = prompts.render_keyword_extraction(
            content=sample_note.ztext,
            max_keywords=5
        )
        
        assert "5 most important keywords" in result
        assert sample_note.ztext in result


class TestPromptTemplateManager:
    """Test prompt template manager functionality."""

    def test_manager_creation(self):
        """Test creating template manager."""
        manager = PromptTemplateManager()
        assert len(manager.templates) > 0  # Should have default templates

    def test_register_template(self):
        """Test registering a new template."""
        manager = PromptTemplateManager()
        
        template = PromptTemplate(
            name="test_template",
            template="Test {value}",
            description="Test template"
        )
        
        manager.register_template(template)
        assert "test_template" in manager.templates
        assert manager.templates["test_template"] == template

    def test_get_template_success(self):
        """Test getting existing template."""
        manager = PromptTemplateManager()
        
        template = PromptTemplate(
            name="existing",
            template="Existing template",
            description="Test"
        )
        manager.register_template(template)
        
        retrieved = manager.get_template("existing")
        assert retrieved == template

    def test_get_template_not_found(self):
        """Test getting non-existent template."""
        manager = PromptTemplateManager()
        
        with pytest.raises(TemplateNotFoundError, match="Template 'nonexistent' not found"):
            manager.get_template("nonexistent")

    def test_list_templates(self):
        """Test listing all templates."""
        manager = PromptTemplateManager()
        
        custom_template = PromptTemplate(
            name="custom",
            template="Custom {data}",
            description="Custom template"
        )
        manager.register_template(custom_template)
        
        templates = manager.list_templates()
        assert len(templates) > 0
        assert "custom" in templates

    def test_render_template(self):
        """Test rendering template through manager."""
        manager = PromptTemplateManager()
        
        template = PromptTemplate(
            name="render_test",
            template="Hello {name}!",
            description="Test template"
        )
        manager.register_template(template)
        
        result = manager.render_template("render_test", name="World")
        assert result == "Hello World!"

    def test_template_exists(self):
        """Test checking if template exists."""
        manager = PromptTemplateManager()
        
        template = PromptTemplate(
            name="exists_test",
            template="Test template",
            description="Test"
        )
        manager.register_template(template)
        
        assert manager.template_exists("exists_test")
        assert not manager.template_exists("does_not_exist")

    def test_load_templates_from_directory(self):
        """Test loading templates from filesystem."""
        manager = PromptTemplateManager()
        
        # Create temporary directory with template files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create template file
            template_file = temp_path / "test_template.txt"
            template_file.write_text("Hello {name}, welcome to {place}!")
            
            # Create metadata file
            metadata_file = temp_path / "test_template.json"
            metadata_file.write_text('{"description": "Test template from file"}')
            
            manager.load_templates_from_directory(temp_path)
            
            assert manager.template_exists("test_template")
            template = manager.get_template("test_template")
            assert template.template == "Hello {name}, welcome to {place}!"
            assert template.description == "Test template from file"

    def test_get_default_templates(self):
        """Test that default templates are loaded."""
        manager = PromptTemplateManager()
        
        # Should have summarization templates
        assert manager.template_exists("brief_summary")
        assert manager.template_exists("detailed_summary")
        assert manager.template_exists("structured_summary")
        assert manager.template_exists("multi_note_summary")
        assert manager.template_exists("keyword_extraction")


# Integration tests
class TestPromptTemplateIntegration:
    """Integration tests for prompt template system."""

    def test_end_to_end_summarization(self, sample_note):
        """Test complete summarization workflow."""
        manager = PromptTemplateManager()
        
        # Get brief summary template
        brief_template = manager.get_template("brief_summary")
        
        # Render with note data
        result = brief_template.render(
            title=sample_note.ztitle,
            content=sample_note.ztext,
            max_words=30
        )
        
        assert "Machine Learning Introduction" in result
        assert "30 words" in result
        assert isinstance(result, str)
        assert len(result) > 0

    def test_template_context_validation_workflow(self):
        """Test template validation in realistic workflow."""
        manager = PromptTemplateManager()
        
        # Register custom template
        custom_template = PromptTemplate(
            name="note_analysis",
            template="Analyze note titled '{title}' with {word_count} words. Focus on {aspect}.",
            description="Note analysis template"
        )
        manager.register_template(custom_template)
        
        # Valid context
        context = {
            "title": "Test Note",
            "word_count": "150",
            "aspect": "key themes"
        }
        
        template = manager.get_template("note_analysis")
        assert template.validate_context(context)
        
        result = manager.render_template("note_analysis", **context)
        assert "Test Note" in result
        assert "150 words" in result
        assert "key themes" in result

    def test_error_handling_workflow(self):
        """Test error handling in template system."""
        manager = PromptTemplateManager()
        
        # Try to use non-existent template
        with pytest.raises(TemplateNotFoundError):
            manager.render_template("non_existent", data="test")
        
        # Try to render with missing context
        template = PromptTemplate(
            name="incomplete",
            template="Hello {name}, your {item} is ready!",
            description="Test template"
        )
        manager.register_template(template)
        
        with pytest.raises(TemplateRenderError):
            manager.render_template("incomplete", name="Alice")  # Missing 'item'