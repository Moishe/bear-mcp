"""ABOUTME: Prompt template system for AI summarization and analysis
ABOUTME: Manages template loading, rendering, and context validation"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import structlog

from bear_mcp.bear_db.models import BearNote

logger = structlog.get_logger(__name__)


class TemplateNotFoundError(Exception):
    """Exception raised when a template is not found."""
    pass


class TemplateRenderError(Exception):
    """Exception raised when template rendering fails."""
    pass


class PromptTemplate:
    """A prompt template with variable substitution."""
    
    def __init__(self, name: str, template: str, description: str = ""):
        """Initialize prompt template.
        
        Args:
            name: Template name/identifier
            template: Template string with {variable} placeholders
            description: Human-readable description
        """
        self.name = name
        self.template = template
        self.description = description
        self._variable_pattern = re.compile(r'\{([^}]+)\}')
    
    def get_variables(self) -> List[str]:
        """Extract all variable names from template.
        
        Returns:
            List of variable names found in template
        """
        return self._variable_pattern.findall(self.template)
    
    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate that context contains all required variables.
        
        Args:
            context: Dictionary of template variables
            
        Returns:
            True if all required variables are present
        """
        required_vars = set(self.get_variables())
        provided_vars = set(context.keys())
        
        return required_vars.issubset(provided_vars)
    
    def render(self, **context) -> str:
        """Render template with provided context.
        
        Args:
            **context: Template variables as keyword arguments
            
        Returns:
            Rendered template string
            
        Raises:
            TemplateRenderError: If rendering fails
        """
        try:
            # Validate context first
            if not self.validate_context(context):
                required = set(self.get_variables())
                provided = set(context.keys())
                missing = required - provided
                raise TemplateRenderError(
                    f"Missing template variables for '{self.name}': {missing}"
                )
            
            # Render template
            return self.template.format(**context)
            
        except KeyError as e:
            raise TemplateRenderError(f"Template variable not found: {e}")
        except Exception as e:
            raise TemplateRenderError(f"Template rendering failed: {e}")


class SummarizationPrompts:
    """Pre-built prompt templates for note summarization."""
    
    def __init__(self):
        """Initialize with default summarization templates."""
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default summarization prompt templates."""
        templates = {}
        
        # Brief summary template
        templates["brief_summary"] = PromptTemplate(
            name="brief_summary",
            template="""Please provide a brief summary of the following note in {max_words} words or less:

Title: {title}

Content:
{content}

Summary:""",
            description="Generate a brief summary of a note"
        )
        
        # Detailed summary template
        templates["detailed_summary"] = PromptTemplate(
            name="detailed_summary",
            template="""Please provide a detailed summary of the following note in {max_sentences} sentences or less:

Title: {title}

Content:
{content}

Please include the main points, key insights, and any important details.

Summary:""",
            description="Generate a detailed summary of a note"
        )
        
        # Structured summary template
        templates["structured_summary"] = PromptTemplate(
            name="structured_summary",
            template="""Please analyze the following note and provide a structured summary with these sections: {sections}

Title: {title}

Content:
{content}

Please format your response with clear headings for each requested section.

Structured Summary:""",
            description="Generate a structured summary with specific sections"
        )
        
        # Multi-note summary template
        templates["multi_note_summary"] = PromptTemplate(
            name="multi_note_summary",
            template="""Please summarize the following notes related to "{theme}" in {max_paragraphs} paragraphs or less:

{notes}

Please identify common themes, connections between the notes, and key insights.

Summary:""",
            description="Generate a summary combining multiple notes"
        )
        
        # Keyword extraction template
        templates["keyword_extraction"] = PromptTemplate(
            name="keyword_extraction",
            template="""Please extract the {max_keywords} most important keywords or phrases from the following content:

{content}

Please list only the keywords, separated by commas.

Keywords:""",
            description="Extract key terms and phrases from content"
        )
        
        return templates
    
    def render_brief_summary(self, title: str, content: str, max_words: int = 50) -> str:
        """Render brief summary template."""
        return self.templates["brief_summary"].render(
            title=title,
            content=content,
            max_words=str(max_words)
        )
    
    def render_detailed_summary(self, title: str, content: str, max_sentences: int = 5) -> str:
        """Render detailed summary template."""
        return self.templates["detailed_summary"].render(
            title=title,
            content=content,
            max_sentences=str(max_sentences)
        )
    
    def render_structured_summary(self, title: str, content: str, sections: List[str]) -> str:
        """Render structured summary template."""
        sections_str = ", ".join(sections)
        return self.templates["structured_summary"].render(
            title=title,
            content=content,
            sections=sections_str
        )
    
    def render_multi_note_summary(self, notes: List[BearNote], theme: str, max_paragraphs: int = 3) -> str:
        """Render multi-note summary template."""
        # Format notes for template
        notes_text = ""
        for i, note in enumerate(notes, 1):
            notes_text += f"\nNote {i}: {note.ztitle or 'Untitled'}\n"
            notes_text += f"{note.ztext or 'No content'}\n"
            notes_text += "-" * 50 + "\n"
        
        return self.templates["multi_note_summary"].render(
            theme=theme,
            notes=notes_text,
            max_paragraphs=str(max_paragraphs)
        )
    
    def render_keyword_extraction(self, content: str, max_keywords: int = 10) -> str:
        """Render keyword extraction template."""
        return self.templates["keyword_extraction"].render(
            content=content,
            max_keywords=str(max_keywords)
        )


class PromptTemplateManager:
    """Manages prompt templates and provides rendering interface."""
    
    def __init__(self):
        """Initialize template manager with default templates."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default summarization templates."""
        summarization_prompts = SummarizationPrompts()
        
        for template_name, template in summarization_prompts.templates.items():
            self.templates[template_name] = template
        
        logger.info("Loaded default templates", count=len(self.templates))
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new template.
        
        Args:
            template: Template to register
        """
        self.templates[template.name] = template
        logger.debug("Registered template", name=template.name)
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template instance
            
        Raises:
            TemplateNotFoundError: If template not found
        """
        if name not in self.templates:
            raise TemplateNotFoundError(f"Template '{name}' not found")
        
        return self.templates[name]
    
    def template_exists(self, name: str) -> bool:
        """Check if template exists.
        
        Args:
            name: Template name
            
        Returns:
            True if template exists
        """
        return name in self.templates
    
    def list_templates(self) -> List[str]:
        """List all available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def render_template(self, template_name: str, **context) -> str:
        """Render template with context.
        
        Args:
            template_name: Template name
            **context: Template variables
            
        Returns:
            Rendered template string
            
        Raises:
            TemplateNotFoundError: If template not found
            TemplateRenderError: If rendering fails
        """
        template = self.get_template(template_name)
        return template.render(**context)
    
    def load_templates_from_directory(self, directory: Path) -> None:
        """Load templates from filesystem directory.
        
        Args:
            directory: Directory containing template files
        """
        if not directory.exists() or not directory.is_dir():
            logger.warning("Template directory not found", path=str(directory))
            return
        
        loaded_count = 0
        
        # Look for template files (*.txt)
        for template_file in directory.glob("*.txt"):
            try:
                template_name = template_file.stem
                template_content = template_file.read_text(encoding="utf-8")
                
                # Look for optional metadata file
                metadata_file = directory / f"{template_name}.json"
                description = ""
                
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                        description = metadata.get("description", "")
                    except Exception as e:
                        logger.warning("Failed to load template metadata", 
                                     file=str(metadata_file), error=str(e))
                
                # Register template
                template = PromptTemplate(
                    name=template_name,
                    template=template_content,
                    description=description
                )
                self.register_template(template)
                loaded_count += 1
                
            except Exception as e:
                logger.error("Failed to load template", 
                           file=str(template_file), error=str(e))
        
        logger.info("Loaded templates from directory", 
                   directory=str(directory), count=loaded_count)
    
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a template.
        
        Args:
            name: Template name
            
        Returns:
            Dictionary with template information
            
        Raises:
            TemplateNotFoundError: If template not found
        """
        template = self.get_template(name)
        
        return {
            "name": template.name,
            "description": template.description,
            "variables": template.get_variables(),
            "template_preview": template.template[:200] + "..." if len(template.template) > 200 else template.template
        }