"""ABOUTME: Content processing for Bear notes including text extraction and keyword analysis
ABOUTME: Handles markdown cleaning, TF-IDF keyword extraction, and note preprocessing"""

from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from bear_mcp.bear_db.models import BearNote
from bear_mcp.semantic.embedding import TextPreprocessor


class ContentProcessor:
    """Processes Bear note content for embedding generation."""
    
    def __init__(self):
        """Initialize content processor."""
        self.text_preprocessor = TextPreprocessor()
    
    def extract_clean_text(self, note: BearNote) -> str:
        """Extract clean text from a Bear note.
        
        Args:
            note: Bear note object
            
        Returns:
            Clean text without markdown formatting
        """
        # Combine title and text
        content_parts = []
        
        if note.ztitle:
            content_parts.append(note.ztitle)
        
        if note.ztext:
            # Clean markdown from note text
            clean_text = self.text_preprocessor.clean_markdown(note.ztext)
            content_parts.append(clean_text)
        
        combined_text = " ".join(content_parts)
        return self.text_preprocessor.normalize_text(combined_text)
    
    def extract_keywords_tfidf(self, texts: List[str], top_k: int = 10) -> List[str]:
        """Extract keywords using TF-IDF.
        
        Args:
            texts: List of documents
            top_k: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        if not texts:
            return []
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=top_k * 2,  # Get more features to select from
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.8  # Maximum document frequency
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores across all documents
            scores = tfidf_matrix.sum(axis=0).A1
            
            # Get top keywords
            top_indices = scores.argsort()[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            return keywords
        except ValueError:
            # Handle case where no valid features are found
            return []
    
    def process_note_for_embedding(self, note: BearNote) -> str:
        """Process a note for embedding generation.
        
        Args:
            note: Bear note object
            
        Returns:
            Processed text ready for embedding
        """
        return self.extract_clean_text(note)