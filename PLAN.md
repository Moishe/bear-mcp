# Bear Notes MCP Server - Implementation Plan

## Overview
Build a Model Context Protocol (MCP) server in Python using FastMCP to provide programmatic access to Bear notes stored locally on macOS. The server will use semantic analysis for finding related content and AI-powered summarization.

## Technical Architecture

### Core Components
1. **Bear Database Interface** - Direct SQLite access to Bear's database
2. **Semantic Analysis Engine** - Extract meaning and find relationships
3. **MCP Server** - FastMCP-based server exposing resources and tools
4. **Caching Layer** - Performance optimization for large note collections
5. **AI Integration** - Local Ollama integration for summarization

### Data Flow
```
Bear SQLite DB → Database Interface → Semantic Analysis → MCP Resources/Tools
                                  ↗                    ↘
                            Caching Layer          AI Summarization (Ollama)
```

## Bear Database Integration

### Database Location
Bear 2 stores notes in: `~/Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite`

### Key Tables (Bear 2 Schema)
- `ZSFNOTE` - Main notes table
  - `ZTITLE` - Note title
  - `ZTEXT` - Full note content (Markdown)
  - `ZCREATIONDATE` - Creation timestamp
  - `ZMODIFICATIONDATE` - Last modified timestamp
  - `ZTRASHED` - Trash status
  - `ZARCHIVED` - Archive status
- `ZSFNOTETAG` - Note-tag relationships
- `ZSFTAG` - Tags/hashtags table

### Database Access Strategy
- Read-only SQLite connections
- Monitor database file modification time for refresh triggers
- Handle database locks gracefully (Bear may be writing)

## MCP Server Structure

### Resources
1. **`note:///{note_id}`** - Individual note content
   - Full markdown content
   - Metadata (title, creation/modification dates, tags)
   
2. **`keywords:///{note_id}`** - Extracted keywords for a specific note
   - AI-extracted keywords and key phrases
   - Frequency information
   
3. **`hashtags:///{note_id}`** - Hashtags for a specific note
   - All hashtags used in the note
   - Hierarchical tag structure if nested

4. **`notes://all`** - List of all notes
   - Note IDs, titles, and basic metadata
   - Excluding trashed notes

### Tools
1. **`find_related_notes`**
   - Input: note_id or search_text
   - Uses semantic similarity to find related notes
   - Returns ranked list of related notes with similarity scores

2. **`find_related_keywords`**
   - Input: keyword or note_id
   - Finds notes containing semantically similar keywords
   - Returns keywords grouped by semantic clusters

3. **`find_related_hashtags`**
   - Input: hashtag or note_id  
   - Finds related hashtags based on co-occurrence patterns
   - Returns hashtag suggestions and usage statistics

4. **`summarize_notes`**
   - Input: list of note_ids or search criteria
   - Uses Ollama to generate AI-powered summaries
   - Configurable summary length and style

5. **`search_notes`**
   - Input: search query (full-text or semantic)
   - Returns matching notes with relevance scores
   - Supports both keyword and semantic search modes

## Semantic Analysis Implementation

### Embedding Strategy
- Use sentence-transformers for local embedding generation
- Model: `all-MiniLM-L6-v2` (good balance of speed/quality)
- Store embeddings in local vector database (ChromaDB or FAISS)

### Content Processing Pipeline
1. **Text Extraction**: Clean markdown, extract plain text
2. **Chunking**: Split long notes into semantic chunks
3. **Embedding Generation**: Create vector representations
4. **Keyword Extraction**: Use TF-IDF + semantic analysis
5. **Index Storage**: Persist embeddings and metadata

### Similarity Computation
- Cosine similarity for semantic relatedness
- Hybrid scoring: semantic similarity + hashtag overlap + keyword matching
- Configurable similarity thresholds

## AI Integration (Ollama)

### Model Selection
- **Primary**: `llama3.2:3b` (good for M4 MacBook Pro with 16GB RAM)
- **Fallback**: `phi3.5:3.8b` if memory issues
- **Summarization**: Fine-tuned for concise, structured output

### Summarization Prompt Templates
```python
SUMMARY_PROMPT = """
Summarize the following notes in a concise, structured format:
- Main themes and topics
- Key insights or conclusions  
- Important facts or data points
- Action items (if any)

Notes:
{note_content}

Summary:
"""
```

### Performance Considerations
- Batch processing for multiple notes
- Streaming responses for long summaries
- Context window management (typically 4K-8K tokens)

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [x] Set up FastMCP server structure
- [ ] Implement Bear SQLite database interface
- [ ] Create basic note reading functionality
- [ ] Add database change monitoring
- [ ] Implement basic MCP resources (notes, hashtags)

### Phase 2: Semantic Analysis (Week 2)
- [ ] Set up sentence-transformers pipeline
- [ ] Implement embedding generation and storage
- [ ] Create keyword extraction system
- [ ] Build similarity computation engine
- [ ] Add semantic search capabilities

### Phase 3: AI Integration (Week 3)  
- [ ] Set up Ollama integration
- [ ] Implement note summarization
- [ ] Create prompt templates and response parsing
- [ ] Add batch processing capabilities
- [ ] Performance optimization

### Phase 4: Advanced Tools (Week 4)
- [ ] Implement `find_related_*` tools
- [ ] Add hybrid search (semantic + keyword)
- [ ] Create caching layer for performance
- [ ] Add error handling and logging
- [ ] Write comprehensive tests

### Phase 5: Polish & Documentation (Week 5)
- [ ] Performance tuning and optimization
- [ ] Add configuration management
- [ ] Create usage documentation
- [ ] Add monitoring and health checks
- [ ] Final testing and debugging

## Technical Dependencies

### Core Libraries
```toml
dependencies = [
    "fastmcp>=0.1.0",
    "sqlite3",  # Built into Python
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "ollama>=0.1.0",
    "watchdog>=3.0.0",  # File monitoring
    "pydantic>=2.0.0",
    "asyncio",
    "aiofiles>=23.0.0"
]
```

### Development Dependencies
```toml
dev-dependencies = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0", 
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0"
]
```

## Configuration Management

### Config File Structure (`config.yaml`)
```yaml
bear:
  database_path: "~/Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite"
  refresh_interval: 300  # 5 minutes
  
semantic:
  model_name: "all-MiniLM-L6-v2"
  embedding_dim: 384
  similarity_threshold: 0.7
  chunk_size: 512
  
ollama:
  base_url: "http://localhost:11434"
  model: "llama3.2:3b"
  temperature: 0.1
  max_tokens: 1000
  
storage:
  cache_dir: "./cache"
  vector_db: "chromadb"
  
logging:
  level: "INFO"
  file: "./logs/bear-mcp.log"
```

## Performance Considerations

### Optimization Strategies
1. **Lazy Loading**: Load embeddings on-demand
2. **Caching**: Cache frequently accessed notes and embeddings
3. **Batch Processing**: Process multiple notes simultaneously
4. **Incremental Updates**: Only re-process changed notes
5. **Memory Management**: Efficient vector storage and retrieval

### Expected Performance
- Initial indexing: ~10-30 seconds for 1000 notes
- Semantic search: <1 second for most queries
- Note summarization: 2-5 seconds per note (depending on length)
- Memory usage: ~200-500MB for 1000 notes with embeddings

## Error Handling & Edge Cases

### Database Issues
- Handle SQLite lock situations gracefully
- Retry logic for transient database errors
- Fallback to cached data when database is unavailable

### AI/ML Issues  
- Handle Ollama service unavailability
- Graceful degradation when embeddings fail to generate
- Timeout handling for long-running AI operations

### Data Quality
- Handle malformed or empty notes
- Unicode and special character support
- Large note handling (>10K characters)

## Security Considerations

### Data Access
- Read-only database access
- No modification of Bear's database
- Local-only operation (no external data transmission)

### Resource Limits
- Rate limiting for expensive operations
- Memory usage caps
- Timeout controls for long-running operations

## Testing Strategy

### Unit Tests
- Database interface functionality
- Semantic analysis accuracy
- MCP resource/tool responses
- Configuration management

### Integration Tests  
- End-to-end note retrieval and processing
- Ollama integration testing
- Performance benchmarks
- Error scenario handling

### Manual Testing
- Real Bear database testing
- Large note collection performance
- Cross-platform compatibility (different macOS versions)

## Deployment & Distribution

### Local Installation
```bash
# Install dependencies
pip install -e .

# Set up Ollama
ollama pull llama3.2:3b

# Initialize embeddings
python -m bear_mcp.setup --initialize

# Run MCP server
python -m bear_mcp.server
```

### MCP Client Configuration
Example for Claude Desktop:
```json
{
  "bear-notes": {
    "command": "python",
    "args": ["-m", "bear_mcp.server"],
    "cwd": "/path/to/bear-mcp"
  }
}
```

## Future Enhancements

### Potential Additions
1. **Advanced Search**: Boolean operators, date ranges, tag filtering
2. **Note Relationships**: Automatic link detection between notes  
3. **Export Tools**: Export filtered note sets to various formats
4. **Collaborative Features**: Share note summaries or insights
5. **Bear Integration**: Two-way sync for new note creation
6. **Multi-language Support**: Handle notes in different languages
7. **Plugin System**: Extensible tool architecture

### Performance Improvements
1. **Vector Database Upgrades**: Move to more sophisticated vector stores
2. **Better Models**: Experiment with domain-specific embedding models
3. **Caching Strategies**: More intelligent cache invalidation
4. **Parallel Processing**: Multi-threaded embedding generation

## Success Metrics

### Functional Goals
- [ ] Successfully read all notes from Bear database
- [ ] Generate meaningful semantic relationships between notes
- [ ] Provide accurate search results (>80% relevance)
- [ ] Generate coherent note summaries
- [ ] Handle 1000+ notes without performance issues

### Performance Goals  
- [ ] <1 second response time for search queries
- [ ] <5 seconds for note summarization
- [ ] <500MB memory usage for typical workload
- [ ] >95% uptime during normal operation

This plan provides a comprehensive roadmap for building a robust, performant MCP server for Bear notes with semantic search and AI-powered features.