# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **LangChain + Ollama-based intelligent customer service chatbot** specifically designed for the "泰迪杯" (Taidi Cup) data mining competition. The system implements advanced RAG (Retrieval-Augmented Generation) architecture with both traditional and optimized modes.

## Key Commands

### Development and Testing
```bash
# Install dependencies
python run_langchain.py --install

# Check environment
python run_langchain.py --check

# Run tests
python test_langchain.py

# Test optimized RAG system
python run_langchain.py --mode optimized
```

### Running the System
```bash
# Web interface (recommended)
python run_langchain.py --mode web

# Command line interface
python run_langchain.py --mode cli

# Command line with optimized mode
python run_langchain.py --mode cli --use-optimized
```

### Ollama Model Management
```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text:latest

# List available models
ollama list
```

## Architecture Overview

### Core Components

**Main Entry Points:**
- `langchain_chatbot.py` - Primary chatbot class with Streamlit web interface
- `run_langchain.py` - Command-line launcher with environment checking
- `config.py` - Main configuration with context-missing fixes
- `optimized_config.py` - Optimized RAG system configuration

**RAG System Architecture:**
1. **Traditional RAG**: Basic LangChain implementation (`src/langchain_rag.py`)
2. **Optimized RAG**: Enhanced 5-layer optimization system (`src/optimized_rag_system.py`)

### Key Features

**5 Core Optimizations:**
1. **Hybrid Retrieval**: BM25 + vector search with RRF fusion
2. **Cross-Encoder Reranking**: ms-marco-MiniLM-L6-v2 intelligent reordering  
3. **Entity Hit Rewards**: Keyword matching bonus in reranking
4. **Document Enhancement**: Chapter-based splitting with title concatenation
5. **Few-shot Prompting**: Intelligent context relevance checking

**Advanced Retrieval Components:**
- `src/advanced_hybrid_retriever.py` - Multi-stage retrieval with force recall
- `src/enhanced_bm25_retriever.py` - Enhanced BM25 implementation
- `src/hierarchical_retriever.py` - Hierarchical search capabilities
- `src/diversity_retriever.py` - Balanced document source retrieval

**Document Processing:**
- `src/enhanced_document_loader.py` - Advanced PDF processing
- `src/incremental_document_loader.py` - Delta loading support
- `src/document_version_manager.py` - Version tracking system

## Configuration Management

### Critical Configuration Parameters
```python
# Context-missing fixes (applied automatically)
MAX_CONTEXTS = 8              # Increased from 5 to prevent information loss
LANGCHAIN_RETRIEVER_K = 15     # Increased retrieval count
LANGCHAIN_HYBRID_ALPHA = 0.3   # Reduced vector weight, increased BM25 weight

# Retrieval stages optimization
RETRIEVAL_STAGES = {
    "stage1_vector_k": 40,      # Initial vector retrieval
    "stage1_bm25_k": 40,        # Initial BM25 retrieval  
    "stage2_candidate_k": 60,   # Candidate pool
    "final_k": 15               # Final results
}
```

### Model Configuration
- **LLM Model**: `deepseek-r1:7b` (primary), supports other Ollama models
- **Embedding Model**: `./bge-large-zh-v1.5` (local BGE model)
- **Ollama Base URL**: `http://localhost:11434`

## Data Flow

### Document Processing Pipeline
1. PDF files loaded from `data/` directory
2. Documents split into chunks with title enhancement
3. Vectorized using BGE embeddings
4. Stored in FAISS vector database (`vectorstore/`)

### Query Processing Pipeline
1. User question analyzed and enhanced
2. Multi-stage retrieval (vector + BM25)
3. Cross-Encoder reranking with entity rewards
4. Context compression and few-shot prompting
5. LLM generates final answer

## File Structure

```
├── data/                    # PDF documents (competition rules)
├── vectorstore/             # FAISS vector database
├── src/                     # Core modules
│   ├── optimized_rag_system.py     # Enhanced RAG with 5 optimizations
│   ├── langchain_rag.py           # Traditional RAG implementation
│   ├── advanced_hybrid_retriever.py # Multi-stage retrieval
│   ├── enhanced_document_loader.py # PDF processing with titles
│   └── [30+ specialized modules]
├── logs/                    # Application logs
├── outputs/                 # Generated results
├── bge-large-zh-v1.5/       # Local embedding model
├── cross-encoder/           # Reranking model
└── knowledge_base/          # Knowledge base storage
```

## Development Guidelines

### Working with the RAG System
- Default to using `use_optimized=True` for new features
- Test both traditional and optimized modes when making changes
- The system automatically applies context-missing fixes on startup

### Adding New Retrieval Components
- Inherit from appropriate base classes in `src/`
- Follow the established pattern for retrieval stages
- Update `RETRIEVAL_STAGES` configuration when modifying retrieval counts

### Configuration Changes
- Modify `config.py` for system-wide settings
- Use `Config.apply_context_missing_fixes()` for critical parameter updates
- Test parameter changes with the optimized RAG system

### Testing
- Use `test_optimized_rag_system.py` for optimization-specific tests
- Run `python run_langchain.py --check` to validate environment
- Test with both web and CLI interfaces

## Common Issues and Solutions

### Ollama Connection Issues
- Ensure Ollama service is running: `ollama serve`
- Verify model availability: `ollama list`
- Check port 11434 accessibility

### Vector Database Issues
- System automatically loads existing vectorstore on startup
- Use `rebuild_knowledge_base()` for complete reconstruction
- Check `vectorstore/` directory permissions

### Memory Optimization
- Monitor `MAX_CONTEXTS` and retrieval parameters
- Adjust chunk sizes in `CHUNK_SIZE` configuration
- Use incremental loading for large document sets

## Model and Performance Notes

### Recommended Models
- **Primary**: `deepseek-r1:7b` (best balance of performance and capability)
- **Alternative**: `qwen2:7b`, `chatglm3:6b`
- **Embedding**: Local BGE model (`./bge-large-zh-v1.5`)

### Performance Tuning
- The optimized RAG system includes automatic performance optimizations
- Multi-stage retrieval reduces computational overhead
- Context compression prevents information loss while managing memory

## Integration Points

### Document Monitoring
- Automatic file watching in `data/` directory
- Real-time knowledge base updates
- Version management with rollback capabilities

### Batch Processing
- Excel-based question batch processing
- Progress tracking with callbacks
- Result export with confidence scoring


# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

## Process

### 1. Planning & Staging

Break complex work into 3-5 stages. Document in `IMPLEMENTATION_PLAN.md`:

```markdown
## Stage N: [Name]
**Goal**: [Specific deliverable]
**Success Criteria**: [Testable outcomes]
**Tests**: [Specific test cases]
**Status**: [Not Started|In Progress|Complete]
```
- Update status as you progress
- Remove file when all stages are done

### 2. Implementation Flow

1. **Understand** - Study existing patterns in codebase
2. **Test** - Write test first (red)
3. **Implement** - Minimal code to pass (green)
4. **Refactor** - Clean up with tests passing
5. **Commit** - With clear message linking to plan

### 3. When Stuck (After 3 Attempts)

**CRITICAL**: Maximum 3 attempts per issue, then STOP.

1. **Document what failed**:
   - What you tried
   - Specific error messages
   - Why you think it failed

2. **Research alternatives**:
   - Find 2-3 similar implementations
   - Note different approaches used

3. **Question fundamentals**:
   - Is this the right abstraction level?
   - Can this be split into smaller problems?
   - Is there a simpler approach entirely?

4. **Try different angle**:
   - Different library/framework feature?
   - Different architectural pattern?
   - Remove abstraction instead of adding?

## Technical Standards

### Architecture Principles

- **Composition over inheritance** - Use dependency injection
- **Interfaces over singletons** - Enable testing and flexibility
- **Explicit over implicit** - Clear data flow and dependencies
- **Test-driven when possible** - Never disable tests, fix them

### Code Quality

- **Every commit must**:
  - Compile successfully
  - Pass all existing tests
  - Include tests for new functionality
  - Follow project formatting/linting

- **Before committing**:
  - Run formatters/linters
  - Self-review changes
  - Ensure commit message explains "why"

### Error Handling

- Fail fast with descriptive messages
- Include context for debugging
- Handle errors at appropriate level
- Never silently swallow exceptions

## Decision Framework

When multiple valid approaches exist, choose based on:

1. **Testability** - Can I easily test this?
2. **Readability** - Will someone understand this in 6 months?
3. **Consistency** - Does this match project patterns?
4. **Simplicity** - Is this the simplest solution that works?
5. **Reversibility** - How hard to change later?

## Project Integration

### Learning the Codebase

- Find 3 similar features/components
- Identify common patterns and conventions
- Use same libraries/utilities when possible
- Follow existing test patterns

### Tooling

- Use project's existing build system
- Use project's test framework
- Use project's formatter/linter settings
- Don't introduce new tools without strong justification

## Quality Gates

### Definition of Done

- [ ] Tests written and passing
- [ ] Code follows project conventions
- [ ] No linter/formatter warnings
- [ ] Commit messages are clear
- [ ] Implementation matches plan
- [ ] No TODOs without issue numbers

### Test Guidelines

- Test behavior, not implementation
- One assertion per test when possible
- Clear test names describing scenario
- Use existing test utilities/helpers
- Tests should be deterministic

## Important Reminders

**NEVER**:
- Use `--no-verify` to bypass commit hooks
- Disable tests instead of fixing them
- Commit code that doesn't compile
- Make assumptions - verify with existing code

**ALWAYS**:
- Commit working code incrementally
- Update plan documentation as you go
- Learn from existing implementations
