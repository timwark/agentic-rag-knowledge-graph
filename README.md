# Agentic RAG with Knowledge Graph

Agentic knowledge retrieval redefined with a **multi-agent AI system** that combines traditional RAG (vector search) with knowledge graph capabilities. Each agent maintains **isolated vector stores** for specialized domains while sharing the same knowledge graph. The system uses PostgreSQL with pgvector for semantic search and Neo4j with Graphiti for temporal knowledge graphs. The goal is to create Agentic RAG at its finest with **multi-agent vector store isolation**.

Built with:

- Pydantic AI for the AI Agent Framework
- Graphiti for the Knowledge Graph
- Postgres with PGVector for the Vector Database
- LlamaIndex Ingestion System for document ingestion for RAG Vector Search 
- Neo4j for the Knowledge Graph Engine (Graphiti connects to this)
- FastAPI for the Agent API
- Claude Code for the AI Coding Assistant (See `CLAUDE.md`, `PLANNING.md`, and `TASK.md`)

## Overview

This system includes three main components:

1. **Document Ingestion Pipeline**: Processes multiple document formats (PDF, DOCX, PPTX, MD) using LlamaIndex OSS patterns with semantic chunking and builds both vector embeddings and knowledge graph relationships
2. **AI Agent Interface**: A conversational agent powered by Pydantic AI that can search across both vector database and knowledge graph
3. **Streaming API**: FastAPI backend with real-time streaming responses and comprehensive search capabilities

## 🚀 Multi-Agent Vector Store Architecture

### What is Multi-Agent Support?

This system supports **multiple isolated agents**, each with their own dedicated vector stores while sharing a common knowledge graph. This architecture enables:

- **Domain Specialization**: Different agents for different topics (e.g., `tech_agent`, `finance_agent`, `research_agent`)
- **Data Isolation**: Each agent's documents and chunks are completely isolated from other agents
- **Shared Knowledge**: All agents can access the same temporal knowledge graph for relationship queries
- **Easy Switching**: Command-line control to specify which agent to use for both server and client

### Agent Naming Convention

- **Agent Names**: Must be alphanumeric with underscores only (e.g., `main_agent`, `test_agent`, `tech_research`)
- **Database Tables**: Each agent gets dedicated tables: `agent_{name}_documents` and `agent_{name}_chunks`
- **Default Agent**: `main_agent` (configurable via `.env` or command line)

### Example Use Cases

```bash
# Financial analysis agent with finance documents
python ingestion/ingest.py --agent-name finance_agent --local-folder finance_docs/

# Technology research agent with tech papers  
python ingestion/ingest.py --agent-name tech_agent --local-folder tech_research/

# General purpose agent with mixed content
python ingestion/ingest.py --agent-name main_agent --local-folder documents/
```

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database (such as Neon)
- Neo4j database (for knowledge graph)
- LLM Provider API key (OpenAI, Ollama, Gemini, etc.)

## Installation

### 1. Set up a virtual environment

```bash
# Create and activate virtual environment
python -m venv venv       # python3 on Linux
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up required tables in Postgres

Execute the SQL in `sql/schema.sql` to create all necessary tables, indexes, and functions.

Be sure to change the embedding dimensions on lines 31, 67, and 100 based on your embedding model. OpenAI's text-embedding-3-small is 1536 and nomic-embed-text from Ollama is 768 dimensions, for reference.

Note that this script will drop all tables before creating/recreating!

### 4. Set up Neo4j

You have a couple easy options for setting up Neo4j:

#### Option A: Using Local-AI-Packaged (Simplified setup - Recommended)
1. Clone the repository: `git clone https://github.com/coleam00/local-ai-packaged`
2. Follow the installation instructions to set up Neo4j through the package
3. Note the username and password you set in .env and the URI will be bolt://localhost:7687

#### Option B: Using Neo4j Desktop
1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and add a local DBMS
3. Start the DBMS and set a password
4. Note the connection details (URI, username, password)

### 5. Configure environment variables

Create a `.env` file in the project root:

```bash
# Multi-Agent Configuration
DEFAULT_AGENT_NAME=main_agent
DEFAULT_INGESTION_FOLDER=documents/
AGENT_TABLE_PREFIX=agent_

# Database Configuration (example Neon connection string)
DATABASE_URL=postgresql://username:password@ep-example-12345.us-east-2.aws.neon.tech/neondb

# Neo4j Configuration  
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Provider Configuration (choose one)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
LLM_CHOICE=gpt-4.1-mini

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-api-key
EMBEDDING_MODEL=text-embedding-3-small

# Ingestion Configuration
INGESTION_LLM_CHOICE=gpt-4.1-nano  # Faster model for processing

# Application Configuration
APP_ENV=development
LOG_LEVEL=INFO
APP_PORT=8058
```

For other LLM providers:
```bash
# Ollama (Local)
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_CHOICE=qwen2.5:14b-instruct

# OpenRouter
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-openrouter-key
LLM_CHOICE=anthropic/claude-3-5-sonnet

# Gemini
LLM_PROVIDER=gemini
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
LLM_API_KEY=your-gemini-key
LLM_CHOICE=gemini-2.5-flash
```

## Quick Start

### 1. Prepare Your Documents

Add your documents to the `documents/` folder. The system supports multiple formats:

```bash
mkdir -p documents
# Add your documents in supported formats:
# - PDF files (reports, research papers)
# - DOCX files (Word documents)
# - PPTX files (PowerPoint presentations)  
# - MD files (markdown documents)
# Example: documents/google_ai_initiatives.pdf
#          documents/microsoft_openai_partnership.docx
#          documents/ai_research_overview.md
```

**Note**: For a comprehensive example with extensive content, you can copy the provided `big_tech_docs` folder:
```bash
cp -r big_tech_docs/* documents/
```
This includes 21 detailed documents about major tech companies and their AI initiatives. Be aware that processing all these files into the knowledge graph will take significant time (potentially 30+ minutes) due to the computational complexity of entity extraction and relationship building.

### 2. Run Document Ingestion

**Important**: You must run ingestion first to populate the databases before the agent can provide meaningful responses.

#### Basic Multi-Agent Ingestion

```bash
# Ingest documents for the default agent (main_agent)
python ingestion/ingest.py --local-folder documents/

# Ingest documents for a specific agent
python ingestion/ingest.py --agent-name test_agent --local-folder test_docs/

# Create specialized agents for different domains
python ingestion/ingest.py --agent-name finance_agent --local-folder finance_docs/
python ingestion/ingest.py --agent-name tech_agent --local-folder tech_research/
```

#### Advanced Ingestion Options

```bash
# Clean existing data for an agent and re-ingest everything
python ingestion/ingest.py --agent-name test_agent --local-folder documents/ --clean

# Process specific file types only for an agent
python ingestion/ingest.py --agent-name research_agent --local-folder docs/ --file-types pdf,docx

# Custom settings for different processing modes
python ingestion/ingest.py --agent-name main_agent --local-folder documents/ --ingestion-mode vector_only --verbose
python ingestion/ingest.py --agent-name graph_agent --local-folder documents/ --ingestion-mode graph_only --chunk-size 800
python ingestion/ingest.py --agent-name hybrid_agent --local-folder documents/ --ingestion-mode both --chunk-size 1000 --chunk-overlap 200
```

**Ingestion Parameters:**
- `--agent-name`: Agent name for vector store isolation (overrides .env DEFAULT_AGENT_NAME)
- `--local-folder`/`--ingestion-folder`: Path to folder containing documents (overrides .env DEFAULT_INGESTION_FOLDER)
- `--file-types`: Comma-separated list of file extensions (default: pdf,docx,pptx,md)
- `--ingestion-mode`: Choose vector_only, graph_only, or both (default: both)
- `--chunk-size`: Text chunk size for splitting (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--clean`: Clean existing data before ingestion
- `--verbose`: Enable verbose logging

The ingestion process will:
- Load documents using specialized LlamaIndex readers for each format
- Parse and semantically chunk your documents using SentenceSplitter
- Generate embeddings for vector search using batch processing
- Extract entities and relationships for the knowledge graph
- Store everything in PostgreSQL and Neo4j

**Performance Notes:**
- The system uses batch processing for efficient embedding generation
- Rate limiting prevents API quota issues
- PDF processing is optimized with PyMuPDF for fast text extraction
- Knowledge graph building can be time-intensive for large document sets

### 3. Configure Agent Behavior (Optional)

Before running the API server, you can customize when the agent uses different tools by modifying the system prompt in `agent/prompts.py`. The system prompt controls:
- When to use vector search vs knowledge graph search
- How to combine results from different sources
- The agent's reasoning strategy for tool selection

### 4. Start the Multi-Agent API Server (Terminal 1)

```bash
# Start the server with the default agent (main_agent)
python agent/api.py

# Start the server with a specific agent
python agent/api.py --agent-name test_agent

# Start the server with custom host and port
python agent/api.py --agent-name finance_agent --host 0.0.0.0 --port 8080

# Server will be available at the specified host:port (default: http://localhost:8058)
```

**Server Command Line Options:**
- `--agent-name`: Agent to use for all API operations (overrides .env DEFAULT_AGENT_NAME)
- `--host`: Host to bind the server to (default: from .env APP_HOST)
- `--port`: Port to bind the server to (default: from .env APP_PORT)

### 5. Use the Multi-Agent Command Line Interface (Terminal 2)

The CLI provides an interactive way to chat with the agent and see which tools it uses for each query.

```bash
# Start the CLI with the default agent
python cli.py

# Start the CLI with a specific agent
python cli.py --agent-name test_agent

# Connect to a different server with specific agent
python cli.py --agent-name finance_agent --url http://localhost:8080

# Connect to a specific port with agent
python cli.py --agent-name tech_agent --port 8080
```

**Client Command Line Options:**
- `--agent-name`: Agent to use for all queries (overrides .env DEFAULT_AGENT_NAME)
- `--url`: Base URL for the API (default: http://localhost:8058)
- `--port`: Port number (overrides URL port)

> **Important**: The server and client agent names should match! If the server is running with `--agent-name test_agent`, the client should also use `--agent-name test_agent` to access the same vector store.

#### CLI Features

- **Real-time streaming responses** - See the agent's response as it's generated
- **Tool usage visibility** - Understand which tools the agent used:
  - `vector_search` - Semantic similarity search
  - `graph_search` - Knowledge graph queries
  - `hybrid_search` - Combined search approach
- **Session management** - Maintains conversation context
- **Color-coded output** - Easy to read responses and tool information

#### Example CLI Session

```
🤖 Agentic RAG with Knowledge Graph CLI
============================================================
Connected to: http://localhost:8058
📊 Agent: test_agent

You: What are Microsoft's AI initiatives?

🤖 Assistant:
Microsoft has several major AI initiatives including...

🛠 Tools Used:
  1. vector_search (query='Microsoft AI initiatives', limit=10)
  2. graph_search (query='Microsoft AI projects')

────────────────────────────────────────────────────────────

You: How is Microsoft connected to OpenAI?

🤖 Assistant:
Microsoft has a significant strategic partnership with OpenAI...

🛠 Tools Used:
  1. hybrid_search (query='Microsoft OpenAI partnership', limit=10)
  2. get_entity_relationships (entity='Microsoft')
```

#### CLI Commands

- `help` - Show available commands
- `health` - Check API connection status
- `clear` - Clear current session
- `exit` or `quit` - Exit the CLI

## 🤖 Working with Multiple Agents

### Complete Multi-Agent Workflow

Here's a complete example of setting up and using multiple specialized agents:

#### Step 1: Create Different Agents with Specialized Data

```bash
# Create a technology research agent
python ingestion/ingest.py --agent-name tech_agent --local-folder tech_docs/

# Create a financial analysis agent  
python ingestion/ingest.py --agent-name finance_agent --local-folder finance_reports/

# Create a general research agent
python ingestion/ingest.py --agent-name research_agent --local-folder academic_papers/
```

#### Step 2: Start Server for Specific Agent

```bash
# Terminal 1: Start server for tech agent
python agent/api.py --agent-name tech_agent

# The server will display:
# 🤖 Starting Agentic RAG API Server
# 📊 Agent: tech_agent
# 🌐 Host: 0.0.0.0:8058
```

#### Step 3: Connect Client to the Same Agent

```bash
# Terminal 2: Connect CLI to tech agent
python cli.py --agent-name tech_agent

# The CLI will display:
# 🤖 Agentic RAG with Knowledge Graph CLI
# Connected to: http://localhost:8058
# 📊 Agent: tech_agent
```

#### Step 4: Switch to Different Agent

```bash
# Stop the current server (Ctrl+C)
# Start server for finance agent
python agent/api.py --agent-name finance_agent

# In CLI terminal, restart with finance agent
python cli.py --agent-name finance_agent
```

### Agent Data Isolation

Each agent maintains completely isolated vector stores:

```bash
# tech_agent data is stored in:
# - agent_tech_agent_documents table
# - agent_tech_agent_chunks table

# finance_agent data is stored in:
# - agent_finance_agent_documents table  
# - agent_finance_agent_chunks table

# main_agent data is stored in:
# - agent_main_agent_documents table
# - agent_main_agent_chunks table
```

### Shared Knowledge Graph

All agents share the same Neo4j knowledge graph, enabling:
- Cross-agent entity relationship queries
- Temporal knowledge across all domains
- Unified fact tracking regardless of which agent ingested the data

### 6. Test the System

#### Health Check
```bash
curl http://localhost:8058/health
```

#### Chat with the Agent (Non-streaming)
```bash
curl -X POST "http://localhost:8058/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are Google'\''s main AI initiatives?"
  }'
```

#### Streaming Chat
```bash
curl -X POST "http://localhost:8058/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Compare Microsoft and Google'\''s AI strategies",
  }'
```

## How It Works

### The Power of Hybrid RAG + Knowledge Graph

This system combines the best of both worlds:

**Vector Database (PostgreSQL + pgvector)**:
- Semantic similarity search across document chunks
- Fast retrieval of contextually relevant information
- Excellent for finding documents about similar topics

**Knowledge Graph (Neo4j + Graphiti)**:
- Temporal relationships between entities (companies, people, technologies)
- Graph traversal for discovering connections
- Perfect for understanding partnerships, acquisitions, and evolution over time

**Intelligent Agent**:
- Automatically chooses the best search strategy
- Combines results from both databases
- Provides context-aware responses with source citations

### Example Queries

The system excels at queries that benefit from both semantic search and relationship understanding:

- **Semantic Questions**: "What AI research is Google working on?" 
  - Uses vector search to find relevant document chunks about Google's AI research

- **Relationship Questions**: "How are Microsoft and OpenAI connected?"
  - Uses knowledge graph to traverse relationships and partnerships

- **Temporal Questions**: "Show me the timeline of Meta's AI announcements"
  - Leverages Graphiti's temporal capabilities to track changes over time

- **Complex Analysis**: "Compare the AI strategies of FAANG companies"
  - Combines vector search for strategy documents with graph traversal for competitive analysis

### Why This Architecture Works So Well

1. **Complementary Strengths**: Vector search finds semantically similar content while knowledge graphs reveal hidden connections

2. **Temporal Intelligence**: Graphiti tracks how facts change over time, perfect for the rapidly evolving AI landscape

3. **LlamaIndex Integration**: Follows OSS best practices with specialized readers for each document format, semantic chunking, and efficient batch processing

4. **Multi-Format Support**: Handles PDF, DOCX, PPTX, and MD files with optimized processing for each format

5. **Flexible LLM Support**: Switch between OpenAI, Ollama, OpenRouter, or Gemini based on your needs

6. **Production Ready**: Comprehensive testing, error handling, and monitoring

## API Documentation

Visit http://localhost:8058/docs for interactive API documentation once the server is running.

## Key Features

- **Multi-Format Document Processing**: Supports PDF, DOCX, PPTX, and MD files with specialized LlamaIndex readers
- **Hybrid Search**: Seamlessly combines vector similarity and graph traversal
- **LlamaIndex Integration**: Follows OSS best practices for document ingestion and processing
- **Temporal Knowledge**: Tracks how information changes over time
- **Streaming Responses**: Real-time AI responses with Server-Sent Events
- **Batch Processing**: Efficient embedding generation with rate limiting
- **Flexible Providers**: Support for multiple LLM and embedding providers
- **Semantic Chunking**: Intelligent document splitting using SentenceSplitter
- **Production Ready**: Comprehensive testing, logging, and error handling

## Project Structure

```
agentic-rag-knowledge-graph/
├── agent/                  # AI agent and API
│   ├── agent.py           # Main Pydantic AI agent
│   ├── api.py             # FastAPI application
│   ├── providers.py       # LLM provider abstraction
│   └── models.py          # Data models
├── ingestion/             # Document processing
│   ├── ingest.py                        # LlamaIndex-based ingestion pipeline
│   └── llamaindex_document_processor.py # Document processor
├── sql/                   # Database schema
├── documents/             # Your documents (PDF, DOCX, PPTX, MD)
├── big_tech_docs/         # Example markdown documents
└── tests/                # Comprehensive test suite
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent --cov=ingestion --cov-report=html

# Run specific test categories
pytest tests/agent/
pytest tests/ingestion/
```

## Troubleshooting

### Common Issues

**Database Connection**: Ensure your DATABASE_URL is correct and the database is accessible
```bash
# Test your connection
psql -d "$DATABASE_URL" -c "SELECT 1;"
```

**Neo4j Connection**: Verify your Neo4j instance is running and credentials are correct
```bash
# Check if Neo4j is accessible (adjust URL as needed)
curl -u neo4j:password http://localhost:7474/db/data/
```

**No Results from Agent**: Make sure you've run the ingestion pipeline first
```bash
python -m ingestion.ingest --local-folder documents/ --verbose
```

**File Format Issues**: Ensure your documents are in supported formats and properly formatted
- PDF files should be text-based (not scanned images)
- DOCX files should be standard Word documents
- PPTX files should contain text content
- MD files should use standard markdown syntax

**LLM API Issues**: Check your API key and provider configuration in `.env`

---

Built with ❤️ using Pydantic AI, FastAPI, PostgreSQL, and Neo4j.