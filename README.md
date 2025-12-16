# Bumblebee AI - Multi-Agent Chat System

A FastAPI-powered multi-agent AI chatbot system with specialized agents for Salesforce operations, GIS mapping, data export, and intelligent query routing. Features real-time WebSocket communication and a modern ChatGPT-style interface.

----

## Features

- **Multi-Agent Architecture**: Specialized agents for different domains
  - 🔹 **Salesforce Agent**: SOQL queries, object discovery, data analysis
  - 🗺️ **GIS Agent**: Geocoding, mapping, spatial analysis with ArcGIS integration
  - 📊 **Data Agent**: CSV exports and file management
  - 🤖 **Root Agent**: General chat and intelligent routing

- **Modern Chat Interface**: ChatGPT-style UI with sidebar sessions
- **Real-time Communication**: WebSocket-based messaging
- **Intelligent Routing**: Automatic agent selection based on query analysis
- **Session Management**: Persistent chat history and session tracking
- **Tool Integration**: 15+ GIS tools and comprehensive Salesforce operations

----

## Project Structure

```
bumblebee/
├── main.py                     # FastAPI application entry point
├── chat_bot.html              # Modern ChatGPT-style web interface
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration
├── .env.example              # Environment template
├── sessions.db               # SQLite database (auto-created)
├── agents/                   # Multi-agent system
│   ├── agent_registry.py     # Central agent registry
│   ├── root_agent.py        # Main routing agent
│   ├── salesforce_agent.py  # Salesforce operations
│   └── gis_agent.py         # GIS and mapping operations
├── tools/                    # Agent tools and capabilities
│   ├── salesforce_tools.py  # Salesforce API tools
│   ├── gis_tools.py         # ArcGIS integration tools
│   └── salesforce_adk_tools.py # Google ADK tool wrappers
├── utils/                    # Utilities and helpers
│   ├── error_handlers.py    # Error handling and logging
│   ├── salesforce_auth.py   # Salesforce authentication
│   └── arcgis_auth.py       # ArcGIS authentication
└── services/                # Business logic
    ├── session_service.py   # Session management
    └── websocket_manager.py # WebSocket connections
```

----

## Usage Guide

### Example Queries

**Salesforce Operations:**
```
- "Show me all Account records with more than 1000 employees"
- "Get opportunity data for Q4 2024"
- "Export all contacts to CSV"
```

**GIS Operations:**
```
- "Geocode this address: 123 Main St, San Francisco, CA"
- "Show sales territories on a map"
- "Find the nearest stores to this location"
```

**Data Operations:**
```
- "Export all account data to CSV"
- "Download the lead report"
```

### API Endpoints

**WebSocket Connection:**
```javascript
const socket = new WebSocket('ws://localhost:8000/ws');
socket.send(JSON.stringify({
    type: 'query',
    query: 'Show me recent opportunities',
    session_id: 'your-session-id'
}));
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

----

## Configuration Details

### Required API Keys

1. **OpenAI API Key**: Get from https://platform.openai.com/api-keys
2. **Salesforce Credentials**: Your Salesforce username, password, security token
3. **ArcGIS API Key**: Get from https://developers.arcgis.com/

### Optional Configurations

- **Database**: Defaults to SQLite, can be configured for PostgreSQL
- **Logging**: Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- **CORS**: Configure allowed origins for web interface

----

## Development

### Running in Development Mode

```bash
# Create Virtual Environment
uv venv venv

# Install development dependencies
uv sync

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Enable debug logging
export LOG_LEVEL=DEBUG
```

### Adding New Agents

1. Create new agent file in `agents/` directory
2. Implement `BaseAgent` interface
3. Register in `agent_registry.py`
4. Add routing keywords if needed

----

## Troubleshooting

### Common Issues

**Connection Refused:**
- Ensure the server is running on port 8000
- Check if port is already in use: `lsof -i :8000`

**API Key Errors:**
- Verify all API keys are correctly set in `.env`
- Check API key permissions and quotas

**Database Errors:**
- Delete `sessions.db` and restart to recreate database
- Check file permissions in project directory

**Import Errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

### Logs

Application logs are available in:
- Console output (when running with `--reload`)
- `agent_errors.log` file for error tracking

### Support

For issues and feature requests:
1. Check existing issues in the repository
2. Create new issue with detailed description
3. Include error logs and configuration (without sensitive data)

----

## Architecture Overview

The Bumblebee system uses a multi-agent architecture where:

1. **Query Reception**: WebSocket receives user queries
2. **Agent Routing**: Smart routing based on query analysis
3. **Agent Processing**: Specialized agents handle domain-specific tasks
4. **Tool Execution**: Agents use tools for external API calls
5. **Response Generation**: Structured responses sent back to client

Each agent is self-contained and can be developed/deployed independently while sharing common utilities and error handling patterns.


