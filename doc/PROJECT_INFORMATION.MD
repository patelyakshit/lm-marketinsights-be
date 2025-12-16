# Project LM Multi Agent - Comprehensive Documentation

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)

**A Production-Grade Multi-Agent AI Chatbot System with Real-Time Audio/Text Streaming**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Concepts](#3-core-concepts)
4. [Key Features](#4-key-features)
5. [API Reference](#5-api-reference)
6. [Getting Started](#6-getting-started)
7. [Agent System Deep Dive](#7-agent-system-deep-dive)
8. [Audio Streaming Architecture](#8-audio-streaming-architecture)
9. [Event Handling System](#9-event-handling-system)
---

## 1. Introduction

### 1.1 Problem Statement

Modern enterprise applications require intelligent conversational AI that can:
- Handle complex multi-domain queries requiring specialized knowledge
- Process both text and voice interactions in real-time
- Integrate with multiple enterprise systems (CRM, GIS, databases)
- Scale to handle thousands of concurrent users
- Provide reliable, observable, and maintainable architecture

Traditional monolithic chatbot solutions struggle with:
- **Domain Expertise**: Single models lack specialized knowledge across domains
- **Tool Integration**: Difficult to manage multiple API integrations
- **Scalability**: Single-agent architectures hit performance bottlenecks
- **Reliability**: No fallback mechanisms when primary services fail
- **Observability**: Limited visibility into agent decision-making and performance

### 1.2 Solution: Project Multi Agent 

LM Multi Agent is a **production-grade, multi-agent AI chatbot system** built on Google's Agent Development Kit (ADK) that provides:

 **Multi-Agent Architecture**: Specialized agents (GIS, Salesforce, General) with intelligent routing
 **Dual Streaming Modes**: Server-Sent Events (SSE) for text, Bidirectional (BIDI) for audio
 **Hybrid LLM Strategy**: Primary Gemini models with fallback support
 **Enterprise Integration**: Salesforce, ArcGIS, Azure Blob Storage, PostgreSQL
 **Production-Ready**: OpenTelemetry observability, async/await patterns, comprehensive error handling
 **Real-Time Audio**: 24kHz PCM audio streaming with deduplication and turn management

### 1.3 Use Cases

**Customer Support**: Voice-enabled support agent with CRM integration for account lookups and case management
**Sales Assistance**: Location-aware sales agent with GIS mapping for territory planning and lead routing
**Technical Support**: Multi-tool technical agent with system diagnostics and knowledge base search
**Field Service**: Mobile-friendly voice interface for field technicians with real-time data access

### 1.4 Technology Stack

| Layer | Technology                   | Purpose |
|-------|------------------------------|---------|
| **Framework** | FastAPI 0.104+               | Async web framework with WebSocket support |
| **AI/ML** | Google ADK, Gemini 2.5 Flash | Multi-agent orchestration and LLM inference |
| **Database** | PostgreSQL 14+, SQLAlchemy   | User data, sessions, agent configs |
| **Message Queue** | RabbitMQ 3.12+               | Event streaming |
| **Observability** | OpenTelemetry, Grafana Tempo | Distributed tracing, metrics |
| **Storage** | Azure Blob Storage           | File uploads, conversation logs |
| **Authentication** | JWT, OAuth2                  | Token-based authentication |
| **Deployment** | Azure | Containerization and orchestration |

---

## 2. Architecture Overview

### 2.1 System Architecture

```
CLIENT LAYER
  Web/Mobile/Voice Apps
         ↓
  WebSocket/HTTP
         ↓
FASTAPI APPLICATION
  WebSocket Manager → UnifiedEventHandler → RootAgent
                                               ↓
                                    ┌──────────┴──────────┐
                                    ↓                     ↓
                              GISAgent            SalesforceAgent
                                                         ↓
                                                  AddressPatternAgent
         ↓
  Google ADK (Runner, Session Service)
         ↓
  Gemini 2.5 Flash + External APIs (Salesforce, ArcGIS)
```

### 2.2 Request Flow

**SSE Mode (Text):**
```
Client → WebSocket → RootAgent.process_query()
  → Runner.run_async(StreamingMode.SSE)
  → UnifiedEventHandler (mode="SSE")
  → Events: text chunks, tool results, completion
  → Client receives streaming response
```

**BIDI Mode (Audio):**
```
Client → Audio chunks → LiveRequestQueue
  → RootAgent.start_audio_streaming()
  → Runner.run_live(StreamingMode.BIDI)
  → UnifiedEventHandler (mode="BIDI")
  → Events: audio, transcription, turn status
  → Client plays audio
```

---

## 3. Core Concepts

### 3.1 Multi-Agent System

**Architecture:**
- **RootAgent**: Orchestrator using `sub_agents` parameter
- **GISAgent**: Map operations (15 tools)
- **SalesforceAgent**: CRM operations (12 tools + AddressPatternAgent)
- **AddressPatternAgent**: Address analysis (nested agent-as-tool)

**Key Principles:**
- Direct `LlmAgent` inheritance (no custom base class)
- Sub-agent coordination via ADK's `sub_agents` parameter
- Streaming callbacks for real-time visibility
- Model override via `SUB_AGENT_MODEL` env var

### 3.2 Streaming Modes

**SSE (Server-Sent Events):**
- Text-only streaming
- 3-state detection: streaming → last chunk → final consolidated
- Tool execution routing (GIS, Salesforce operations)
- Completion signal via `usage_metadata`

**BIDI (Bidirectional Audio):**
- Real-time audio streaming (16kHz input, 24kHz output)
- Non-blocking audio transmission via `asyncio.create_task()`
- Memory optimization: audio data stripped after sending
- Simple turn management: `turn_complete` and `interrupted` flags

### 3.3 Google ADK Integration

**What We Use:**
- `LlmAgent` base class for all agents
- `Runner` for session execution
- `SessionService` for memory management
- `FunctionTool` wrappers for tools
- Streaming callbacks for execution visibility

**What ADK Provides:**
- Agent lifecycle management
- Conversation memory persistence
- Tool execution framework
- Event streaming (SSE and BIDI modes)

**Example:**
```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

class GISAgent(LlmAgent):
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        super().__init__(
            model=model,
            name="gis_agent",
            description="GIS specialist",
            instruction=self._get_system_instruction(),
            tools=[geocode_tool, zoom_tool, ...],
        )
```

---

## 4. Key Features

### 4.1 Multi-Agent Coordination

**Pattern**: RootAgent uses `sub_agents=[SalesforceAgent(), GISAgent()]`

**Example Flow:**
```
User: "Map accounts in California"
  → RootAgent analyzes intent
  → Delegates to SalesforceAgent
  → SalesforceAgent executes export_query_to_geojson
  → Returns GeoJSON to client
```

**Benefits:**
- Automatic delegation (no manual routing code)
- Parallel execution where applicable
- Isolated failures per agent

### 4.2 Real-Time Audio Streaming

**Features:**
- 24kHz PCM audio output
- Non-blocking transmission
- Memory optimization (audio stripping)
- Turn management (complete/interrupted)

**Pipeline:**
```
Microphone → 16kHz PCM → Base64 → WebSocket
  → Gemini transcription → Processing
  → Gemini TTS → 24kHz PCM → Base64 → WebSocket
  → Audio playback
```

### 4.3 Tool Execution

**Pattern**: All tools are `FunctionTool` wrappers

```python
async def geocode_address(address: str) -> str:
    result = await gis_executor.geocode(address)
    return json.dumps(result, indent=0)

geocode_tool = FunctionTool(geocode_address)
```

**Tool Categories:**
- **GIS**: 15 tools (layers, filters, geocoding, analytics)
- **Salesforce**: 12 tools (SOQL, exports, geocoding)
- **Address Analysis**: 1 nested agent-as-tool

### 4.4 Observability

**What We Track:**
- Agent invocations
- Tool executions
- Streaming events
- Session memory usage

**Implementation:**
- Streaming callbacks (before/after tool, before/after model)
- Logging at agent and tool level
- Session cleanup metrics

---

## 5. API Reference

### 5.1 WebSocket Endpoints

#### **Endpoint 1: `/ws` - Text Streaming (SSE Mode)**

**Purpose**: Text-based chat with streaming responses
**Model**: `gemini-2.5-flash-lite` (TEXT_ROOT_AGENT)

**Query Parameters:**
- `session_id` (optional): Existing session UUID
- `token` / `jwt_token` / `drf_token`: Authentication token

**Supported Message Types:**

**Client → Server:**
```json
// Text message
{
  "type": "CHAT/SEND",
  "payload": {
    "message": "Show me accounts in California"
  }
}

// Map context update
{
  "type": "CHAT/MAP_CONTEXT",
  "payload": {
    "map_context": { /* map state */ }
  }
}

// Map context patch (JSON Patch)
{
  "type": "CHAT/PATCH",
  "payload": {
    "patched_data": [ /* JSON Patch operations */ ]
  }
}

// Ping
{
  "type": "CHAT/PING"
}
```

**Server → Client:**
```json
// Session established
{
  "type": "CHAT/SESSION",
  "payload": {
    "sessionId": "uuid",
    "connection_id": "conn_123",
    "streaming_enabled": true
  }
}

// Streaming text response
{
  "type": "CHAT/RECEIVE",
  "payload": { /* streamed response */ }
}

// Pong
{
  "type": "CHAT/PONG"
}

{
    "type": "CHAT/OPERATION_DATA",
    "payload": {
        "operations": [
            {
                "type": "TOGGLE_LAYER_VISIBILITY",
                "payload": {
                    "layerId": "196f11afada-layer-12",
                    "layerName": "Stores",
                    "visible": false
                }
            }
        ]
    }
}

{
    "type": "CHAT/OPERATION_DATA",
    "payload": {
        "operations": [
            {
                "type": "ZOOM_MAP",
                "payload": {
                    "zoom_action": "zoom_in",
                    "zoom_percentage": 20
                }
            }
        ]
    }
}

{
    "type": "CHAT/OPERATION_DATA",
    "payload": {
        "operations": [
            {
                "type": "APPLY_FILTER",
                "payload": {
                    "layerId": "196f11afada-layer-12",
                    "whereClause": "Total_Revenue > 300000",
                    "spatialLock": false
                }
            }
        ]
    }
}

{
    "type": "CHAT/OPERATION_DATA",
    "payload": {
        "operations": [
            {
                "type": "ZOOM_TO_LOCATION",
                "payload": {
                    "latitude": 32.7779765,
                    "longitude": -96.7962148
                }
            }
        ]
    }
}

{
    "type": "CHAT/OPERATION_DATA",
    "payload": {
        "operations": [
            {
                "type": "SUGGEST_PIN",
                "payload": {
                    "pins": [
                        {
                            "id": "863b9666-3923-4aff-b9c2-c873476af49b",
                            "score": 100,
                            "address": "Denver, Colorado",
                            "latitude": 39.738453,
                            "longitude": -104.984853,
                            "note": ""
                        }
                    ]
                }
            }
        ]
    }
}


{
    "type": "CHAT/OPERATION_DATA",
    "payload": {
        "operations": [
            {
                "type": "REMOVE_PIN",
                "payload": {
                    "pinIds": [
                        "863b9666-3923-4aff-b9c2-c873476af49b"
                    ]
                }
            }
        ]
    }
}

{
    "type": "CHAT/OPERATION_DATA",
    "payload": {
        "operations": [
            {
                "type": "SUGGEST_PIN",
                "payload": {
                    "pins": [
                        {
                            "id": "863b9666-3923-4aff-b9c2-c873476af49b",
                            "score": 100,
                            "address": "Denver, Colorado",
                            "latitude": 39.738453,
                            "longitude": -104.984853,
                            "note": ""
                        }
                    ]
                }
            }
        ]
    }
}
```

---

#### **Endpoint 2: `/ws/audio` - Audio Streaming (BIDI Mode)**

**Purpose**: Real-time bidirectional audio conversation
**Model**: `gemini-live-2.5-flash-preview-native-audio` (AUDIO_ROOT_AGENT)

**Query Parameters:**
- `session_id` (optional): Existing session UUID
- `token` / `jwt_token` / `drf_token`: Authentication token

**Supported Message Types:**

**Client → Server:**
```json
// Audio chunk
{
  "type": "CHAT/SEND_AUDIO",
  "payload": {
    "audio_base64": "base64_encoded_pcm_16khz",
    "turn_complete": false
  }
}
// Map context update
{
  "type": "CHAT/MAP_CONTEXT",
  "payload": {
    "map_context": { /* map state */ }
  }
}

// Ping
{
  "type": "CHAT/PING"
}
```

**Server → Client:**
```json
// Stream ready (initial signal)
{
  "type": "CHAT/STREAM_READY",
  "payload": {
    "connection_id": "conn_123",
    "session_id": "uuid",
    "message": "Audio stream ready - you can start recording"
  }
}

// Audio chunk (24kHz PCM)
{
  "type": "CHAT/AUDIO_CHUNK",
  "payload": {
    "data": "base64_encoded_pcm_24khz",
    "is_final": false
  }
}

// Turn status
{
  "type": "CHAT/TURN_STATUS",
  "payload": {
    "turn_complete": true,
    "interrupted": false,
    "final_text": "Transcribed response"
  }
}

// Thinking indicator
{
  "type": "CHAT/THINKING",
  "payload": {
    "response": "Retrieving information..."
  }
}
```

---

### 5.2 Core Agent Methods

**RootAgent.process_query()** - Used by `/ws` endpoint for SSE text streaming
- Creates/retrieves session from `SessionService`
- Executes `Runner.run_async(StreamingMode.SSE)`
- Processes events with `UnifiedEventHandler(mode="SSE")`
- Streams responses via WebSocket

**RootAgent.start_audio_streaming()** - Used by `/ws/audio` endpoint for BIDI audio
- Creates/retrieves session from `SessionService`
- Executes `Runner.run_live(StreamingMode.BIDI)`
- Returns `(live_events, live_request_queue)` tuple
- Launches two concurrent tasks:
  - `agent_to_client_messaging`: Streams audio/text from agent
  - `client_to_agent_messaging`: Receives audio from client

---

### 5.3 Authentication

Both endpoints require authentication via query parameters:
- `token`: JWT token
- `jwt_token`: JWT token (alternative)
- `drf_token`: Django REST Framework token

**Authentication Flow:**
1. Client provides token in query parameters
2. Server validates token via `authenticate_websocket_connection()`
3. If invalid: Server sends `CHAT/ERROR` and closes connection (code 4001)
4. If valid: Server sends `CHAT/SESSION` with session details

---

## 6. Getting Started

### 6.1 Prerequisites

**System Requirements:**
- **OS**: Linux, macOS, or Windows with WSL2
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB available

**Required Software:**
| Component | Version | Installation |
|-----------|---------|--------------|
| Python | 3.12+ | `pyenv install 3.12` |
| PostgreSQL | 14+ | `brew install postgresql@14` |
| RabbitMQ | 3.12+ | `brew install rabbitmq` |
| Docker | 20.10+ | [docker.com](https://docker.com) |

**Required API Keys & Credentials:**
- **Google Cloud Platform**:
  - Gemini API Key (for ADK)
  - GCP Service Account JSON file
- **ArcGIS** (3 API keys required):
  - Geolocation API Key (address to coordinates, reverse geocoding)
  - GeoEnrichment API Key (demographic data, tapestry segmentation)
  - Places API Key (Points of Interest data)
- **Azure Blob Storage**:
  - Connection String or Account Name + Key
  - Container name for file exports
- **Authentication Service**:
  - Auth Service API Key (for WebSocket authentication)

---

### 6.2 Installation

#### 6.2.1 Clone Repository
```bash
git clone https://github.com/Attri-Inc/lm-multi-agent.git
cd lm-multi-agent
```

#### 6.2.2 Create Virtual Environment
```bash
uv venv --python 3.12
uv sync
```

#### 6.2.3 Environment Configuration

Create `.env` file:
```bash
cp .env.example .env
```

**Edit `.env` with your credentials:**

```bash
# Optional: set SUB_AGENT_MODEL to the desired model. default is gemini-2.5-flash-lite, 
ROOT_MODEL=gemini-2.5-flash-lite 

# Optional: set SUB_AGENT_MODEL to the desired model. default is gemini-2.5-flash.
SUB_AGENT_MODEL=gemini-2.5-flash

# ArcGIS Configuration
# Geocoding API Key - Used for address to coordinate conversion and reverse geocoding
ARCGIS_GEOLOCATION_API_KEY=your_arcgis_key


# Legacy API Key (for backward compatibility)
ARCGIS_API_KEY=
ARCGIS_PORTAL_URL=

# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=
RABBITMQ_PASSWORD=
RABBITMQ_VHOST=/
RABBITMQ_URL=amqp://localhost:5672/

# Authentication Service Configuration
AUTH_SERVICE_URL=http://localhost:8000/user/v2/validate-auth
AUTH_SERVICE_TIMEOUT=5
AUTH_SERVICE_MAX_RETRIES=3
AUTH_SERVICE_API_KEY=your_microservice_api_key_here

# Application settings
# LOG_LEVEL: Controls application-wide logging verbosity
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Default: INFO
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Azure Blob Storage Configuration (for file exports)
# All file exports (CSV, GeoJSON) are stored in PRIVATE Azure Blob Storage
# Files are accessed via pre-signed SAS URLs with time-limited access

# Azure Storage Using Account Name + Key
AZURE_STORAGE_ACCOUNT_NAME=
AZURE_STORAGE_ACCOUNT_KEY=

# Blob Storage Container Configuration (container is private by default)
AZURE_STORAGE_CONTAINER_NAME=salesforce-exports

# SAS URL expiry time in hours (pre-signed URLs expire after this time)
# Default: 24 hours
AZURE_STORAGE_SAS_EXPIRY_HOURS=24

# Audio Streaming Configuration
ENABLE_AUDIO_STREAMING=true
AUDIO_RECORDING_SAMPLE_RATE=16000
AUDIO_PLAYBACK_SAMPLE_RATE=24000
AUDIO_BUFFER_SIZE=4096
AUDIO_BUFFER_DURATION_MS=100
AUDIO_MAX_DURATION_SECONDS=60
AUDIO_CHUNK_SIZE=1024
AUDIO_MAX_MESSAGE_SIZE=65536
AUDIO_PING_INTERVAL=20
ENABLE_AUDIO_DEBUGGING=false
AUDIO_TARGET_LATENCY_MS=200
AUDIO_ADAPTIVE_BUFFERING=true

# Database
DATABASE_URL=postgresql+psycopg2://postgres:root@localhost:5432/lm-multi-agent

# Google ADK Settings
# If using Service Account JSON file, set GOOGLE_GENAI_USE_VERTEXAI=1
# GOOGLE_CLOUD_LOCATION and GOOGLE_CLOUD_PROJECT are required in case of Service Account JSON file
# GOOGLE_API_KEY does not work for certain models and incase service account details are not provided priority is Service Account JSON file > GOOGLE_API_KEY 
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_CLOUD_PROJECT=<Project Name>
GOOGLE_APPLICATION_CREDENTIALS=<Service Account Json File Location>
GOOGLE_API_KEY=<Google Vertex AI API Key>
```

---

## 7. Agent System Deep Dive

### 7.1 Agent Architecture

All agents in LM Multi Agent inherit directly from Google ADK's `LlmAgent` class, providing native integration with the ADK runtime:

```python
# agents/root_agent.py, gis_agent.py, salesforce_agent.py, address_pattern_agent.py
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool

class RootAgent(LlmAgent):
    """
    Pure ADK agent with sub-agent coordination.

    Inherits directly from LlmAgent (not a custom base class).
    Uses ADK's native sub_agents parameter for orchestration.
    """

    def __init__(
        self,
        model: str | Gemini = "gemini-2.5-flash-lite",
        allow_sub_agent_override: bool = True
    ):
        super().__init__(
            model=model,
            name="root_agent",
            description="Root orchestrator agent that coordinates specialized sub-agents",
            instruction=self._get_system_instruction(),
            sub_agents=[  # ADK-native delegation
                SalesforceAgent(model, allow_override=allow_sub_agent_override),
                GISAgent(model, allow_override=allow_sub_agent_override),
            ],
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
        )
```

**Key Architectural Principles:**

1. **Direct `LlmAgent` Inheritance**: No custom base class; agents use Google ADK's `LlmAgent` directly
2. **Sub-Agent Coordination**: RootAgent uses `sub_agents` parameter for automatic delegation
3. **Streaming Callbacks**: All agents register callbacks for real-time execution visibility
4. **Model Flexibility**: Model parameter accepts both string and `Gemini` object
5. **Override Mechanism**: Sub-agents can use `SUB_AGENT_MODEL` env var when `allow_override=True`

**Agent Hierarchy:**
```
RootAgent (Orchestrator)
├── SalesforceAgent
│   └── AddressPatternAgent (nested agent-as-tool)
└── GISAgent
```

### 7.2 RootAgent (Orchestrator)

**Purpose**: Primary entry point for all user queries; analyzes intent and coordinates specialized agents

**Location**: `agents/root_agent.py`

**Key Responsibilities:**
1. Query reception and intent analysis
2. Sub-agent coordination via ADK's `sub_agents` parameter
3. Session state management
4. Streaming mode orchestration (SSE for text, BIDI for audio)
5. Session memory cleanup

**Configuration:**
```python
root_agent = RootAgent(
    model=config("ROOT_MODEL", "gemini-2.5-flash-lite"),
    allow_sub_agent_override=True  # Allows sub-agents to use SUB_AGENT_MODEL
)
```

**System Instruction Highlights:**

The RootAgent's system instruction defines clear routing rules:

**Sub-Agent Routing:**
- **Salesforce Agent**: ALL Salesforce data operations including:
  - SOQL/SOSL queries, schema analysis, data exports
  - **Salesforce mapping** with built-in geocoding (via `export_query_to_geojson`)
  - Record counts, pagination, aggregations

- **GIS Agent**: Map operations for NON-Salesforce data:
  - Map navigation (zoom, pan, center)
  - Layer visibility & filtering
  - Standalone geocoding & reverse geocoding
  - Pin/marker management

**Routing Decision Flow:**
```
1. Parse Intent → Identify keywords & context
2. Route Decision:
   - Salesforce data? → Salesforce Agent
   - Map/GIS operation (non-Salesforce)? → GIS Agent
   - Ambiguous? → Ask clarification
   - Other? → Answer directly
3. Delegate → Pass complete context to chosen agent
4. Present Results → Format response clearly
```

**Important Rule**: Never split Salesforce mapping between agents. Salesforce Agent handles end-to-end mapping for Salesforce data.

**Voice Mode Guidelines (BIDI):**
- **Root Agent**: SOLE TTS authority in Voice Mode
  - Receives ALL sub-agent data before proceeding
  - Validates all data before TTS generation
  - Keeps responses brief (target: ~20 words)

- **Sub Agents**: NO direct TTS in Voice Mode
  - Return structured data only
  - MUST hand off to Root Agent

**Core Methods:**

**1. `process_query()` - SSE Text Streaming**
```python
async def process_query(
    self,
    query: str,
    session_id: str = "default",
    connection_id: str = "",
    extra_information: dict = {},
) -> str:
    """Process query using ADK Runner with SSE streaming"""

    # Setup session with DatabaseSessionService
    example_session = await session_service.create_session(
        app_name="root_agent_app",
        user_id=f"user_{session_id}",
        session_id=session_id,
    )

    # Update session state
    example_session.state["session_id"] = session_id
    example_session.state["connection_id"] = connection_id
    if extra_information.get("map_context"):
        example_session.state["map_context"] = extra_information.get("map_context")

    # Create runner
    runner = Runner(
        agent=self,
        app_name="root_agent_app",
        session_service=session_service,
    )

    # Run with SSE streaming
    content = types.Content(role="user", parts=[types.Part(text=query)])
    events = runner.run_async(
        user_id=f"user_{session_id}",
        session_id=example_session.id,
        new_message=content,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    # Process events with UnifiedEventHandler
    event_handler = UnifiedEventHandler(
        connection_id=connection_id,
        session_id=session_id,
        manager=manager,
        mode="SSE",
    )

    async for event in events:
        await event_handler.handle_event(event)
```

**2. `start_audio_streaming()` - BIDI Audio Streaming**
```python
async def start_audio_streaming(
    self,
    session_id: str,
    connection_id: str,
    extra_information: dict = None,
):
    """
    Initialize ADK audio streaming session (Google's recommended pattern).

    Returns:
        tuple: (live_events, live_request_queue)
            - live_events: AsyncGenerator of ADK events
            - live_request_queue: Queue for sending audio/text to ADK agent
    """

    # Setup session
    example_session = await session_service.create_session(
        app_name="root_agent_app",
        user_id=f"user_{session_id}",
        session_id=session_id,
    )

    # Set session state
    example_session.state["session_id"] = session_id
    example_session.state["connection_id"] = connection_id
    example_session.state["map_context"] = extra_information.get("map_context")

    # Create runner
    runner = Runner(
        agent=self,
        app_name="root_agent_app",
        session_service=session_service,
    )

    # Configure BIDI audio streaming
    run_config = RunConfig(
        session_resumption=types.SessionResumptionConfig(transparent=True),
        save_input_blobs_as_artifacts=True,
        streaming_mode=StreamingMode.BIDI,
        max_llm_calls=1000,
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            language_code="en-US",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=config("SPEAKER_VOICE_NAME", "Kore")
                )
            ),
        ),
    )

    # Create LiveRequestQueue for bidirectional communication
    live_request_queue = LiveRequestQueue()

    # Start live streaming
    live_events = runner.run_live(
        session=example_session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

    return (live_events, live_request_queue)
```



**Capabilities:**
- Multi-agent workflow orchestration
- Salesforce CRM and GIS agent coordination
- Intelligent query analysis and routing
- Real-time streaming of agent execution
- Complex workflow pattern detection
- Cross-agent data transformation
- Session context management
- Fallback and error handling
- Thinking token transparency
- ADK sub-agent integration

### 7.3 GISAgent (Geographic Intelligence)

**Purpose**: Handle all location-based queries, geocoding, and mapping for NON-Salesforce data

**Location**: `agents/gis_agent.py`

**Parent Class**: `LlmAgent`

**Model Configuration:**
```python
def __init__(
    self, model: str | Gemini = "gemini-2.5-flash-lite", allow_override: bool = True
):
    # Determine final model based on override setting
    final_model = (
        config("SUB_AGENT_MODEL", default=None) or model
        if allow_override
        else model
    )

    super().__init__(
        model=final_model,
        name="gis_agent",
        description="Specialized agent for GIS operations, mapping, geocoding, and spatial analysis",
        instruction=self._dynamic_system_instruction,
        tools=GIS_ADK_TOOLS,
        after_tool_callback=after_tool_modifier,
        before_tool_callback=before_tool_modifier,
        after_model_callback=after_model_modifier,
        before_model_callback=before_model_modifier,
    )
```

**Dynamic System Instruction:**

The GIS Agent uses `_dynamic_system_instruction(context: ReadonlyContext)` to inject real-time map state:

```python
def _dynamic_system_instruction(self, context: ReadonlyContext) -> str:
    """Inject current map context into system instruction"""
    map_context_state = context.state.get(
        "map_context",
        "Current map view: Not available. Layers: Not available."
    )

    return f"""
    # 🌍 GIS Agent
    You are **GIS Agent**, an AI assistant for Geographic Information System (GIS) operations.

    **CURRENT MAP STATE**:
    ```json
    {map_context_state}
    ```

    ## 📌 State Access
    - **map_context** → Current map view (center, zoom, extent)
    - **layers** → Available layers (IDs, titles, schemas, visibility, metadata)
    - **pin_locations** → Current map pins (IDs, coordinates)

    ⚠️ **Never invent layer IDs or field names** — always use values from `layers`.

    ## 🛠️ Available Tools
    ... [See complete instruction in agents/gis_agent.py lines 81-238]
    """
```

**Tools (from tools/gis_adk_tools.py):**

All tools are `FunctionTool` wrappers that call `gis_executor` methods and return JSON payloads:

```python
from google.adk.tools import FunctionTool
from tools.gis_tools import gis_executor

async def toggle_layer_visibility(layer_id: str, layer_name: str, visible: bool) -> str:
    """Show or hide a map layer."""
    try:
        result = await gis_executor.toggle_layer_visibility(layer_id, layer_name, visible)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in toggle_layer_visibility: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})

toggle_layer_visibility_tool = FunctionTool(toggle_layer_visibility)
```

**Complete Tool List (15 tools):**

1. **Layer Operations:**
   - `toggle_layer_visibility(layer_id, layer_name, visible)` - Show/hide layers
   - `toggle_sublayer_visibility(layer_id, sublayer_id, visible)` - Sublayer control

2. **Filter Operations:**
   - `apply_layer_filter(layer_id, where_clause, spatial_lock)` - SQL WHERE filtering
   - `remove_layer_filter(layer_id)` - Clear filters

3. **Map Navigation:**
   - `zoom_map(zoom_action, zoom_percentage)` - Zoom in/out with percentage
   - `zoom_to_location(latitude, longitude)` - Center map at coordinates
   - `pan_map(direction, distance)` - Pan (north/south/east/west)

4. **Pin Operations:**
   - `add_map_pin(address, latitude, longitude, note)` - Add marker
   - `remove_map_pins(pin_ids)` - Remove pins (supports `["all"]`)

5. **Label Operations:**
   - `toggle_layer_labels(layer_id, enabled, label_field)` - Show/hide labels

6. **Geocoding:**
   - `geocode_address(address, max_candidates)` - Address → coordinates
   - `reverse_geocode_location(latitude, longitude)` - Coordinates → address

7. **Analytics:**
   - `get_layer_statistics(layer_id, field_names, service_url, layer_index)` - Field stats
   - `get_location_intelligence(latitude, longitude)` - POI + demographics
   - `identify_map_location(latitude, longitude)` - Reverse geocode + intelligence

**Example Interaction:**
```
User: "Show the population layer and filter for cities over 100,000"
GISAgent:
  1. Accesses map_context from state to get layer IDs
  2. Finds layer with title matching "Population"
  3. Calls toggle_layer_visibility("layer123", "Population", True)
  4. Calls apply_layer_filter("layer123", "POPULATION > 100000")
  5. Returns: "I've shown the Population layer and filtered for cities with population over 100,000."
```

**Capabilities:**
- Single and bulk address geocoding
- Reverse geocoding (coordinates to addresses)
- Map layer management and visualization
- Route planning and navigation
- Spatial analysis and filtering
- Service area calculations
- Map extent and zoom control
- Map image export
- Coordinate validation and transformation
- Real-time tool execution streaming

### 7.4 SalesforceAgent (CRM Operations)

**Purpose**: Execute Salesforce CRM operations including SOQL queries, data exports, and geocoding

**Location**: `agents/salesforce_agent.py`

**Parent Class**: `LlmAgent`

**Unique Architecture**: Uses nested agent-as-tool pattern with AddressPatternAgent

**Model Configuration:**
```python
def __init__(
    self, model: str | Gemini = "gemini-2.5-flash-lite", allow_override: bool = True
):
    final_model = (
        config("SUB_AGENT_MODEL", default=None) or model
        if allow_override
        else model
    )

    # Create specialized address pattern analysis agent
    pattern_agent = AddressPatternAgent(final_model)

    super().__init__(
        model=final_model,
        name="salesforce_agent",
        description="Specialized agent for Salesforce SOQL queries, CRM operations, and intelligent address pattern analysis",
        instruction=self._get_system_instruction(),
        tools=[*SALESFORCE_ADK_TOOLS, AgentTool(agent=pattern_agent)],  # Agent-as-tool!
        after_tool_callback=after_tool_modifier,
        before_tool_callback=before_tool_modifier,
        after_model_callback=after_model_modifier,
        before_model_callback=before_model_modifier,
    )
```

**System Instruction Highlights:**

```markdown
# 🧩 Salesforce Agent

## 🎯 Expertise
- SOQL/SOSL query construction & optimization
- Salesforce schema & field analysis
- Object relationships & dependencies
- Large dataset handling & pagination
- Aggregation & performance optimization
- Data export (CSV, GeoJSON)
- Intelligent address analysis for GIS integration

## 🛠️ Tools
- `get_all_objects` → List all objects
- `get_all_fields_for_object` → Field details of an object
- `query_soql` → Run SOQL query
- `validate_soql_query` → Check query syntax
- `query_sosl` → Run SOSL search
- `get_record_count` → Fast record count
- `query_with_pagination` → Paginated query for large datasets
- `aggregate_query` → Aggregations (COUNT, SUM, etc.)
- `export_query_to_csv` → Export SOQL to CSV
- `export_query_to_geojson` → SOQL + geocode → GeoJSON for mapping
- `geocode_existing_records` → Geocode already fetched records
- `analyze_address_patterns` → Detect & structure address fields (nested agent)

## 📜 Query Guidelines
- Default: `SELECT FIELDS(ALL)` with `LIMIT 200`
- **All SOQL should have Limit** - No SOQL without Limit Strictly
- Use **WHERE filters** for performance (e.g., `BillingCity != null`)
- Prefer pagination for >200 records (up to 50k)
- Always include **Id, Name** + address fields when mapping

## 🗺️ Mapping Workflows
- **Map Accounts** →
  `export_query_to_geojson("SELECT FIELDS(ALL) FROM Account LIMIT 200")`

- **Large dataset (5000)** →
  `... WHERE BillingCity != null`, `use_pagination=True`

- **Two-step workflow** →
  1. Fetch with `query_soql` / `query_with_pagination`
  2. Map with `geocode_existing_records(records)`
```

**Tools (from tools/salesforce_adk_tools.py):**

All tools are `FunctionTool` wrappers following the same pattern:

```python
from google.adk.tools import FunctionTool
from tools.salesforce_tools import salesforce_tools

def get_all_objects() -> str:
    """Retrieve all available Salesforce objects in the organization."""
    try:
        result = salesforce_tools.get_all_objects()
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in get_all_objects tool: {e}")
        return json.dumps({"success": False, "error": str(e)})

get_all_objects_tool = FunctionTool(get_all_objects)
```

**Complete Tool List (12 tools):**

1. **Schema Discovery:**
   - `get_all_objects()` - List all Salesforce objects
   - `get_all_fields_for_object(object_name)` - Field metadata
   - `get_object_relationships(object_name, relationship_depth)` - Object relationships

2. **Query Execution:**
   - `query_soql(soql_query)` - Execute SOQL
   - `query_sosl(sosl_query)` - Execute SOSL search
   - `validate_soql_syntax(soql_query)` - SOQL validation

3. **Data Retrieval:**
   - `get_record_count(object_name, where_clause)` - Count records
   - `query_with_pagination(soql_query, batch_size)` - Paginated queries (up to 50k records)

4. **Aggregations:**
   - `aggregate_query(object_name, aggregate_fields, group_by_fields, where_clause)` - COUNT, SUM, AVG, etc.

5. **Export:**
   - `export_query_to_csv(soql_query, include_headers)` - CSV export
   - `export_query_to_geojson(soql_query, use_pagination, max_records)` - Geocode + GeoJSON
   - `geocode_existing_records(records)` - Geocode fetched records

**Mapping Tool Deep Dive:**

`export_query_to_geojson` is the primary tool for Salesforce mapping:

```python
async def export_query_to_geojson(
    soql_query: str, use_pagination: bool = False, max_records: int = 2000
) -> str:
    """
    Execute SOQL query, geocode address fields, and return GeoJSON FeatureCollection.

    Automatically detects address fields (BillingStreet, ShippingCity, etc.),
    geocodes them using ArcGIS, and creates map-ready GeoJSON output.

    Args:
        soql_query: SOQL query with address fields (use FIELDS(ALL) for automatic detection)
        use_pagination: Set to True for large datasets (automatically fetches all records up to max_records)
        max_records: Maximum records to geocode (default 2000, max 50000 for performance)

    Returns:
        JSON string with GeoJSON FeatureCollection ready for mapping

    Examples:
        # Small dataset
        export_query_to_geojson("SELECT FIELDS(ALL) FROM Account LIMIT 200")

        # Large dataset
        export_query_to_geojson(
            "SELECT FIELDS(ALL) FROM Account WHERE BillingCity != null",
            use_pagination=True,
            max_records=5000
        )
    """
    try:
        result = await salesforce_tools.export_query_to_geojson(
            soql_query, use_pagination, max_records
        )
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in export_query_to_geojson tool: {e}")
        return json.dumps({"success": False, "error": str(e), "query": soql_query})
```

**Example Interaction:**
```
User: "Map all accounts in California"
SalesforceAgent:
  1. Calls get_all_fields_for_object("Account") to verify address fields exist
  2. Constructs query: "SELECT FIELDS(ALL) FROM Account WHERE BillingState = 'CA' LIMIT 200"
  3. Calls export_query_to_geojson(query)
  4. Tool:
     - Executes SOQL
     - Detects BillingStreet, BillingCity, BillingState, BillingPostalCode
     - Calls AddressPatternAgent to analyze address structure
     - Geocodes each record using ArcGIS
     - Creates GeoJSON FeatureCollection
  5. Returns GeoJSON to client (sent via PLOT_GEOJSON operation in UnifiedEventHandler)
```

**Capabilities:**
- SOQL query construction and execution
- SOSL cross-object searches
- Object and field schema analysis
- Data relationship mapping
- Large dataset pagination
- Aggregate query operations
- CSV data export
- Query performance optimization
- Address data extraction for mapping
- Intelligent address pattern analysis for geocoding
- Pydantic-validated address pattern responses
- Agent-as-Tool pattern analysis delegation
- Real-time tool execution streaming
- GeoJSON export for map visualization
- Automatic address field detection (billing/shipping/mailing)
- Batch geocoding with ArcGIS integration
- Multi-address type support in GeoJSON output

### 7.5 AddressPatternAgent (Nested Agent-as-Tool)

**Purpose**: Analyze Salesforce records to identify address field patterns for geocoding

**Location**: `agents/address_pattern_agent.py`

**Parent Class**: `LlmAgent`

**Usage Pattern**: Invoked by SalesforceAgent as `AgentTool` for intelligent address analysis

**Architecture:**

```python
class AddressPatternAgent(LlmAgent):
    """
    Specialized agent for analyzing Salesforce record address field patterns.

    Returns Pydantic-validated JSON responses using AddressPatternAnalysis schema.
    """

    def __init__(self, model: str | Gemini = "gemini-2.5-flash-lite"):
        super().__init__(
            model=model,
            name="address_pattern_agent",
            description="Specialized agent for analyzing Salesforce record address field patterns with validated JSON output",
            instruction=self._get_system_instruction(),
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
        )
```

**System Instruction:**

The agent receives the full `AddressPatternAnalysis` Pydantic schema and is instructed to return ONLY valid JSON:

```python
def _get_system_instruction(self) -> str:
    # Get the JSON schema and example from Pydantic model
    schema = get_address_pattern_schema()
    example = AddressPatternAnalysis.Config.json_schema_extra["example"]

    return f"""You are an expert at analyzing Salesforce record structures to identify address field patterns for geocoding operations.

MISSION: Analyze Salesforce records and return structured JSON that exactly matches the required schema for seamless GIS integration.

REQUIRED JSON SCHEMA:
{json.dumps(schema, indent=2)}

EXAMPLE RESPONSE:
{json.dumps(example, indent=2)}

ANALYSIS INSTRUCTIONS:

1. PATTERN IDENTIFICATION:
   - Identify address field groupings: billing, shipping, mailing, office, etc.
   - Detect pattern types: "compound_object" vs "individual_fields"
   - Map each address component to exact field access path

2. PATTERN TYPES:
   - compound_object: Address stored as nested object (e.g., record["BillingAddress"]["street"])
   - individual_fields: Address components in separate fields (e.g., record["BillingStreet"])

3. ACCESS PATHS:
   - For compound objects: "BillingAddress.street", "BillingAddress.city"
   - For individual fields: "BillingStreet", "BillingCity"
   - Use exact field names as they appear in the records

4. DATA COMPLETENESS:
   - Calculate percentage of records with non-null, non-empty values
   - Range: 0.0 to 1.0 (e.g., 0.85 = 85% complete)

5. GEOCODING STRATEGIES:
   - compound_preferred: Use compound object if available
   - individual_preferred: Use individual fields if available
   - compound_preferred_with_individual_fallback: Try compound first, fallback to individual
   - individual_preferred_with_compound_fallback: Try individual first, fallback to compound

Return ONLY valid JSON matching the schema - no additional text, explanations, or markdown formatting.
"""
```

**Pydantic Schema (schemas/address_patterns.py):**

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class AddressComponent(BaseModel):
    """Individual address component mapping"""
    component_type: Literal["street", "city", "state", "postal_code", "country"]
    field_path: str  # e.g., "BillingStreet" or "BillingAddress.street"
    data_completeness: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0

class AddressPattern(BaseModel):
    """Address pattern for a specific grouping (billing, shipping, etc.)"""
    pattern_name: str  # e.g., "billing", "shipping", "mailing"
    pattern_type: Literal["compound_object", "individual_fields"]
    components: List[AddressComponent]
    data_completeness: float = Field(ge=0.0, le=1.0)

class AddressPatternAnalysis(BaseModel):
    """Complete address pattern analysis result"""
    patterns: List[AddressPattern]
    recommended_strategy: Literal[
        "compound_preferred",
        "individual_preferred",
        "compound_preferred_with_individual_fallback",
        "individual_preferred_with_compound_fallback"
    ]
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    coordinate_fields: Optional[dict] = None  # e.g., {"latitude": "Latitude__c", "longitude": "Longitude__c"}

    class Config:
        json_schema_extra = {
            "example": {
                "patterns": [
                    {
                        "pattern_name": "billing",
                        "pattern_type": "individual_fields",
                        "components": [
                            {"component_type": "street", "field_path": "BillingStreet", "data_completeness": 0.85},
                            {"component_type": "city", "field_path": "BillingCity", "data_completeness": 0.90},
                            {"component_type": "state", "field_path": "BillingState", "data_completeness": 0.88},
                            {"component_type": "postal_code", "field_path": "BillingPostalCode", "data_completeness": 0.82}
                        ],
                        "data_completeness": 0.86
                    }
                ],
                "recommended_strategy": "individual_preferred",
                "confidence": 0.95,
                "coordinate_fields": {"latitude": "Latitude__c", "longitude": "Longitude__c"}
            }
        }
```

**Integration with SalesforceAgent:**

```python
# In SalesforceAgent.__init__
pattern_agent = AddressPatternAgent(final_model)

super().__init__(
    # ...
    tools=[*SALESFORCE_ADK_TOOLS, AgentTool(agent=pattern_agent)],  # Added as tool
    # ...
)

# Parsing the validated response
def parse_address_pattern_analysis(self, analysis_response: str) -> AddressPatternAnalysis:
    """Parse and validate address pattern analysis response from agent tool."""
    try:
        import json
        analysis_dict = json.loads(analysis_response)
        return validate_address_pattern_analysis(analysis_dict)
    except Exception as e:
        logger.error(f"Failed to parse address pattern analysis: {e}")
        raise ValueError(f"Invalid address pattern analysis format: {e}")
```

**Example Usage:**

```
SalesforceAgent receives: "Map these 100 account records"
↓
Calls AddressPatternAgent with records sample
↓
AddressPatternAgent analyzes:
  - Finds BillingStreet, BillingCity, BillingState, BillingPostalCode fields
  - Detects "individual_fields" pattern
  - Calculates 86% data completeness
  - Returns JSON matching AddressPatternAnalysis schema
↓
SalesforceAgent uses analysis to:
  - Construct field access paths: record["BillingStreet"], etc.
  - Geocode using ArcGIS with assembled address strings
  - Create GeoJSON with properly mapped coordinates
```

**Capabilities:**
- Salesforce record structure analysis
- Address field pattern identification
- Compound vs individual field detection
- Data completeness assessment
- Geocoding strategy recommendations
- Pydantic-validated JSON responses
- Python access path generation
- Multi-address type support (billing, shipping, mailing)
- Field naming convention analysis
- Real-time pattern confidence scoring

### 7.6 Streaming Callbacks

**Purpose**: Provide real-time visibility into agent execution via WebSocket

**Location**: `agents/streaming_callbacks.py`

**Integration**: All agents register these callbacks during initialization

**Callback Functions:**

**1. `before_tool_modifier` - Pre-Tool Execution**
```python
async def before_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """
    Callback executed BEFORE tool execution.

    Sends "Retrieving information..." message to client via WebSocket.
    """
    connection_id = tool_context.state.get("connection_id")

    await manager.send_thinking_message(
        connection_id,
        {"response": f"Retrieving the requested information…"},
    )
    return None
```

**2. `after_tool_modifier` - Post-Tool Execution**
```python
def after_tool_modifier(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict
) -> Optional[Dict]:
    """
    Callback executed AFTER tool execution completes.

    Tool execution data is extracted directly from events in UnifiedEventHandler,
    so this callback only performs logging.
    """
    agent_name = tool_context.agent_name
    tool_name = tool.name
    logger.info(f"Tool executed: {tool_name} by {agent_name}")
    return None
```

**3. `before_model_modifier` - Pre-LLM Call**
```python
async def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    last_user_message = ""  # Could extract from llm_request if needed
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")
    return None
```

**4. `after_model_modifier` - Post-LLM Call**
```python
async def after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Logs model completion."""
    agent_name = callback_context.agent_name
    print(f"[Callback] After model call for agent: {agent_name}")
    return None
```

**Registration Pattern:**

Every agent registers all four callbacks:

```python
super().__init__(
    model=model,
    name="agent_name",
    # ...
    after_tool_callback=after_tool_modifier,
    before_tool_callback=before_tool_modifier,
    after_model_callback=after_model_modifier,
    before_model_callback=before_model_modifier,
)
```

**Execution Flow:**

```
User sends query
↓
before_model_modifier → Logs LLM request
↓
LLM processes query
↓
after_model_modifier → Logs LLM completion
↓
before_tool_modifier → Sends "Retrieving information..." to client
↓
Tool executes (e.g., geocode_address)
↓
after_tool_modifier → Logs tool completion
↓
Result returned to client
```

**WebSocket Message Example:**

When `before_tool_modifier` executes:
```json
{
  "type": "CHAT/THINKING",
  "payload": {
    "connection_id": "abc123",
    "response": "Retrieving the requested information…"
  }
}
```

### 7.7 Tool Implementation Pattern

All tools follow the same Google ADK `FunctionTool` wrapper pattern:

**Step 1: Define Async Function**
```python
async def tool_name(param1: str, param2: int) -> str:
    """
    Tool description shown to LLM.

    Args:
        param1: Parameter description
        param2: Parameter description

    Returns:
        JSON string with result or error
    """
    try:
        # Call executor or service
        result = await executor.method(param1, param2)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in tool_name: {e}")
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})
```

**Step 2: Wrap in FunctionTool**
```python
from google.adk.tools import FunctionTool

tool_name_tool = FunctionTool(tool_name)
```

**Step 3: Add to Tool List**
```python
AGENT_TOOLS = [
    tool_name_tool,
    other_tool,
    # ...
]
```

**Step 4: Register with Agent**
```python
super().__init__(
    model=model,
    name="agent_name",
    tools=AGENT_TOOLS,
    # ...
)
```

**Return Format Convention:**

All tools return JSON strings with consistent structure:

**Success:**
```json
{
  "type": "OPERATION_TYPE",
  "payload": {
      "operations": [
        {
          "type": "TOGGLE_LAYER_VISIBILITY",
          "payload": {
            "layer_id": "layer123",
            "visible": true
          }
        }
      ]
    }
}
```

**Error:**
```json
{
  "type": "ERROR",
  "payload": {
    "error": "Error message description"
  }
}
```

**GIS Operations Example:**
```json
{
  "operations": [
    {
      "type": "TOGGLE_LAYER_VISIBILITY",
      "payload": {
        "layer_id": "layer123",
        "visible": true
      }
    }
  ]
}
```

### 7.8 Adding Custom Agents

**Step 1: Create Agent File**

```python
# agents/custom_agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.models import Gemini
import json

# Define tools
async def custom_tool(param: str) -> str:
    """Custom tool description"""
    try:
        result = await perform_operation(param)
        return json.dumps(result, indent=0)
    except Exception as e:
        return json.dumps({"type": "ERROR", "payload": {"error": str(e)}})

custom_tool_tool = FunctionTool(custom_tool)

# Define agent
class CustomAgent(LlmAgent):
    """Custom specialized agent"""

    def __init__(self, model: str | Gemini = "gemini-2.5-flash-lite"):
        super().__init__(
            model=model,
            name="custom_agent",
            description="Specialized agent for custom operations",
            instruction=self._get_system_instruction(),
            tools=[custom_tool_tool],
            after_tool_callback=after_tool_modifier,
            before_tool_callback=before_tool_modifier,
            after_model_callback=after_model_modifier,
            before_model_callback=before_model_modifier,
        )

    def _get_system_instruction(self) -> str:
        return """
        # Custom Agent
        You are a specialized agent for custom operations.

        ## Tools
        - custom_tool: Performs custom operation

        ## Guidelines
        - Use tool when user requests custom operation
        - Return structured results
        """
```

**Step 2: Register as Sub-Agent in RootAgent**

```python
# agents/root_agent.py
from agents.custom_agent import CustomAgent

class RootAgent(LlmAgent):
    def __init__(self, model: str | Gemini = "gemini-2.5-flash-lite", allow_sub_agent_override: bool = True):
        super().__init__(
            model=model,
            name="root_agent",
            # ...
            sub_agents=[
                SalesforceAgent(model, allow_override=allow_sub_agent_override),
                GISAgent(model, allow_override=allow_sub_agent_override),
                CustomAgent(model),  # Add new agent
            ],
            # ...
        )
```

**Step 3: Update RootAgent System Instruction**

```python
def _get_system_instruction(self) -> str:
    return """
    # Master Orchestrator Agent

    ## Available Sub-Agents

    ### 1. Salesforce Agent
    Handles ALL Salesforce data operations...

    ### 2. GIS Agent
    Handles map operations for NON-Salesforce data...

    ### 3. Custom Agent  # Add documentation
    Handles custom operations:
    - Custom operation type 1
    - Custom operation type 2

    ## Routing Rules

    ### → Custom Agent
    - Query pattern X
    - Keywords: "custom", "special operation"
    """
```

The RootAgent will automatically:
1. Analyze the query intent
2. Recognize custom operation keywords
3. Delegate to CustomAgent via ADK's sub_agents mechanism
4. Return CustomAgent's response to the client

---


---

## 8. Audio Streaming Architecture

### 8.1 Audio Pipeline Overview

```
                        CLIENT SIDE
┌───────────────────────────────────────────────────────────┐
│ Microphone Input → VAD → PCM Encoding (16kHz) → Base64   │
│                                                            │
│ Base64 → Decoding → PCM (24kHz) → Audio Playback         │
└───────────────────────────────────────────────────────────┘
                          ↕
                   WebSocket (BIDI)
                          ↕
                        SERVER SIDE
┌───────────────────────────────────────────────────────────┐
│ WebSocket Handler (main.py:272-432)                       │
│ - Receive audio chunks                                    │
│ - Validate message schema                                 │
│ - Push to LiveRequestQueue                                │
│                   ↓                                        │
│ LiveRequestQueue (Google ADK)                             │
│ - Buffer audio chunks                                     │
│ - Stream to agent                                         │
│                   ↓                                        │
│ Agent Runtime (RootAgent → Specialized Agents)            │
│ - Transcribe audio (Gemini Live)                          │
│ - Process query                                           │
│ - Execute tools                                           │
│ - Generate TTS response                                   │
│                   ↓                                        │
│ UnifiedEventHandler (handlers/event_handler.py)           │
│ - Listen to agent live_events stream                      │
│ - Process audio, text, turn status events                 │
│ - Non-blocking audio transmission                         │
│ - Memory optimization (strip audio data)                  │
│ - Forward events to client                                │
└───────────────────────────────────────────────────────────┘
                          ↕
                   WebSocket (BIDI)
                          ↕
                        CLIENT SIDE
┌───────────────────────────────────────────────────────────┐
│ Receive audio_chunk events → Decode base64 → Play audio   │
└───────────────────────────────────────────────────────────┘
```

### 8.2 Audio Streaming Implementation

**Starting Audio Session (root_agent.py:294-406):**

The RootAgent's `start_audio_streaming()` method initializes a BIDI audio session using Google ADK's `run_live()` pattern:

```python
async def start_audio_streaming(
    self,
    session_id: str,
    connection_id: str,
    extra_information: dict = None,
):
    """
    Initialize ADK audio streaming session (Google's recommended pattern).

    Returns:
        tuple: (live_events, live_request_queue)
            - live_events: AsyncGenerator of ADK events (audio, text, turn signals)
            - live_request_queue: Queue for sending audio/text to ADK agent
    """
    # Setup session with DatabaseSessionService
    example_session = await session_service.create_session(
        app_name="root_agent_app",
        user_id=f"user_{session_id}",
        session_id=session_id,
    )

    # Set session state
    example_session.state["session_id"] = session_id
    example_session.state["connection_id"] = connection_id
    example_session.state["map_context"] = extra_information.get("map_context")

    # Create runner
    runner = Runner(
        agent=self,
        app_name="root_agent_app",
        session_service=session_service,
    )

    # Configure for BIDI audio streaming with session resumption
    run_config = RunConfig(
        session_resumption=types.SessionResumptionConfig(transparent=True),
        save_input_blobs_as_artifacts=True,
        streaming_mode=StreamingMode.BIDI,  # BIDI mode for audio
        max_llm_calls=1000,
        output_audio_transcription=types.AudioTranscriptionConfig(),  # Enable transcription
        speech_config=types.SpeechConfig(
            language_code="en-US",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=config("SPEAKER_VOICE_NAME", "Kore")
                )
            ),
        ),
    )

    # Create LiveRequestQueue for bidirectional communication
    live_request_queue = LiveRequestQueue()

    # Start live streaming with audio support
    live_events = runner.run_live(
        session=example_session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

    return (live_events, live_request_queue)
```

**Key Configuration Parameters:**

1. **session_resumption**: `transparent=True` - Enables conversation continuity across turns
2. **save_input_blobs_as_artifacts**: `True` - For future playback/analysis
3. **streaming_mode**: `StreamingMode.BIDI` - Bidirectional audio streaming
4. **max_llm_calls**: `1000` - Maximum LLM calls per session
5. **output_audio_transcription**: Enables text transcription of audio responses
6. **speech_config**: Voice selection (default: "Kore")

**WebSocket Handler (main.py):**

```python
@app.websocket("/ws/audio/{session_id}")
async def audio_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    user_info: Optional[dict] = None
):
    """WebSocket endpoint for bidirectional audio streaming"""

    await websocket.accept()
    connection_id = str(uuid.uuid4())

    # Initialize audio streaming session
    live_events, live_request_queue = await root_agent.start_audio_streaming(
        session_id=session_id,
        connection_id=connection_id,
        extra_information={"map_context": map_context}
    )

    # Create event handler
    event_handler = UnifiedEventHandler(
        connection_id=connection_id,
        session_id=session_id,
        manager=manager,
        mode="BIDI",  # Audio mode
    )

    # Process events in background
    async def process_events():
        async for event in live_events:
            await event_handler.handle_event(event)

    event_task = asyncio.create_task(process_events())

    # Handle incoming audio from client
    try:
        async for message in websocket.iter_json():
            if message["message_type"] == "audio":
                # Decode base64 audio
                audio_data = base64.b64decode(message["data"])

                # Push to LiveRequestQueue
                await live_request_queue.put_audio(audio_data)

            elif message["message_type"] == "text":
                # Text input (optional in BIDI mode)
                await live_request_queue.put_text(message["text"])

    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket disconnected: {connection_id}")
        event_task.cancel()
```

### 8.3 Audio Event Handling

**UnifiedEventHandler Audio Processing:**

The `UnifiedEventHandler` processes audio events with non-blocking transmission and memory optimization:

**Audio Content Handling (handlers/event_handler.py:332-365):**

```python
async def _handle_audio_content(self, event: Any, part: Any) -> None:
    """
    Handle audio streaming in BIDI mode.

    - Sends audio chunks to client (non-blocking)
    - Strips audio data from event to save memory
    - Detects final chunk based on turn_complete or finish_reason

    Args:
        event: Event containing audio
        part: Part with inline_data (audio)
    """
    if not hasattr(part.inline_data, "data"):
        return

    audio_data = part.inline_data.data

    # Determine if this is the final audio chunk
    is_final = (
        getattr(event, "turn_complete", False)
        or getattr(event, "finish_reason", None) is not None
    )

    # Send audio to client (non-blocking)
    asyncio.create_task(
        self.manager.send_audio_to_client(
            self.connection_id,
            audio_data,
            is_final=is_final,
        )
    )

    # Strip audio data to save memory
    part.inline_data.data = b""
```

**Key Design Decisions:**

1. **Non-Blocking Transmission**: Uses `asyncio.create_task()` to avoid blocking event loop
2. **Memory Optimization**: Strips `audio_data` after sending to prevent memory accumulation
3. **Final Chunk Detection**: Uses `turn_complete` or `finish_reason` flags
4. **No Deduplication**: Audio is sent directly without invocation-based filtering

**Audio Transcription Handling (handlers/event_handler.py:367-392):**

Audio transcriptions provide text representation of audio responses for accessibility:

```python
async def _handle_audio_transcription(self, event: Any) -> None:
    """
    Handle audio transcription events (text representation of audio response).

    Provides accessibility support by sending text transcription of
    audio responses to the client for display.

    Args:
        event: Event with output_transcription
    """
    transcription_text = event.output_transcription.text

    logger.debug(f"[AUDIO_TRANSCRIPTION] : {transcription_text}")

    await self.manager.send_message(
        self.connection_id,
        {
            "type": "CHAT/AUDIO_TRANSCRIPTION",
            "payload": {
                "text": transcription_text,
                "timestamp": time.time(),
            },
        },
    )
```

**Client Message Format:**
```json
{
  "type": "CHAT/AUDIO_TRANSCRIPTION",
  "payload": {
    "text": "The Golden Gate Bridge is located at coordinates 37.8199, -122.4783.",
    "timestamp": 1704067200.123
  }
}
```

### 8.4 Turn Management

**Turn Status Events (handlers/event_handler.py:398-427):**

Turn management in LM Multi Agent uses simple boolean flags, not complex state machines:

```python
async def _handle_turn_status(self, event: Any) -> None:
    """
    Handle turn status events (turn_complete, interrupted).

    Following official ADK pattern: both flags in one message.

    Args:
        event: Event with turn status
    """
    turn_complete = getattr(event, "turn_complete", False)
    interrupted = getattr(event, "interrupted", False)

    if turn_complete or interrupted:
        await self.manager.send_message(
            self.connection_id,
            {
                "type": WEBSOCKET_TYPE_TURN_STATUS,
                "payload": {
                    "turn_complete": turn_complete,
                    "interrupted": interrupted,
                    "final_text": self.response_text if turn_complete else "",
                    "timestamp": time.time(),
                },
            },
        )

        logger.info(
            f"Sent turn status: turn_complete={turn_complete}, "
            f"interrupted={interrupted}"
        )
```

**Turn Lifecycle:**

```
1. User Starts Speaking
   ↓
2. Client Sends Audio Chunks (is_final: false)
   ↓
3. User Stops Speaking
   ↓
4. Client Sends Final Chunk (is_final: true)
   ↓
5. Agent Processes Complete Utterance
   ↓
6. Agent Generates Audio Response
   ↓
7. Agent Emits turn_complete Event
   ↓
8. System Ready for Next Turn
```

**Turn Status Flags:**

- **turn_complete**: `true` when agent has finished responding
- **interrupted**: `true` when user interrupts agent mid-response
- **final_text**: Complete text response (when `turn_complete=true`)

**Example Turn Status Message:**
```json
{
  "type": "CHAT/TURN_STATUS",
  "payload": {
    "turn_complete": true,
    "interrupted": false,
    "final_text": "I've found 5 accounts in San Francisco matching your criteria.",
    "timestamp": 1704067200.456
  }
}
```

**Interrupt Handling:**

When a user interrupts the agent:

```json
{
  "type": "CHAT/TURN_STATUS",
  "payload": {
    "turn_complete": false,
    "interrupted": true,
    "final_text": "",
    "timestamp": 1704067200.789
  }
}
```

The client can use this to:
- Stop audio playback
- Clear partial responses
- Reset UI state
- Resume listening for new input

### 8.5 Audio Encoding Specifications

**Client → Server (Input Audio):**
- **Format**: PCM (Pulse Code Modulation)
- **Sample Rate**: 16,000 Hz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit signed integer
- **Encoding**: Base64
- **Chunk Size**: 1024 samples (~64ms per chunk)

**Server → Client (Output Audio):**
- **Format**: PCM
- **Sample Rate**: 24,000 Hz
- **Channels**: Mono
- **Bit Depth**: 16-bit signed integer
- **Encoding**: Base64

**JavaScript Audio Capture Example:**
```javascript
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        const audioContext = new AudioContext({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(1024, 1, 1);

        processor.onaudioprocess = (event) => {
            const inputData = event.inputBuffer.getChannelData(0);

            // Convert Float32Array to Int16Array
            const pcmData = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
            }

            // Encode to base64
            const base64Audio = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)));

            // Send via WebSocket
            websocket.send(JSON.stringify({
                message_type: "audio",
                data: base64Audio,
                sample_rate: 16000,
                session_id: sessionId
            }));
        };

        source.connect(processor);
        processor.connect(audioContext.destination);
    });
```

**Python Audio Playback Example:**
```python
import base64
import wave
import pyaudio

async def play_audio_response(audio_chunks: List[str]):
    """Play received audio chunks"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )

    for chunk_base64 in audio_chunks:
        # Decode base64
        chunk_bytes = base64.b64decode(chunk_base64)

        # Play chunk
        stream.write(chunk_bytes)

    stream.stop_stream()
    stream.close()
    p.terminate()
```


### 8.7 BIDI vs SSE Mode Differences

**BIDI Mode (Audio Streaming):**

```python
# Configuration
run_config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    speech_config=types.SpeechConfig(...),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)

# Event Handler
event_handler = UnifiedEventHandler(
    connection_id=connection_id,
    session_id=session_id,
    manager=manager,
    mode="BIDI",
)

# Events Processed
- audio content (inline_data)
- audio transcription (output_transcription)
- turn status (turn_complete, interrupted)
- text content (for fallback/accessibility)
- thinking events
```

**SSE Mode (Text-Only Streaming):**

```python
# Configuration
run_config = RunConfig(
    streaming_mode=StreamingMode.SSE,
)

# Event Handler
event_handler = UnifiedEventHandler(
    connection_id=connection_id,
    session_id=session_id,
    manager=manager,
    mode="SSE",
)

# Events Processed
- text content (3-state detection)
- tool execution results
- completion metadata (usage_metadata)
- finish_reason signals
```

**Key Differences:**

| Feature | BIDI Mode                            | SSE Mode |
|---------|--------------------------------------|----------|
| **Audio** | Yes (PCM chunks)                     | No |
| **TTS** | Server-generated                     | No |
| **STT** | Server-processed                     | No |
| **Turn Management** | Yes (`turn_complete`, `interrupted`) | No |
| **Text Streaming** | Simple (partial flag)                | 3-state detection |
| **Tool Results** | Full operation data                  | Full operation data |
| **Completion Signal** | `turn_complete` event                | `usage_metadata` in final event |
| **User Echo** | Work In Progress                     | Not applicable |

**Mode Selection Logic:**

```python
# Audio streaming request
if message.get("type") == "audio_streaming_start":
    live_events, live_request_queue = await root_agent.start_audio_streaming(
        session_id=session_id,
        connection_id=connection_id
    )
    # Uses BIDI mode

# Text query request
elif message.get("type") == "query":
    await root_agent.process_query(
        query=message["query"],
        session_id=session_id,
        connection_id=connection_id
    )
    # Uses SSE mode
```

### 8.8 WebSocket Message Types

**Client → Server:**

1. **Audio Chunk:**
```json
{
  "message_type": "audio",
  "data": "base64_encoded_pcm_data",
  "sample_rate": 16000,
  "session_id": "abc123",
  "is_final": false
}
```

2. **Text Input (Optional in BIDI):**
```json
{
  "message_type": "text",
  "text": "What's the weather?",
  "session_id": "abc123"
}
```

**Server → Client:**

1. **Audio Chunk:**
```json
{
  "type": "CHAT/AUDIO_CHUNK",
  "payload": {
    "connection_id": "abc123",
    "data": "base64_encoded_pcm_data",
    "is_final": false,
    "timestamp": 1704067200.123
  }
}
```

2. **Audio Transcription:**
```json
{
  "type": "CHAT/AUDIO_TRANSCRIPTION",
  "payload": {
    "text": "I found 5 accounts in California.",
    "timestamp": 1704067200.456
  }
}
```

3. **Turn Status:**
```json
{
  "type": "CHAT/TURN_STATUS",
  "payload": {
    "turn_complete": true,
    "interrupted": false,
    "final_text": "Complete response text",
    "timestamp": 1704067200.789
  }
}
```

4. **Text Stream (BIDI):**
```json
{
  "type": "CHAT/STREAM",
  "payload": {
    "connection_id": "abc123",
    "session_id": "def456",
    "text": "Partial text chunk",
    "is_partial": true,
    "timestamp": 1704067200.123
  }
}
```

5. **Thinking Message:**
```json
{
  "type": "CHAT/THINKING",
  "payload": {
    "response": "Retrieving the requested information…"
  }
}
```

### 8.9 Error Handling

**Audio Stream Errors:**

```python
try:
    async for event in live_events:
        await event_handler.handle_event(event)
except Exception as e:
    logger.error(f"Audio streaming error: {e}")
    await manager.send_message(
        connection_id,
        {
            "type": "CHAT/ERROR",
            "payload": {
                "error": "Audio streaming interrupted",
                "details": str(e),
                "recoverable": True
            }
        }
    )
```

**WebSocket Disconnection:**

```python
try:
    async for message in websocket.iter_json():
        # Process message
        pass
except WebSocketDisconnect:
    logger.info(f"WebSocket disconnected: {connection_id}")
    # Cleanup tasks
    event_task.cancel()
    await manager.disconnect(connection_id)
```

**Audio Decoding Errors:**

```python
try:
    audio_data = base64.b64decode(message["data"])
    await live_request_queue.put_audio(audio_data)
except Exception as e:
    logger.error(f"Audio decode error: {e}")
    await manager.send_message(
        connection_id,
        {
            "type": "CHAT/ERROR",
            "payload": {
                "error": "Invalid audio format",
                "expected": "base64-encoded PCM 16kHz mono",
                "recoverable": True
            }
        }
    )
```

---


---

## 9. Event Handling System

### 9.1 Overview

The `UnifiedEventHandler` class in `handlers/event_handler.py` provides consolidated event processing for both **SSE** (text) and **BIDI** (audio) streaming modes. It eliminates duplicate code and provides consistent event handling across the application.

**Key Features:**
- **Mode-aware processing**: Handles SSE and BIDI modes with mode-specific logic
- **Tool execution routing**: Routes tool results from specialized agents (GIS, Salesforce)
- **Three-state SSE detection**: Streaming chunk → Last chunk → Final consolidated
- **Audio streaming**: Non-blocking audio transmission with memory optimization
- **Turn management**: Proper turn_complete and interrupt handling

### 9.2 Event Handler Architecture

```python
class UnifiedEventHandler:
    """
    Unified event handler for both SSE and BIDI streaming modes.

    Modes:
    - SSE: Text-only streaming with tool execution and 3-state completion detection
    - BIDI: Audio + text streaming with turn status and interrupts
    """

    def __init__(self, connection_id: str, session_id: str, manager: Any, mode: str = "SSE"):
        self.connection_id = connection_id
        self.session_id = session_id
        self.manager = manager
        self.mode = mode.upper()  # "SSE" or "BIDI"
        self.response_text = ""
```

### 9.3 Event Processing Flow

**Main Event Dispatcher:**
```python
async def handle_event(self, event: Any) -> None:
    """Main event dispatcher - routes events to appropriate handlers."""

    # BIDI mode: Skip user echoes AND sub-agent responses
    if self.mode == "BIDI":
        is_model_response = hasattr(event, "author") and event.author != "user"
        if not is_model_response:
            return

    # Process content events (text, audio, tools)
    if hasattr(event, "content") and event.content:
        await self._process_content(event)

    # Process non-content events (thinking, turn status)
    await self._process_non_content_events(event)
```

### 9.4 Content Processing

**SSE Mode - Text and Tool Processing:**
```python
async def _process_content(self, event: Any) -> None:
    """Process content events (text, audio, tool execution)."""

    if self.mode == "SSE":
        for part in event.content.parts:
            # Tool execution results
            if hasattr(part, "function_response") and part.function_response:
                await self._handle_tool_execution(event, part)

            # Text streaming
            if hasattr(part, "text") and part.text:
                await self._handle_text_content(event, part)
```

**BIDI Mode - Audio, Text, and Tool Processing:**
```python
    elif self.mode == "BIDI":
        part = event.content.parts[0]

        # Text streaming
        if hasattr(part, "text") and part.text:
            await self._handle_text_content(event, part)

        # Tool execution
        if hasattr(part, "function_response") and part.function_response:
            await self._handle_tool_execution(event, part)

        # Audio streaming
        if hasattr(part, "inline_data") and part.inline_data:
            await self._handle_audio_content(event, part)
```

### 9.5 SSE Mode: Three-State Text Detection

The handler detects three distinct states during SSE streaming:

**State Detection Logic:**
```python
async def _handle_sse_text(self, event: Any, text: str) -> None:
    """
    Handle SSE text streaming with 3-state detection.

    States:
    1. Streaming chunk: partial=True, finish_reason=None
    2. Last streaming chunk: finish_reason="STOP"
    3. Final consolidated: partial=None, usage_metadata populated
    """

    # State 1: Streaming chunk
    is_streaming_chunk = event.partial == True and event.finish_reason is None

    # State 2: Last streaming chunk
    is_last_streaming_chunk = event.finish_reason == "STOP"

    # State 3: Final consolidated
    is_final_consolidated = (
        event.partial is None
        and event.usage_metadata
        and event.usage_metadata.total_token_count
    )

    if is_streaming_chunk:
        # Send partial text
        await self.manager.send_streaming_message(
            self.connection_id,
            {"stream_response": text, "stream_stop": False, "stream_id": event.id}
        )

    elif is_last_streaming_chunk:
        # Send last chunk with stop flag
        await self.manager.send_streaming_message(
            self.connection_id,
            {"stream_response": text, "stream_stop": True, "stream_id": event.id}
        )

    elif is_final_consolidated:
        # Send completion with token usage
        await self._send_sse_completion(text, event.usage_metadata)
```

### 9.6 Tool Execution Routing

**Agent-Specific Tool Handling:**
```python
async def _handle_tool_execution(self, event: Any, part: Any) -> None:
    """
    Handle tool execution results from sub-agents.

    Routes to specialized handlers based on agent type:
    - gis_agent: Send operation data
    - salesforce_agent: Send PLOT_GEOJSON for mapping tools
    """
    agent_name = event.author  # "gis_agent", "salesforce_agent"
    tool_name = part.function_response.name
    tool_result = part.function_response.response.get("result", "")

    # Route to specialized handlers
    if agent_name == "gis_agent" and tool_result:
        await self._handle_gis_tool(tool_result, tool_name)

    if agent_name == "salesforce_agent":
        await self._handle_salesforce_tool(tool_name, tool_result)
```

**GIS Tool Handler:**
```python
async def _handle_gis_tool(self, tool_result: Any, tool_name: str) -> None:
    """Send GIS operation data to client for map rendering."""
    operation_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
    await self.manager.send_operations_data(self.connection_id, operation_data)
```

**Salesforce Tool Handler:**
```python
async def _handle_salesforce_tool(self, tool_name: str, tool_result: Any) -> None:
    """
    Send PLOT_GEOJSON operation for mapping tools:
    - export_query_to_geojson_tool
    - geocode_existing_records_tool
    """
    if tool_name in ["export_query_to_geojson_tool", "geocode_existing_records_tool"]:
        tool_response_json = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
        file_info = tool_response_json.get("file_information", {})

        if file_info:
            operation_data = {
                "operations": [{
                    "type": "PLOT_GEOJSON",
                    "payload": file_info
                }]
            }
            await self.manager.send_operations_data(self.connection_id, operation_data)
```

### 9.7 Audio Streaming (BIDI Mode)

**Non-Blocking Audio Transmission:**
```python
async def _handle_audio_content(self, event: Any, part: Any) -> None:
    """
    Handle audio streaming in BIDI mode.

    - Sends audio chunks to client (non-blocking)
    - Strips audio data from event to save memory
    - Detects final chunk based on turn_complete or finish_reason
    """
    if not hasattr(part.inline_data, "data"):
        return

    audio_data = part.inline_data.data

    # Determine if this is the final audio chunk
    is_final = (
        getattr(event, "turn_complete", False)
        or getattr(event, "finish_reason", None) is not None
    )

    # Send audio to client (non-blocking)
    asyncio.create_task(
        self.manager.send_audio_to_client(
            self.connection_id,
            audio_data,
            is_final=is_final
        )
    )

    # Strip audio data to save memory
    part.inline_data.data = b""
```

**Audio Transcription Support:**
```python
async def _handle_audio_transcription(self, event: Any) -> None:
    """
    Provide text transcription of audio responses for accessibility.
    Sends text representation alongside audio chunks.
    """
    transcription_text = event.output_transcription.text

    await self.manager.send_message(
        self.connection_id,
        {
            "type": "CHAT/AUDIO_TRANSCRIPTION",
            "payload": {
                "text": transcription_text,
                "timestamp": time.time()
            }
        }
    )
```

### 9.8 Turn Management (BIDI Mode)

**Turn Status Handling:**
```python
async def _handle_turn_status(self, event: Any) -> None:
    """
    Handle turn status events (turn_complete, interrupted).
    Following official ADK pattern: both flags in one message.
    """
    turn_complete = getattr(event, "turn_complete", False)
    interrupted = getattr(event, "interrupted", False)

    if turn_complete or interrupted:
        await self.manager.send_message(
            self.connection_id,
            {
                "type": "CHAT/TURN_STATUS",
                "payload": {
                    "turn_complete": turn_complete,
                    "interrupted": interrupted,
                    "final_text": self.response_text if turn_complete else "",
                    "timestamp": time.time()
                }
            }
        )
```

### 9.9 Message Types

**SSE Mode Messages:**

| Message Type | Description | Payload |
|--------------|-------------|---------|
| `CHAT/STREAM` | Streaming text chunk | `stream_response`, `stream_stop`, `stream_id` |
| `CHAT/STREAM_INFO` | Stream completion with token usage | `final_text`, `token_usage`, `finish_reason` |
| `CHAT/OPERATIONS` | Tool execution results (GIS, Salesforce) | `operations` array |

**BIDI Mode Messages:**

| Message Type | Description | Payload |
|--------------|-------------|---------|
| `CHAT/STREAM` | Text chunk | `text`, `is_partial`, `timestamp` |
| `CHAT/AUDIO` | Audio chunk | `audio_data` (base64), `is_final` |
| `CHAT/AUDIO_TRANSCRIPTION` | Audio transcription | `text`, `timestamp` |
| `CHAT/TURN_STATUS` | Turn completion/interrupt | `turn_complete`, `interrupted`, `final_text` |
| `CHAT/THINKING` | Agent thinking status | `message` |

### 9.10 Usage Example

**Initialization:**
```python
# SSE mode
sse_handler = UnifiedEventHandler(
    connection_id="conn_123",
    session_id="sess_456",
    manager=websocket_manager,
    mode="SSE"
)

# BIDI mode
bidi_handler = UnifiedEventHandler(
    connection_id="conn_789",
    session_id="sess_012",
    manager=websocket_manager,
    mode="BIDI"
)
```

**Event Processing:**
```python
# Process events from ADK stream
async for event in agent_stream:
    await handler.handle_event(event)

# Send completion message (BIDI mode)
await handler.send_completion_message()
```

---------|-------------|----------------|
| `text_chunk` | Incremental text response (SSE mode) | `content`, `is_final` |
| `audio_chunk` | Audio data (BIDI mode) | `data` (base64), `sample_rate` |
| `tool_call` | Tool execution started | `tool_name`, `tool_input` |
| `tool_result` | Tool execution completed | `tool_name`, `tool_output` |
| `turn_complete` | Agent finished responding | - |
| `interrupted` | User interrupted agent | - |
| `error` | Error occurred | `error_code`, `error_message` |
| `connection_established` | WebSocket connected | `connection_id`, `session_id` |
| `connection_closed` | WebSocket disconnected | `reason` |

---
