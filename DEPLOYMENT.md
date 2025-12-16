# Deployment Guide - Market Insights AI Backend

## Railway Deployment

### Prerequisites
- Railway account (https://railway.app)
- GitHub repository connected to Railway

### Step 1: Connect Repository

1. Go to Railway Dashboard → New Project
2. Select "Deploy from GitHub repo"
3. Choose: `patelyakshit/lm-marketinsights-be`
4. Select branch: `main` (or your preferred branch)

### Step 2: Configure Environment Variables

In Railway Dashboard → Your Project → Variables, add:

```bash
# ============================================
# REQUIRED - Google AI / Vertex AI
# ============================================
# Option A: Use Google AI Studio (simpler)
GOOGLE_API_KEY=your_google_ai_studio_api_key

# Option B: Use Vertex AI (for production)
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account","project_id":"..."}
# Note: Paste the entire JSON content of your service account key

# ============================================
# REQUIRED - ArcGIS
# ============================================
ARCGIS_API_KEY=your_arcgis_api_key
ARCGIS_GEOLOCATION_API_KEY=your_arcgis_geolocation_key

# ============================================
# OPTIONAL - Salesforce (if using Salesforce agent)
# ============================================
SALESFORCE_USERNAME=
SALESFORCE_PASSWORD=
SALESFORCE_SECURITY_TOKEN=
SALESFORCE_DOMAIN=
SALESFORCE_INSTANCE_URL=

# ============================================
# OPTIONAL - Azure Storage (for file exports)
# ============================================
AZURE_STORAGE_CONNECTION_STRING=
AZURE_STORAGE_CONTAINER_NAME=salesforce-exports

# ============================================
# OPTIONAL - RabbitMQ (for message queue)
# ============================================
# Railway can provision RabbitMQ - use their plugin
# Or set RABBITMQ_URL manually
RABBITMQ_URL=amqp://user:password@host:5672/

# ============================================
# OPTIONAL - Qdrant (for vector search)
# ============================================
QDRANT_DB_URL=your_qdrant_url
QDRANT_DB_PORT=6333
QDRANT_COLLECTION_NAME=your_collection

# ============================================
# OPTIONAL - OpenAI (for embeddings)
# ============================================
OPENAI_API_KEY=your_openai_key

# ============================================
# APPLICATION SETTINGS
# ============================================
LOG_LEVEL=INFO
ENABLE_AUDIO_STREAMING=true
```

### Step 3: Handle Google Credentials

**Option A: Google AI Studio (Recommended for testing)**
- Get API key from https://makersuite.google.com/app/apikey
- Set `GOOGLE_API_KEY` in Railway

**Option B: Vertex AI (Production)**
- Create service account in GCP Console
- Download JSON key
- In Railway, create variable `GOOGLE_APPLICATION_CREDENTIALS_JSON`
- Paste the entire JSON content as the value
- Update code to handle JSON from env var (see below)

### Step 4: Deploy

Railway auto-deploys on push to connected branch.

### Health Check

Once deployed, verify at: `https://your-app.up.railway.app/health`

Expected response:
```json
{"status": "healthy", "service": "lm-multi-agent-api"}
```

---

## Frontend Connection

Update your frontend `.env` with the Railway URL:

```bash
VITE_SOCKET_BASE=wss://your-app.up.railway.app/ws
VITE_SOCKET_VOICE=wss://your-app.up.railway.app/ws/audio
VITE_API_BASE_URL=https://your-app.up.railway.app
```

---

## Troubleshooting

### Build Fails
- Check Dockerfile syntax
- Ensure all dependencies in requirements.txt

### Health Check Fails
- Check logs in Railway dashboard
- Verify environment variables are set

### WebSocket Connection Issues
- Ensure Railway URL uses `wss://` not `ws://`
- Check CORS settings in main.py

### Google AI Errors
- Verify API key is valid
- Check quotas in Google Cloud Console
