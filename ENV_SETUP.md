# Aegis RAG: Environment Setup Guide

This guide explains where to obtain the API keys and configurations needed to run **Aegis RAG** with cloud providers (Google Gemini, OpenAI, Groq, Pinecone, and Upstash Redis).

All configuration should be added to the active `.env` file at the root of the project.

---

## Quick Reference Checklist

| Environment Variable | Service | Free Tier Available? | Purpose |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google AI Studio | Yes (15 RPM) | Default LLM core & Agentic Planner |
| `OPENAI_API_KEY` | OpenAI Platform | Credit-based | Alternative LLM Provider |
| `GROQ_API_KEY` | Groq Console | Yes (Rate-limited) | Alternative LLM Provider (Ultra-fast) |
| `PINECONE_API_KEY` | Pinecone Cloud | Yes (1 Index) | Serverless Cloud Vector Database |
| `REDIS_URL` | Upstash Redis | Yes (10k commands/day) | Cloud Caching Engine |
| `TAVILY_API_KEY` | Tavily AI | Yes (1,000 queries/mo) | Web Search Tool (Phase 2 Agentic RAG) |

---

## Step-by-Step Setup

### 1. Google Gemini (Default LLM Core)
1. Go to **[Google AI Studio](https://aistudio.google.com/)**.
2. Sign in with your Google account.
3. Click **"Get API Key"** in the top left corner.
4. Click **"Create API Key"**, select a Google Cloud project, and copy the key.
5. Paste it in `.env` under:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### 2. OpenAI (Optional LLM Provider)
1. Go to the **[OpenAI API Platform](https://platform.openai.com/)**.
2. Navigate to **API Keys** on the left menu.
3. Click **"Create new secret key"**, name it, and copy it.
4. Paste it in `.env` under:
   ```ini
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### 3. Groq (Optional Fast LLM Provider)
1. Go to the **[Groq Developer Console](https://console.groq.com/)**.
2. Click on **"API Keys"** in the left sidebar.
3. Click **"Create API Key"**, copy the key.
4. Paste it in `.env` under:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   ```

### 4. Pinecone (Cloud Vector DB)
To use Pinecone, set `VECTOR_DB=pinecone` in your `.env` file.
1. Sign up/log in at the **[Pinecone Console](https://app.pinecone.io/)**.
2. Navigate to **API Keys** in the left menu.
3. Copy the default API Key (or create a new one).
4. Paste it in `.env` under:
   ```ini
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=aegis-rag  # You can name this anything you prefer
   ```
   *Note: Aegis RAG will automatically create the index for you on AWS us-east-1 serverless (or use an existing index of the same name).*

### 5. Upstash Redis (Cloud Caching)
To cache responses and embeddings in the cloud:
1. Go to the **[Upstash Console](https://console.upstash.com/)**.
2. Create a new **Redis database** (select a region closest to you).
3. In the database dashboard, scroll down to the **"Connect to your database"** section.
4. Copy the connection string under the **"Redis Connect"** tab starting with `rediss://...`.
5. Paste it in `.env` under:
   ```ini
   REDIS_URL=rediss://default:your_password@your_endpoint.upstash.io:6379
   ```

### 6. Tavily AI (Agent Web Search Tool)
If you enable agentic mode and want the planner to execute web searches:
1. Go to **[Tavily AI](https://tavily.com/)**.
2. Create a free developer account.
3. Copy the API Key from the dashboard.
4. Paste it in `.env` under:
   ```ini
   TAVILY_API_KEY=your_tavily_api_key_here
   ENABLE_WEB_SEARCH=true
   ```

---

## Activating Configurations

Once you have filled out your keys in the `.env` file:
1. Ensure `VECTOR_DB` is set to `pinecone` to route vector index calls to the cloud.
2. Start the application locally or via Docker:
   ```bash
   sudo docker compose up --build
   ```
