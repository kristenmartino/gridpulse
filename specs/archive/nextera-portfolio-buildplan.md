# NextEra Analytics Portfolio — Master Build Plan

## Overview
4 production-grade projects deployed to Google Cloud Run, showcasing the exact tech stack NextEra Analytics uses. Each project demonstrates a different competency area they're hiring for.

---

## Shared Setup (Do This First)

### GCP Account & Billing Setup

**If you don't have a Google Cloud account:**
1. Go to https://cloud.google.com/free
2. Click "Get started for free"
3. Sign in with your Google account
4. You'll get **$300 in free credits** for 90 days — more than enough for this portfolio
5. Enter billing info (credit card required but won't be charged until free credits expire)

**If you have an account but need to add/verify billing:**
1. Go to https://console.cloud.google.com/billing
2. Click "Link a billing account" or "Create account"
3. Add a payment method
4. Link billing to your project after creating it below

**Cost estimate for this portfolio:** Cloud Run is pay-per-use and extremely cheap for demo apps. With the free tier (2 million requests/month free) + $300 credits, expect to spend **$0-5 total** for these projects. Cloud Run scales to zero when not in use.

**Install the gcloud CLI (if not already installed):**
```bash
# macOS
brew install --cask google-cloud-sdk

# Or download directly
# https://cloud.google.com/sdk/docs/install

# After install, authenticate
gcloud auth login
gcloud auth application-default login
```

### GCP Project Setup
```bash
# Create a dedicated GCP project
gcloud projects create nextera-portfolio --name="NextEra Portfolio"
gcloud config set project nextera-portfolio

# Link billing (replace BILLING_ACCOUNT_ID with yours)
# Find your billing account ID:
gcloud billing accounts list
# Then link it:
gcloud billing projects link nextera-portfolio --billing-account=BILLING_ACCOUNT_ID

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

# Create Artifact Registry repo for Docker images
gcloud artifacts repositories create portfolio \
  --repository-format=docker \
  --location=us-east1 \
  --description="NextEra portfolio Docker images"

# Set region
gcloud config set run/region us-east1
```

### Shared Deployment Pattern (each project)
```bash
# Build and deploy pattern for each project
cd <project-dir>
gcloud builds submit --tag us-east1-docker.pkg.dev/nextera-portfolio/portfolio/<app-name>
gcloud run deploy <app-name> \
  --image us-east1-docker.pkg.dev/nextera-portfolio/portfolio/<app-name> \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
# Note: Energy Forecast needs --memory 2Gi (Prophet + XGBoost). 
# Other projects can use --memory 1Gi if desired.
```

---

## Project 1: Energy Demand Forecasting Dashboard
**Time estimate: 4-6 hours** | **Frontend: Plotly Dash**

> **📄 See `project1-expanded-spec.md` for the full spec** including user personas, 
> all data sources, feature engineering tables, model configurations, and interview talking points.

### What It Does
Weather-aware energy demand forecasting dashboard combining real grid data (EIA) with meteorological features (Open-Meteo) to predict electricity demand across 8 U.S. balancing authorities. Includes 4 ML models (Prophet, SARIMAX, XGBoost, Ensemble), 6 dashboard tabs, NOAA severe weather alerts, a scenario simulator, and a persona switcher.

### Tech Stack
- **Backend/Frontend**: Plotly Dash + Dash Bootstrap Components
- **ML**: Prophet (with weather regressors), SARIMAX, XGBoost, Weighted Ensemble
- **Data**: EIA API v2 (energy), Open-Meteo (weather, no key needed), NOAA (alerts)
- **Viz**: Plotly (interactive charts), SHAP (feature importance)
- **Deploy**: Docker → Cloud Run (2Gi memory)

### Data Sources
| Source | Data | Key Required? |
|--------|------|--------------|
| EIA API v2 | Hourly demand, generation by fuel type, interchange | Yes (free) |
| Open-Meteo | Temperature, wind, solar radiation, humidity, 80+ years history | No |
| NOAA/NWS | Severe weather alerts | No |

### User Personas
1. **Grid Operations Manager** — Needs 24-72hr demand forecasts for generation scheduling
2. **Renewable Portfolio Analyst** — Forecasts wind/solar output for market bidding
3. **Energy Trading Analyst** — Identifies demand-supply imbalances before price moves
4. **Data Scientist** — Compares model performance, analyzes features

### 6 Dashboard Tabs
1. **Demand Forecast** — Actual vs forecast with weather overlay, peak prediction, alerts
2. **Weather-Energy Correlation** — Scatter plots, heatmaps, feature importance, seasonal decomposition
3. **Model Comparison** — Metrics table, residual analysis, error by hour/weather
4. **Generation Mix** — Stacked area by fuel type, renewable penetration, carbon intensity
5. **Extreme Events** — NOAA alerts, anomaly detection, stress indicators
6. **Scenario Simulator** ⭐ — "What-if" weather scenarios with real-time demand/price/generation impact. Includes preset historical extremes (Winter Storm Uri, 2023 Heat Dome, Hurricane Irma) and custom scenario builder with sliders. Merit-order pricing model estimates price spikes.

### Persona Switcher
Header dropdown: Grid Ops | Renewables Analyst | Trader | Data Scientist
Each persona gets different default tab, KPI cards, alert thresholds, and a contextual welcome briefing. Demonstrates role-based UX and multi-stakeholder product thinking.

### Key Weather Features Engineered
- Cooling/Heating Degree Days (CDD/HDD)
- Wind speed at hub height (80m, 120m) → wind generation estimate
- Global Horizontal Irradiance → solar capacity factor
- Apparent temperature, dew point → AC load proxy
- Temperature deviation from 30-day rolling average
- Cyclical time encoding (sin/cos hour, day-of-week)
- Lag features (same hour yesterday, same hour last week)
- Interaction terms (temperature × hour)

### Regions (8 Balancing Authorities)
ERCOT (Texas), CAISO (California), PJM (Mid-Atlantic), MISO (Midwest), 
NYISO (New York), **FPL (Florida — NextEra's subsidiary!)**, SPP (Southwest), ISO-NE (New England)

### Claude Code Instructions
```
Build a weather-aware energy demand forecasting dashboard. 
Read the full spec from project1-expanded-spec.md for complete details on:
- Data sources and API endpoints
- Weather feature engineering table  
- Model configurations (Prophet, SARIMAX, XGBoost, Ensemble)
- 6 dashboard tabs with component specifications
- Scenario Simulator with preset historical extremes and custom builder
- Persona Switcher with per-role configuration
- Project structure with 25+ files
- Dockerfile and Cloud Run deployment

Key points:
- Use EIA API v2 for energy data, Open-Meteo for weather (no key needed)
- Include FPL (NextEra's subsidiary) as a selectable region
- Dark theme, professional energy utility aesthetic
- SQLite caching for API data
- Feature engineering: CDD, HDD, wind power estimate, solar capacity factor,
  cyclical time encoding, lag features, temperature interactions
- 4 forecasting models with ensemble combining
- SHAP values for XGBoost feature importance
- NOAA weather alerts integration
- Tab 6 Scenario Simulator: sliders for temp/wind/cloud/humidity,
  preset scenarios (Uri, Heat Dome, Irma, etc.), real-time impact charts,
  simplified merit-order pricing model, scenario comparison mode
- Persona Switcher in header: 4 roles with different default tabs,
  KPI cards, alert thresholds, and contextual welcome briefings
- Gunicorn with 2 workers, port 8080
- structlog for logging
- Type hints and docstrings throughout
```

---

## Project 2: Grid Anomaly Detection & Alert System
**Time estimate: 3-4 hours** | **Frontend: Streamlit + Plotly**

### What It Does
Real-time grid monitoring dashboard that ingests simulated sensor data (voltage, frequency, load), detects anomalies using multiple ML approaches (Isolation Forest, DBSCAN, statistical), and surfaces alerts with root cause analysis.

### Tech Stack
- **Frontend**: Streamlit with Plotly charts
- **ML**: scikit-learn (Isolation Forest, DBSCAN, LOF), scipy (statistical tests)
- **Data**: Simulated realistic grid data + EIA actuals + Open-Meteo weather (anomaly correlation)
- **Deploy**: Docker → Cloud Run

### Key Features
1. Real-time grid topology visualization
2. Multi-method anomaly detection (ensemble approach)
3. Alert severity classification (Critical/Warning/Info)
4. Historical anomaly timeline
5. Root cause analysis suggestions
6. Configurable detection thresholds
7. **Weather-correlated anomaly analysis** (do anomalies spike during storms/heat waves?)
8. Export alerts to CSV

### Architecture
```
Simulated Grid Data → Anomaly Detection Pipeline
  → Isolation Forest ──────┐
  → Statistical Tests ───── ├─→ Ensemble Scorer → Alert System → Streamlit UI
  → DBSCAN Clustering ─────┘
```

### Claude Code Instructions
```
Build a grid anomaly detection and alert system using Streamlit, scikit-learn, and Plotly.

Project Structure:
grid-anomaly/
├── app.py                    # Streamlit app entry point
├── data/
│   ├── generator.py          # Realistic grid data simulator
│   ├── schemas.py            # Pydantic data models
│   └── sample_data/          # Pre-generated datasets
├── detection/
│   ├── isolation_forest.py   # Isolation Forest detector
│   ├── statistical.py        # Z-score, Grubbs, CUSUM
│   ├── clustering.py         # DBSCAN-based detection
│   └── ensemble.py           # Ensemble anomaly scorer
├── alerts/
│   ├── classifier.py         # Severity classification
│   └── root_cause.py         # Root cause analysis logic
├── visualization/
│   ├── grid_topology.py      # Network/topology visualization
│   ├── timeseries.py         # Time series charts with anomaly markers
│   └── alert_dashboard.py    # Alert summary components
├── Dockerfile
├── requirements.txt
└── README.md

Data Simulation:
- Generate realistic voltage (±5% of 120V/240V/13.8kV), frequency (59.95-60.05 Hz), 
  load (MW), power factor, temperature readings
- Inject realistic anomalies: voltage sags, frequency deviations, 
  sudden load spikes, cascading failures
- Time-series with 15-min intervals over 90 days
- Multiple substations/feeders

Detection Requirements:
- Isolation Forest: contamination parameter tunable via sidebar
- Statistical: Z-score with configurable sigma, CUSUM for drift detection
- DBSCAN: for spatial clustering of related anomalies
- Ensemble: weighted voting across methods with configurable weights
- Confusion matrix and detection metrics on labeled synthetic data

UI Requirements:
- Professional dark theme (st.set_page_config with dark mode)
- Sidebar with detection parameters and filters
- Main view: real-time-style time series with anomaly markers
- Grid topology view showing affected nodes
- Alert table with severity, timestamp, affected equipment, suggested action
- Model performance comparison tab
- Use st.columns for responsive layout
- Plotly for all charts (no matplotlib)

Best Practices:
- Pydantic models for data validation
- Type hints and docstrings
- Modular detection pipeline (easy to add new methods)
- Streamlit caching (@st.cache_data) for expensive computations
- Session state for interactive filtering
```

---

## Project 3: Renewable Energy RAG Assistant
**Time estimate: 4-5 hours** | **Frontend: Streamlit**

### What It Does
AI-powered assistant that answers questions about renewable energy policy, regulations, and technical standards using RAG over public FERC, EIA, and DOE documents. Demonstrates production-grade RAG with evaluation.

### Tech Stack
- **Framework**: LangChain
- **Vector Store**: ChromaDB (local) or Vertex AI Vector Search
- **LLM**: Claude API (via Anthropic SDK) or Gemini via Vertex AI
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) or Vertex AI embeddings
- **Frontend**: Streamlit
- **Deploy**: Docker → Cloud Run

### Key Features
1. Chat interface with source citations
2. Document ingestion pipeline (PDF, HTML, TXT)
3. Hybrid search (semantic + keyword via BM25)
4. Chunk visualization — show which chunks were retrieved
5. Retrieval quality metrics
6. Conversation memory with context window management
7. Admin panel for document management

### Architecture
```
Documents (FERC/EIA/DOE PDFs) → Chunking → Embedding → ChromaDB
                                                          ↓
User Query → Hybrid Search (Semantic + BM25) → Re-ranking → LLM → Response + Citations
                                                                      ↓
                                                              Streamlit Chat UI
```

### Claude Code Instructions
```
Build a Renewable Energy RAG Assistant using LangChain, ChromaDB, and Streamlit.

Project Structure:
energy-rag/
├── app.py                      # Streamlit app entry point
├── config.py                   # Settings and environment config
├── ingestion/
│   ├── loader.py               # Document loaders (PDF, HTML, TXT)
│   ├── chunker.py              # Smart chunking with overlap
│   ├── embedder.py             # Embedding pipeline
│   └── pipeline.py             # End-to-end ingestion orchestrator
├── retrieval/
│   ├── vector_search.py        # ChromaDB semantic search
│   ├── bm25_search.py          # BM25 keyword search
│   ├── hybrid.py               # Hybrid search with RRF fusion
│   └── reranker.py             # Cross-encoder re-ranking
├── generation/
│   ├── chain.py                # LangChain RAG chain
│   ├── prompts.py              # System prompts with few-shot examples
│   └── memory.py               # Conversation memory management
├── evaluation/
│   ├── retrieval_metrics.py    # Precision@k, Recall@k, MRR
│   └── generation_metrics.py   # Faithfulness, relevance scoring
├── ui/
│   ├── chat.py                 # Chat interface components
│   ├── sources.py              # Source citation display
│   ├── admin.py                # Document management panel
│   └── metrics.py              # Retrieval quality visualization
├── documents/                  # Pre-loaded public energy docs
│   └── README.md               # Links to source documents
├── Dockerfile
├── requirements.txt
└── README.md

Document Sources (public, no copyright issues):
- EIA Annual Energy Outlook summaries
- FERC Order 2222 (distributed energy resources)
- DOE SunShot Initiative reports
- NREL technical reports on wind/solar
- Public utility commission rulings

LLM Configuration:
- Primary: Anthropic Claude (via anthropic SDK) — use claude-sonnet-4-20250514
- Fallback: Can swap to Gemini via langchain-google-genai
- Temperature: 0.1 for factual responses
- System prompt emphasizing citation accuracy and energy domain expertise

Chunking Strategy:
- Recursive character splitter: 1000 chars, 200 overlap
- Preserve section headers as metadata
- Store document source, page number, section in metadata

Search Configuration:
- Hybrid search: 0.7 semantic + 0.3 BM25, fused with Reciprocal Rank Fusion
- Top-k: retrieve 10, re-rank to top 5
- Cross-encoder re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2

UI Requirements:
- Clean chat interface with message history
- Expandable source citations under each response
- Sidebar: document selector, search method toggle, chunk size controls
- Admin tab: upload new documents, view ingestion status
- Metrics tab: show retrieval quality on test queries
- Professional energy company styling

Best Practices:
- Async document processing
- Streaming responses
- Error handling for LLM failures
- Rate limiting awareness
- Comprehensive logging
- Unit tests for retrieval pipeline
```

---

## Project 4: Multi-Agent Energy News Intelligence
**Time estimate: 5-6 hours** | **Frontend: React + FastAPI**

### What It Does
LangGraph-powered multi-agent system that autonomously researches energy news across multiple sources, classifies and analyzes articles, generates executive briefings, and surfaces actionable market intelligence. The most technically impressive project.

### Tech Stack
- **Agent Framework**: LangGraph (multi-agent orchestration)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: React (Vite) with Tailwind CSS
- **LLM**: Claude API
- **Data**: News APIs (NewsAPI, Google News RSS), web scraping
- **Deploy**: Docker multi-stage → Cloud Run

### Key Features
1. Multi-agent workflow visualization (real-time graph state)
2. Parallel research agents (solar, wind, grid, policy, market)
3. Source credibility scoring
4. Conflict resolution between contradictory sources
5. Executive summary generation with key takeaways
6. Topic trend analysis over time
7. WebSocket real-time updates during agent execution

### Architecture
```
User Query → Orchestrator Agent
  → Research Agent (Solar) ────┐
  → Research Agent (Wind) ───── │
  → Research Agent (Grid) ───── ├─→ Synthesis Agent → Quality Evaluator → Executive Brief
  → Research Agent (Policy) ─── │
  → Research Agent (Market) ───┘
                                        ↓
FastAPI (WebSocket) ←──────────── LangGraph State ──────────→ React Dashboard
```

### Claude Code Instructions
```
Build a Multi-Agent Energy News Intelligence system using LangGraph, FastAPI, and React.

Project Structure:
energy-intel/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings
│   ├── agents/
│   │   ├── orchestrator.py        # Main LangGraph orchestrator
│   │   ├── researcher.py          # Research agent (parameterized by topic)
│   │   ├── synthesizer.py         # Synthesis and conflict resolution
│   │   ├── evaluator.py           # Quality and credibility scoring
│   │   └── briefing.py            # Executive summary generator
│   ├── graph/
│   │   ├── state.py               # LangGraph state definition
│   │   ├── nodes.py               # Graph node functions
│   │   ├── edges.py               # Conditional edge logic
│   │   └── workflow.py            # Graph compilation
│   ├── sources/
│   │   ├── news_api.py            # NewsAPI client
│   │   ├── rss_feeds.py           # Energy RSS feed parser
│   │   └── web_scraper.py         # Fallback web scraper
│   ├── models/
│   │   ├── schemas.py             # Pydantic models
│   │   └── enums.py               # Topic categories, severity levels
│   ├── api/
│   │   ├── routes.py              # REST endpoints
│   │   └── websocket.py           # WebSocket for real-time updates
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── AgentWorkflow.jsx    # Visual graph of agent execution
│   │   │   ├── BriefingCard.jsx     # Executive summary display
│   │   │   ├── SourceList.jsx       # Source credibility display
│   │   │   ├── TopicFilter.jsx      # Topic category filters
│   │   │   ├── TrendChart.jsx       # Recharts trend visualization
│   │   │   └── StatusIndicator.jsx  # Agent execution status
│   │   ├── hooks/
│   │   │   └── useWebSocket.js      # WebSocket hook
│   │   └── api/
│   │       └── client.js            # API client
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
└── README.md

LangGraph Workflow:
1. Orchestrator receives user query, determines relevant topics
2. Parallel research agents fan out (one per topic)
3. Each researcher: searches news → scrapes content → extracts key facts
4. Synthesis agent: merges findings, resolves conflicts, identifies themes
5. Evaluator: scores source credibility, checks for bias
6. Briefing agent: generates executive summary with actionable insights
7. State updates streamed via WebSocket at each step

Agent Design:
- Each agent is a LangGraph node with typed state
- Use conditional edges for routing (e.g., skip synthesis if only one topic)
- Implement retry logic with exponential backoff
- Human-in-the-loop breakpoints for high-stakes decisions

FastAPI Backend:
- POST /api/research — start new research workflow
- GET /api/briefings — list past briefings
- GET /api/briefings/{id} — get specific briefing
- WS /ws/research/{id} — real-time execution updates
- GET /health — health check

React Frontend:
- Vite + React 18 + Tailwind CSS
- Real-time agent execution visualization (show which agents are active)
- Recharts for trend analysis
- Responsive grid layout
- Dark mode energy-industry aesthetic
- Loading states and error boundaries

Best Practices:
- Pydantic v2 for all data models
- Structured logging (structlog)
- Async throughout (FastAPI + httpx)
- Docker multi-stage builds (node build → nginx for frontend, python for backend)
- CORS configuration for local dev
- Environment-based configuration
- Comprehensive error handling
- Rate limiting on news APIs
```

---

## Portfolio Landing Page
After all 4 projects are deployed, create a simple landing page:

```
portfolio-landing/
├── index.html          # Single page with links to all 4 apps
├── style.css           # Professional dark theme
├── Dockerfile
└── README.md
```

Deploy to Cloud Run as `portfolio.nextera-portfolio` and map a custom domain if desired.

---

## Build Order & Timeline

| Day | Session | Project | Hours |
|-----|---------|---------|-------|
| 1 | Morning | GCP Setup + Energy Forecast Dashboard (core tabs 1-3) | 4-5h |
| 1 | Afternoon | Forecast Dashboard (tabs 4-6, polish) + Grid Anomaly Detection | 4-5h |
| 2 | Morning | RAG Assistant | 4-5h |
| 2 | Afternoon | Multi-Agent News Intel + Landing Page | 5-6h |

**Tip**: If time gets tight, the Forecast Dashboard tabs 4-6 and Grid Anomaly Detection can each stand alone as MVPs with fewer features. Ship core functionality first, then polish.

---

## Claude Code Tips for Fastest Execution

1. **Start each project** with: `claude "Read the build plan in nextera-portfolio-buildplan.md for Project N and build it end-to-end"`
2. **Use `/init`** at the start to set up CLAUDE.md with project conventions
3. **Test locally first**: `docker build . && docker run -p 8080:8080 .`
4. **Deploy pattern**: `gcloud run deploy <name> --source . --allow-unauthenticated`
5. **API keys**: Store in GCP Secret Manager, reference in Cloud Run env vars
6. **If stuck on frontend styling**: Tell Claude Code "make it look like a professional energy utility dashboard with dark theme"

## Pre-Flight Checklist
- [ ] GCP account created (https://cloud.google.com/free — $300 free credits)
- [ ] Billing linked to GCP project
- [ ] gcloud CLI installed and authenticated (`gcloud auth login`)
- [ ] EIA API key (free): https://www.eia.gov/opendata/register.php
- [ ] Anthropic API key (for RAG + Agents): https://console.anthropic.com/
- [ ] NewsAPI key (free tier): https://newsapi.org/register
