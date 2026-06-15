"use client";

import React, { useState, useEffect, useRef } from "react";

interface SourceDoc {
  source: string;
  type: string;
  page?: number | null;
  filename?: string | null;
  url?: string | null;
}

interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  mode?: "basic" | "agentic";
  sources?: SourceDoc[];
  steps?: { name: string; desc: string; status: "done" | "active" | "error" | "pending" }[];
  cacheHit?: boolean;
}

interface SystemStats {
  totalQueries: number;
  hitRatePercent: number;
  hits: number;
  misses: number;
  llmModel: string;
  vectorDb: string;
  chunkSize: number;
}

interface IngestedDoc {
  filename: string;
  filepath: string;
  file_hash: string;
  last_modified: number;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "initial",
      sender: "assistant",
      text: "Welcome to Aegis RAG. Upload files in the sidebar to index them, then configure your LLM and query the database.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [useAgentic, setUseAgentic] = useState(true);
  const [returnSources, setReturnSources] = useState(true);
  const [userId, setUserId] = useState("aegis_dev_user");
  
  // Custom LLM Settings
  const [llmProvider, setLlmProvider] = useState("gemini");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmModel, setLlmModel] = useState("gemini-1.5-flash");
  
  // Ingested Documents State
  const [documents, setDocuments] = useState<IngestedDoc[]>([]);
  const [uploading, setUploading] = useState(false);
  
  // System states
  const [backendStatus, setBackendStatus] = useState<"online" | "offline">("offline");
  const [systemInfo, setSystemInfo] = useState({
    environment: "dev",
    vectorDb: "Chroma",
    cacheEnabled: true,
  });
  const [stats, setStats] = useState<SystemStats>({
    totalQueries: 0,
    hitRatePercent: 0,
    hits: 0,
    misses: 0,
    llmModel: "gemini-1.5-flash",
    vectorDb: "Chroma",
    chunkSize: 500,
  });

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Helper to set default model based on provider choice
  const handleProviderChange = (provider: string) => {
    setLlmProvider(provider);
    if (provider === "gemini") {
      setLlmModel("gemini-1.5-flash");
    } else if (provider === "openai") {
      setLlmModel("gpt-4o");
    } else if (provider === "groq") {
      setLlmModel("llama3-8b-8192");
    }
  };

  // Fetch documents list
  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (e) {
      console.warn("Failed to fetch documents catalog", e);
    }
  };

  // Poll backend stats and health
  const checkHealthAndStats = async () => {
    try {
      const healthRes = await fetch(`${API_BASE_URL}/health`);
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setBackendStatus("online");
        setSystemInfo({
          environment: healthData.environment,
          vectorDb: healthData.vector_db,
          cacheEnabled: healthData.cache_enabled,
        });

        // Fetch stats
        const statsRes = await fetch(`${API_BASE_URL}/stats`);
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          const cache = statsData.cache_stats || {};
          setStats({
            totalQueries: cache.total_queries || 0,
            hitRatePercent: Math.round(cache.hit_rate_percent || 0),
            hits: cache.cache_hits || 0,
            misses: cache.cache_misses || 0,
            llmModel: statsData.config?.llm_model || "gemini-1.5-flash",
            vectorDb: statsData.config?.vector_db || "Chroma",
            chunkSize: statsData.config?.chunk_size || 500,
          });
        }
      } else {
        setBackendStatus("offline");
      }
    } catch (error) {
      setBackendStatus("offline");
    }
  };

  // Ingest/Upload document handler
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        await fetchDocuments();
        await checkHealthAndStats();
        
        setMessages((prev) => [
          ...prev,
          {
            id: `upload_${Date.now()}`,
            sender: "assistant",
            text: `Successfully ingested file(s): ${Array.from(files)
              .map((f) => f.name)
              .join(", ")}.\nThe content has been chunked, embedded, and indexed into the ${systemInfo.vectorDb} vector store.`,
          },
        ]);
      } else {
        const errData = await res.json();
        alert(`Ingestion failed: ${errData.detail || "Server error"}`);
      }
    } catch (err) {
      console.error("Upload error", err);
      alert("Failed to connect to RAG backend for document ingestion.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  useEffect(() => {
    checkHealthAndStats();
    fetchDocuments();
    const interval = setInterval(() => {
      checkHealthAndStats();
      fetchDocuments();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `assistant_${Date.now()}`;

    // Add user message
    const userMessage: Message = {
      id: userMsgId,
      sender: "user",
      text: query,
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);

    try {
      // Send query to API
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: userMessage.text,
          user_id: userId,
          return_sources: returnSources,
          use_agentic: useAgentic,
          llm_provider: llmProvider,
          llm_api_key: llmApiKey.trim() || null,
          llm_model: llmModel,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server error");
      }

      const data = await response.json();
      
      // Determine steps if agentic
      let steps = undefined;
      if (useAgentic) {
        steps = [
          { name: "Classification", desc: `Routed using provider: ${llmProvider.toUpperCase()}`, status: "done" as const },
          { name: "Execution Plan", desc: `Generated plan with ${data.metadata?.retry_count ? data.metadata.retry_count + 1 : 1} steps`, status: "done" as const },
          { name: "Routing & Logic", desc: data.metadata?.question_rewritten ? "Rewrote original query for retrieval" : "Standard query routing", status: "done" as const },
          { name: "Generation", desc: `Synthesized final answer using ${llmModel}`, status: "done" as const }
        ];
      }

      const assistantMessage: Message = {
        id: assistantMsgId,
        sender: "assistant",
        text: data.answer,
        mode: useAgentic ? "agentic" : "basic",
        sources: data.sources || [],
        steps,
        cacheHit: data.metadata?.cache_stats?.cache_hits > 0 || false,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      
      // Refresh stats
      checkHealthAndStats();

    } catch (err: any) {
      console.warn("API Call failed, falling back to local simulation.", err);
      
      const errorMsg = err.message || "Failed to contact API server.";

      // Add error feedback
      const assistantMessage: Message = {
        id: assistantMsgId,
        sender: "assistant",
        text: `Error: ${errorMsg}\n\nPlease check that your API Key is valid or that the backend service is running locally on port 8000.`,
      };
      setMessages((prev) => [...prev, assistantMessage]);

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand-section">
          <h1 className="brand-title">
            AEGIS <span>RAG</span>
          </h1>
          <p className="brand-subtitle">Autonomous Retrieval</p>
          
          <nav className="nav-links">
            <div className="nav-item active">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="9" rx="1" />
                <rect x="14" y="3" width="7" height="5" rx="1" />
                <rect x="14" y="12" width="7" height="9" rx="1" />
                <rect x="3" y="16" width="7" height="5" rx="1" />
              </svg>
              Query Engine
            </div>
          </nav>
        </div>

        {/* Document Ingestion / Upload */}
        <div className="config-card" style={{ background: "rgba(25, 25, 25, 0.4)", border: "1px solid var(--border-color)", padding: "1rem", marginTop: "0.5rem", borderRadius: "8px" }}>
          <h3 className="config-title" style={{ fontSize: "0.85rem", letterSpacing: "0.05em", color: "var(--accent)" }}>INGEST DOCUMENTS</h3>
          <div className="config-group" style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "0.5rem" }}>
            
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              multiple
              accept=".txt,.md,.pdf,.docx"
              style={{ display: "none" }}
            />
            
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              style={{
                width: "100%",
                background: "var(--accent)",
                color: "var(--background)",
                border: "none",
                borderRadius: "4px",
                padding: "0.5rem",
                fontWeight: "600",
                fontSize: "0.85rem",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
            >
              {uploading ? "Ingesting..." : "Upload Files"}
            </button>
            <p style={{ fontSize: "0.7rem", color: "#808080", textAlign: "center" }}>
              Supports .txt, .md, .pdf, .docx
            </p>
          </div>
        </div>

        {/* LLM Connection Panel */}
        <div className="config-card" style={{ background: "rgba(25, 25, 25, 0.4)", border: "1px solid var(--border-color)", padding: "1rem", marginTop: "0.5rem", borderRadius: "8px" }}>
          <h3 className="config-title" style={{ fontSize: "0.85rem", letterSpacing: "0.05em", color: "var(--accent)" }}>LLM CONNECTION</h3>
          <div className="config-group" style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginTop: "0.5rem" }}>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <label style={{ fontSize: "0.75rem", color: "#a0a0a0" }}>Provider</label>
              <select
                value={llmProvider}
                onChange={(e) => handleProviderChange(e.target.value)}
                style={{
                  background: "var(--card-bg)",
                  color: "var(--foreground)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "4px",
                  padding: "0.35rem 0.5rem",
                  fontSize: "0.85rem",
                  outline: "none"
                }}
              >
                <option value="gemini">Google Gemini</option>
                <option value="openai">OpenAI</option>
                <option value="groq">Groq</option>
              </select>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <label style={{ fontSize: "0.75rem", color: "#a0a0a0" }}>Model</label>
              <select
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                style={{
                  background: "var(--card-bg)",
                  color: "var(--foreground)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "4px",
                  padding: "0.35rem 0.5rem",
                  fontSize: "0.85rem",
                  outline: "none"
                }}
              >
                {llmProvider === "gemini" && (
                  <>
                    <option value="gemini-1.5-flash">gemini-1.5-flash</option>
                    <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                  </>
                )}
                {llmProvider === "openai" && (
                  <>
                    <option value="gpt-4o">gpt-4o</option>
                    <option value="gpt-4-turbo">gpt-4-turbo</option>
                    <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                  </>
                )}
                {llmProvider === "groq" && (
                  <>
                    <option value="llama3-8b-8192">llama3-8b-8192</option>
                    <option value="llama3-70b-8192">llama3-70b-8192</option>
                    <option value="mixtral-8x7b-32768">mixtral-8x7b-32768</option>
                  </>
                )}
              </select>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <label style={{ fontSize: "0.75rem", color: "#a0a0a0" }}>API Key</label>
              <input
                type="password"
                value={llmApiKey}
                onChange={(e) => setLlmApiKey(e.target.value)}
                placeholder={
                  llmProvider === "gemini" 
                    ? "Falls back to server env key" 
                    : `Enter ${llmProvider.toUpperCase()} API Key`
                }
                style={{
                  background: "var(--card-bg)",
                  color: "var(--foreground)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "4px",
                  padding: "0.35rem 0.5rem",
                  fontSize: "0.85rem",
                  outline: "none"
                }}
              />
            </div>

          </div>
        </div>

        <div className="status-panel">
          <div className="status-header">Server Health</div>
          <div className="status-badge">
            <span className="status-label">API Server</span>
            <span className="status-value">
              <span className={`status-indicator ${backendStatus === "online" ? "online" : ""}`} />
              {backendStatus === "online" ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
          <div className="status-badge">
            <span className="status-label">Database</span>
            <span className="status-value">{systemInfo.vectorDb}</span>
          </div>
          <div className="status-badge">
            <span className="status-label">Caching</span>
            <span className="status-value">{systemInfo.cacheEnabled ? "ACTIVE" : "INACTIVE"}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <div className="header-container">
          <div>
            <h2 className="page-title">Cognitive Query Engine</h2>
            <p className="page-description">
              Direct interface into the agentic reasoning, grading, and document retrieval graph.
            </p>
          </div>
        </div>

        {/* Playground Grid */}
        <div className="playground-grid">
          {/* Chat Container */}
          <div className="chat-card">
            <div className="chat-header">
              <div className="chat-title-group">
                <div className="chat-dot" />
                <span className="chat-header-title">Interaction Pipeline</span>
              </div>
              <span className="status-value" style={{ textTransform: "uppercase" }}>
                {useAgentic ? "Agentic Mode (LangGraph)" : "Standard Mode (Basic RAG)"}
              </span>
            </div>

            <div className="chat-messages">
              {messages.map((msg) => (
                <div key={msg.id} className={`message-bubble ${msg.sender}`}>
                  <div className="message-avatar-group">
                    <span className={`message-sender ${msg.sender === "assistant" ? "assistant-label" : ""}`}>
                      {msg.sender === "user" ? "Client Prompt" : "Aegis System"}
                    </span>
                    {msg.cacheHit && (
                      <span className="status-value" style={{ color: "var(--success)", borderColor: "var(--success)", fontSize: "0.65rem", padding: "0 0.25rem" }}>
                        CACHE HIT
                      </span>
                    )}
                  </div>
                  <div className="message-content">
                    <p style={{ whiteSpace: "pre-line" }}>{msg.text}</p>
                    
                    {/* Steps display if agentic */}
                    {msg.steps && msg.steps.length > 0 && (
                      <div className="agent-steps">
                        <div className="sources-header">Agent Execution Trace</div>
                        {msg.steps.map((step, idx) => (
                          <div key={idx} className="agent-step done">
                            <div className="agent-step-icon">✓</div>
                            <div className="agent-step-details">
                              <span className="agent-step-name">{step.name}</span>
                              <span className="agent-step-desc">{step.desc}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Sources display */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources-section">
                        <div className="sources-header">Reference Material</div>
                        <div className="sources-grid">
                          {msg.sources.map((src, idx) => (
                            <span key={idx} className="source-tag">
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14 2 14 8 20 8" />
                              </svg>
                              {src.filename || src.source}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message-bubble assistant">
                  <div className="message-avatar-group">
                    <span className="message-sender assistant-label">Aegis System</span>
                  </div>
                  <div className="skeleton-loading">
                    <div className="skeleton-line" />
                    <div className="skeleton-line" />
                    <div className="skeleton-line short" />
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="chat-input-container">
              <form onSubmit={handleSend} className="chat-input-form">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={`Ask a question using ${llmProvider.toUpperCase()} (${llmModel})...`}
                  className="chat-input"
                  disabled={loading}
                />
                <button type="submit" className="send-button" disabled={loading || !query.trim()}>
                  {loading ? "Thinking..." : "Query"}
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polyline points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                </button>
              </form>
            </div>
          </div>

          {/* Right Panel / Configuration & Document Catalog */}
          <div className="config-panel">
            {/* Document Catalog Explorer */}
            <div className="config-card">
              <h3 className="config-title">Ingested Knowledge Base</h3>
              <div className="config-group" style={{ maxHeight: "200px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                {documents.length === 0 ? (
                  <p style={{ fontSize: "0.8rem", color: "#808080", fontStyle: "italic", textAlign: "center", padding: "1rem 0" }}>
                    No documents ingested yet. Upload files to index them in the database.
                  </p>
                ) : (
                  documents.map((doc, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: "rgba(255, 255, 255, 0.03)",
                        border: "1px solid var(--border-color)",
                        borderRadius: "4px",
                        padding: "0.5rem",
                        display: "flex",
                        flexDirection: "column",
                        gap: "0.25rem"
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                        </svg>
                        <span style={{ fontSize: "0.8rem", fontWeight: "600", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {doc.filename}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "#a0a0a0" }}>
                        <span>Hash: {doc.file_hash.substring(0, 8)}...</span>
                        <span>{new Date(doc.last_modified * 1000).toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="config-card">
              <h3 className="config-title">Options</h3>
              <div className="config-group">
                <div className="toggle-container" onClick={() => setUseAgentic(!useAgentic)}>
                  <div className="toggle-label-group">
                    <span className="toggle-label">Agentic Reasoning</span>
                    <span className="toggle-desc">Plan-and-solve execution loop</span>
                  </div>
                  <label className="switch">
                    <input type="checkbox" checked={useAgentic} readOnly />
                    <span className="slider" />
                  </label>
                </div>

                <div className="toggle-container" onClick={() => setReturnSources(!returnSources)}>
                  <div className="toggle-label-group">
                    <span className="toggle-label">Inspect Citations</span>
                    <span className="toggle-desc">Return referenced documents</span>
                  </div>
                  <label className="switch">
                    <input type="checkbox" checked={returnSources} readOnly />
                    <span className="slider" />
                  </label>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem", marginTop: "0.5rem" }}>
                  <label className="toggle-label" style={{ fontSize: "0.8rem" }}>Tenant / Client Profile</label>
                  <input
                    type="text"
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    className="chat-input"
                    style={{ padding: "0.5rem 0.75rem", fontSize: "0.85rem" }}
                  />
                </div>
              </div>
            </div>

            <div className="config-card">
              <h3 className="config-title">Performance Analytics</h3>
              <div className="config-group">
                <div className="stat-box">
                  <div className="stat-item">
                    <div className="stat-num">{stats.totalQueries}</div>
                    <div className="stat-lbl">Queries</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-num">{stats.hitRatePercent}%</div>
                    <div className="stat-lbl">Hit Rate</div>
                  </div>
                </div>

                <div className="stat-box" style={{ marginTop: 0 }}>
                  <div className="stat-item">
                    <div className="stat-num">{stats.hits}</div>
                    <div className="stat-lbl">Cache Hits</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-num">{stats.misses}</div>
                    <div className="stat-lbl">Cache Misses</div>
                  </div>
                </div>

                <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "1rem", marginTop: "0.5rem" }}>
                  <div className="status-badge" style={{ marginBottom: "0.5rem" }}>
                    <span className="status-label">LLM Core</span>
                    <span className="status-value">{stats.llmModel}</span>
                  </div>
                  <div className="status-badge" style={{ marginBottom: "0.5rem" }}>
                    <span className="status-label">DB Backend</span>
                    <span className="status-value">{stats.vectorDb}</span>
                  </div>
                  <div className="status-badge">
                    <span className="status-label">Chunk Boundary</span>
                    <span className="status-value">{stats.chunkSize} chars</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
