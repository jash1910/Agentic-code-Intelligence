import { create } from 'zustand';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const useArchonStore = create((set, get) => ({
  messages: [{
    id: 'welcome',
    role: 'assistant',
    content: '### Welcome to ArchonAI\n\nAgentic code intelligence platform with hybrid RAG, knowledge graphs, and multi-agent orchestration.\n\nSelect a repository and ask anything about LangChain, CrewAI, LlamaIndex, and more.',
    citations: [],
    confidence: null,
    agentTrace: [],
  }],
  input: '',
  isGenerating: false,
  repository: 'langchain',
  repositories: ['langchain', 'crewai', 'llamaindex', 'pydantic-ai', 'haystack', 'dspy', 'autogen'],
  sessionId: `session-${Date.now()}`,
  sidebarOpen: true,
  citationsOpen: false,
  traceOpen: false,
  activeMessageId: null,
  backendHealth: { status: 'checking', mode: 'unknown', model: 'unknown', version: '2.0.0' },

  setInput: (input) => set({ input }),
  setRepository: (repository) => set({ repository }),
  toggleSidebar: () => set(s => ({ sidebarOpen: !s.sidebarOpen })),
  setCitationsOpen: (open) => set({ citationsOpen: open }),
  setTraceOpen: (open) => set({ traceOpen: open }),
  setActiveMessageId: (id) => set({ activeMessageId: id }),

  fetchHealth: async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      set({ backendHealth: { status: 'online', ...data } });
    } catch {
      set({ backendHealth: { status: 'offline', mode: 'offline', model: 'offline', version: '2.0.0' } });
    }
  },

  sendMessage: async (text) => {
    const query = text || get().input;
    if (!query.trim() || get().isGenerating) return;

    const userId = `msg-${Date.now()}`;
    const assistantId = `msg-${Date.now() + 1}`;

    set(s => ({
      input: text ? s.input : '',
      isGenerating: true,
      citationsOpen: false,
      messages: [
        ...s.messages,
        { id: userId, role: 'user', content: query, citations: [] },
        { id: assistantId, role: 'assistant', content: '', citations: [], agentTrace: [], confidence: null, isStreaming: true },
      ],
    }));

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query,
          session_id: get().sessionId,
          repository: get().repository,
        }),
      });

      if (!res.ok) throw new Error('API error');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let citations = [];
      let agentTrace = [];
      let confidence = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          try {
            const payload = JSON.parse(trimmed.slice(6));

            if (payload.citations) {
              citations = payload.citations;
              set(s => ({
                messages: s.messages.map(m =>
                  m.id === assistantId ? { ...m, citations } : m
                ),
                citationsOpen: citations.length > 0,
                activeMessageId: assistantId,
              }));
            }

            if (payload.text) {
              set(s => ({
                messages: s.messages.map(m =>
                  m.id === assistantId ? { ...m, content: m.content + payload.text } : m
                ),
              }));
            }

            if (payload.agent) {
              agentTrace = [...agentTrace, payload];
              set(s => ({
                messages: s.messages.map(m =>
                  m.id === assistantId ? { ...m, agentTrace: [...agentTrace] } : m
                ),
              }));
            }

            if (payload.confidence_score !== undefined) {
              confidence = payload.confidence_score;
              agentTrace = payload.agent_trace || agentTrace;
              set(s => ({
                messages: s.messages.map(m =>
                  m.id === assistantId
                    ? { ...m, confidence, agentTrace, isStreaming: false }
                    : m
                ),
                traceOpen: agentTrace.length > 0,
                activeMessageId: assistantId,
              }));
            }

            if (payload.status === 'completed') {
              set(s => ({
                messages: s.messages.map(m =>
                  m.id === assistantId ? { ...m, isStreaming: false } : m
                ),
              }));
            }
          } catch { /* skip parse errors */ }
        }
      }
    } catch (err) {
      set(s => ({
        messages: s.messages.map(m =>
          m.id === assistantId
            ? { ...m, content: `⚠️ **Error:** ${err.message}. Start backend with \`uvicorn backend.app.main:app\`.`, isStreaming: false }
            : m
        ),
      }));
    } finally {
      set({ isGenerating: false });
    }
  },
}));
