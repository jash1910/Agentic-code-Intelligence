import { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Sparkles, Terminal, Menu, X, FileText, GitBranch,
  Activity, ChevronRight, Shield, Zap
} from 'lucide-react';
import { useArchonStore } from './store/archonStore';
import {
  MessageBubble, AgentTracePanel, CitationsPanel, LoadingDots
} from './components/ChatUI';

const PRESETS = [
  'How do you define a custom Tool with multiple arguments in LangChain?',
  'Explain ChatPromptTemplate with System and Human messages.',
  'What is the difference between LLMChain and LCEL pipe chains?',
  'How does AgentExecutor depend on BaseAgent and BaseTool?',
  'Implement RunnablePassthrough in an LCEL chain.',
  'How do you set up hybrid retrieval with ChromaDB?',
];

export default function App() {
  const {
    messages, input, isGenerating, repository, repositories,
    sidebarOpen, citationsOpen, traceOpen, activeMessageId, backendHealth,
    setInput, setRepository, toggleSidebar, setCitationsOpen, setTraceOpen,
    setActiveMessageId, fetchHealth, sendMessage,
  } = useArchonStore();

  useEffect(() => { fetchHealth(); }, [fetchHealth]);

  const activeMessage = messages.find(m => m.id === activeMessageId);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#06080f] text-slate-100">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="flex flex-col border-r border-white/5 bg-[#080c16]/80 backdrop-blur-xl shrink-0 overflow-hidden"
          >
            <div className="p-5 border-b border-white/5 flex items-center gap-3">
              <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                <Terminal size={18} className="text-indigo-400" />
              </div>
              <div>
                <p className="font-bold text-sm">ArchonAI</p>
                <p className="text-[10px] text-slate-500 uppercase tracking-widest">Control Panel</p>
              </div>
            </div>

            {/* Status */}
            <div className="p-4 mx-4 mt-4 rounded-2xl glass-card">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-slate-400">System Status</span>
                <div className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${backendHealth.status === 'online' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
                  <span className="text-xs capitalize">{backendHealth.status}</span>
                </div>
              </div>
              <div className="space-y-1.5 text-[11px] text-slate-500">
                <div className="flex justify-between"><span>Mode</span><span className="text-slate-300 font-mono uppercase">{backendHealth.mode}</span></div>
                <div className="flex justify-between"><span>Agents</span><span className="text-slate-300">7 active</span></div>
                <div className="flex justify-between"><span>Version</span><span className="text-slate-300">{backendHealth.version || '2.0.0'}</span></div>
              </div>
            </div>

            {/* Repository Selector */}
            <div className="px-4 mt-4">
              <label className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2 block">Repository</label>
              <select
                value={repository}
                onChange={e => setRepository(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500/50"
              >
                {repositories.map(r => (
                  <option key={r} value={r} className="bg-slate-900">{r}</option>
                ))}
              </select>
            </div>

            {/* Presets */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 px-1 mb-2">Quick Prompts</p>
              {PRESETS.map((p, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(p)}
                  disabled={isGenerating}
                  className="w-full text-left p-3 rounded-xl text-xs border border-white/5 hover:border-indigo-500/30 bg-white/[0.02] hover:bg-indigo-500/5 transition-all text-slate-400 hover:text-slate-200 flex gap-2"
                >
                  <ChevronRight size={14} className="text-indigo-400 shrink-0 mt-0.5" />
                  <span className="line-clamp-2">{p}</span>
                </button>
              ))}
            </div>

            <div className="p-4 border-t border-white/5 flex items-center justify-center gap-1.5 text-[10px] text-slate-600">
              <Shield size={12} className="text-indigo-500" />
              ArchonAI Enterprise
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 px-6 flex items-center justify-between border-b border-white/5 glass-bar shrink-0">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button onClick={toggleSidebar} className="p-2 rounded-lg hover:bg-white/5 text-slate-400">
                <Menu size={18} />
              </button>
            )}
            <Sparkles className="text-indigo-400" size={20} />
            <h1 className="font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              ArchonAI — Agentic Code Intelligence
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => { setTraceOpen(!traceOpen); setCitationsOpen(false); }}
              className={`p-2 rounded-lg transition ${traceOpen ? 'bg-indigo-500/20 text-indigo-300' : 'hover:bg-white/5 text-slate-400'}`}
            >
              <Activity size={18} />
            </button>
            <button
              onClick={() => { setCitationsOpen(!citationsOpen); setTraceOpen(false); }}
              className={`p-2 rounded-lg transition ${citationsOpen ? 'bg-indigo-500/20 text-indigo-300' : 'hover:bg-white/5 text-slate-400'}`}
            >
              <FileText size={18} />
            </button>
            <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">v2.0</span>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map(msg => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onSelect={(m) => {
                if (m.citations?.length) { setActiveMessageId(m.id); setCitationsOpen(true); }
                if (m.agentTrace?.length) { setActiveMessageId(m.id); setTraceOpen(true); }
              }}
            />
          ))}
          {isGenerating && messages.at(-1)?.content === '' && <LoadingDots />}
        </main>

        <footer className="p-6 border-t border-white/5 glass-bar shrink-0">
          <div className="max-w-4xl mx-auto relative">
            <div className="absolute left-3 top-1/2 -translate-y-1/2 flex items-center gap-1 text-[10px] text-slate-500">
              <GitBranch size={12} />
              <span>{repository}</span>
            </div>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendMessage()}
              placeholder="Ask about code architecture, generate solutions, debug errors..."
              disabled={isGenerating}
              className="w-full bg-white/5 border border-white/10 hover:border-white/20 focus:border-indigo-500/40 rounded-2xl py-4 pl-24 pr-14 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500/20 transition-all"
            />
            <button
              onClick={() => sendMessage()}
              disabled={isGenerating || !input.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 rounded-xl transition shadow-lg shadow-indigo-500/20"
            >
              <Send size={16} />
            </button>
          </div>
        </footer>
      </div>

      {/* Right Panel */}
      <AnimatePresence>
        {(citationsOpen || traceOpen) && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 384, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="border-l border-white/5 bg-[#080c16]/90 backdrop-blur-xl shrink-0 flex flex-col overflow-hidden"
          >
            <div className="p-5 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                {traceOpen ? <Activity size={16} className="text-indigo-400" /> : <FileText size={16} className="text-indigo-400" />}
                <span className="font-bold text-sm">{traceOpen ? 'Agent Trace' : 'Citations'}</span>
              </div>
              <button onClick={() => { setCitationsOpen(false); setTraceOpen(false); }} className="text-slate-400 hover:text-white">
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {traceOpen
                ? <AgentTracePanel trace={activeMessage?.agentTrace} />
                : <CitationsPanel citations={activeMessage?.citations} />
              }
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}
