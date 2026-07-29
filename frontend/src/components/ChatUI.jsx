import { motion, AnimatePresence } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { useState } from 'react';

export function MessageBubble({ message, onSelect }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onClick={() => onSelect?.(message)}
      className={`flex gap-4 p-4 rounded-2xl cursor-pointer transition-all max-w-4xl mx-auto ${
        message.role === 'user'
          ? 'bg-white/5 border border-white/10'
          : 'glass-card hover:border-indigo-500/30'
      }`}
    >
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center text-xs font-bold shrink-0 ${
        message.role === 'user'
          ? 'bg-slate-800 text-slate-300'
          : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
      }`}>
        {message.role === 'user' ? 'U' : 'A'}
      </div>

      <div className="flex-1 min-w-0 space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
            {message.role === 'user' ? 'Query' : 'ArchonAI'}
          </span>
          {message.confidence != null && (
            <ConfidenceBadge score={message.confidence} />
          )}
          {message.citations?.length > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              {message.citations.length} sources
            </span>
          )}
        </div>
        <MarkdownContent content={message.content} onCopy={handleCopy} copied={copied} msgId={message.id} />
      </div>
    </motion.div>
  );
}

export function ConfidenceBadge({ score }) {
  const pct = Math.round(score * 100);
  const cls = pct >= 80
    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
    : pct >= 60
    ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
    : 'bg-rose-500/10 text-rose-400 border-rose-500/20';
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full border font-mono ${cls}`}>
      {pct}% confidence
    </span>
  );
}

function MarkdownContent({ content, onCopy, copied, msgId }) {
  if (!content) return null;
  const blocks = content.split(/(```[\s\S]*?```)/g);

  return blocks.map((block, i) => {
    if (block.startsWith('```')) {
      const match = block.match(/```(\w*)\n([\s\S]*?)```/);
      const lang = match?.[1] || 'python';
      const code = match?.[2] || block;
      const id = `${msgId}-${i}`;

      return (
        <div key={i} className="my-3 rounded-xl overflow-hidden border border-white/10">
          <div className="flex items-center justify-between px-3 py-2 bg-black/40 border-b border-white/5">
            <span className="text-[10px] font-mono text-slate-500 uppercase">{lang}</span>
            <button onClick={() => onCopy(code, id)} className="text-slate-400 hover:text-white text-xs flex items-center gap-1">
              {copied === id ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
              Copy
            </button>
          </div>
          <SyntaxHighlighter language={lang || 'python'} style={oneDark} customStyle={{ margin: 0, fontSize: '12px', background: '#0a0a0f' }}>
            {code}
          </SyntaxHighlighter>
        </div>
      );
    }

    return (
      <div key={i} className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
        {block.split('\n').map((line, li) => {
          if (line.startsWith('### ')) return <h3 key={li} className="text-base font-bold text-white mt-4 mb-1">{line.slice(4)}</h3>;
          if (line.startsWith('## ')) return <h2 key={li} className="text-lg font-bold text-white mt-4 mb-1">{line.slice(3)}</h2>;
          if (line.startsWith('- ')) return <li key={li} className="ml-4 list-disc text-slate-400">{line.slice(2)}</li>;
          return line.trim() ? <p key={li}>{line}</p> : <br key={li} />;
        })}
      </div>
    );
  });
}

export function AgentTracePanel({ trace }) {
  if (!trace?.length) return null;

  return (
    <div className="space-y-2">
      {trace.map((step, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05 }}
          className="flex items-start gap-3 p-3 rounded-xl bg-white/5 border border-white/10"
        >
          <div className="w-6 h-6 rounded-lg bg-indigo-500/20 flex items-center justify-center text-[10px] font-bold text-indigo-300">
            {i + 1}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-indigo-300 capitalize">{step.agent}</span>
              <span className="text-[10px] text-slate-500 font-mono">{Math.round(step.duration_ms || 0)}ms</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1 truncate">{step.output}</p>
          </div>
          <div className={`w-2 h-2 rounded-full mt-1.5 ${step.status === 'completed' ? 'bg-emerald-400' : step.status === 'failed' ? 'bg-rose-400' : 'bg-amber-400'}`} />
        </motion.div>
      ))}
    </div>
  );
}

export function CitationsPanel({ citations }) {
  if (!citations?.length) {
    return (
      <div className="text-center py-12 text-slate-500 text-sm">No citations for this response.</div>
    );
  }

  return (
    <AnimatePresence>
      {citations.map((cite, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.04 }}
          className="p-4 rounded-xl glass-card space-y-2"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-mono text-indigo-400 truncate">{cite.source}</span>
            <div className="flex gap-1">
              <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-white/5 text-slate-500">{cite.type}</span>
              {cite.score != null && (
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400">{cite.score.toFixed?.(2) ?? cite.score}</span>
              )}
            </div>
          </div>
          <pre className="text-[11px] font-mono text-slate-400 bg-black/30 p-3 rounded-lg overflow-x-auto max-h-40 overflow-y-auto">{cite.content}</pre>
        </motion.div>
      ))}
    </AnimatePresence>
  );
}

export function LoadingDots() {
  return (
    <div className="flex items-center gap-3 max-w-4xl mx-auto p-4">
      {[0, 1, 2].map(i => (
        <motion.span
          key={i}
          className="w-2 h-2 rounded-full bg-indigo-400"
          animate={{ y: [0, -6, 0] }}
          transition={{ repeat: Infinity, duration: 0.6, delay: i * 0.15 }}
        />
      ))}
      <span className="text-xs text-slate-500 font-mono">Agents executing pipeline...</span>
    </div>
  );
}

// ArchonAI UI components - architecture reference
