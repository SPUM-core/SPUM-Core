import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const API_BASE = 'http://localhost:8765';

export default function ChatPanel({ sym, stockName }: { sym: string; stockName: string }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-analyze when opened
  useEffect(() => {
    if (open && !analyzed) {
      sendMessage('请分析这支股票', true);
      setAnalyzed(true);
    }
  }, [open]);

  const sendMessage = async (text?: string, isAuto = false) => {
    const msg = text || input;
    if (!msg.trim() || loading) return;

    if (!isAuto) setInput('');

    const userMsg: Message = { role: 'user', content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sym,
          message: msg,
          history: messages.slice(-8).map((m) => ({ role: m.role, content: m.content })),
        }),
      });
      const data = await res.json();
      const assistantMsg: Message = { role: 'assistant', content: data.reply || '暂无回复' };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠ 连接后端失败: ${e.message}。请确保后端已启动。` },
      ]);
    }
    setLoading(false);
  };

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setOpen(!open)}
        className={`fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full shadow-lg transition-all duration-300 flex items-center justify-center text-xl ${
          open
            ? 'bg-zinc-800 rotate-45 scale-90'
            : 'bg-gradient-to-r from-amber-500 to-emerald-500 hover:scale-110'
        }`}
        title={open ? '关闭' : 'AI 分析'}
      >
        {open ? '✕' : '🤖'}
      </button>

      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-24 right-6 z-50 w-[380px] h-[520px] bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl flex flex-col overflow-hidden fade-in">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 bg-zinc-950/50">
            <div>
              <h3 className="text-sm font-semibold text-zinc-200">AI 分析</h3>
              <p className="text-[10px] text-zinc-500">{stockName}</p>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="w-7 h-7 rounded-lg bg-zinc-800 hover:bg-zinc-700 flex items-center justify-center text-xs text-zinc-400 transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 scrollbar-thin">
            {messages.length === 0 && (
              <div className="text-center text-zinc-600 text-xs mt-8">
                <div className="text-2xl mb-2">🤖</div>
                SPUM AI 分析已连接<br />
                自动分析中...
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[85%] rounded-xl px-3.5 py-2 text-sm leading-relaxed whitespace-pre-wrap ${
                    m.role === 'user'
                      ? 'bg-amber-600/20 text-zinc-200 border border-amber-700/20'
                      : 'bg-zinc-800/50 text-zinc-300 border border-zinc-700/30'
                  }`}
                >
                  {m.content}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-zinc-800/50 rounded-xl px-4 py-3 border border-zinc-700/30">
                  <div className="flex gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="px-4 py-3 border-t border-zinc-800">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="追问分析..."
                className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-3.5 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-amber-500/40 transition-all"
              />
              <button
                onClick={() => sendMessage()}
                disabled={loading || !input.trim()}
                className="px-4 py-2 rounded-xl bg-amber-600/20 border border-amber-700/30 text-amber-300 text-sm font-medium hover:bg-amber-600/30 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              >
                发送
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
