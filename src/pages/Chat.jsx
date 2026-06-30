import { useState, useRef, useEffect } from "react";
import ChatSidebar from "../components/chat/ChatSidebar";

function Chat() {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentSession, setCurrentSession] = useState(null);
  const [refreshChats, setRefreshChats] = useState(false);

  const loadMessages = async (sessionId) => {
    if (!sessionId) return;
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/chat-messages/${sessionId}`
      );
      const data = await response.json();
      setMessages(data.messages);
    } catch (err) {
      console.log(err);
    }
  };

  useEffect(() => {
    loadMessages(currentSession);
  }, [currentSession]);

  const messagesEndRef = useRef(null);

  // التعديل 3: الحصول على اسم المستخدم
  const user = JSON.parse(localStorage.getItem("user"));
  const username = user?.name || "Student";

  const [messages, setMessages] = useState([
    {
      sender: "ai",
      text: `👋 Hello ${username}!

I'm your AI Study Assistant.

Ask me anything about your uploaded documents.`,
      citations: [],
    },
  ]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages]);

  const handleSend = async () => {
    if (!message.trim() || loading) return;

    const userMessage = message.trim();

    if (
      currentSession &&
      messages.length === 1
    ) {
      await fetch(
        "http://127.0.0.1:8000/rename-chat",
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            session_id: currentSession,
            title: userMessage.substring(0, 35),
          }),
        }
      );
      setRefreshChats(prev => !prev);
    }

    setMessages((prev) => [
      ...prev,
      {
        sender: "user",
        text: userMessage,
      },
    ]);

    if (currentSession) {
      await fetch("http://127.0.0.1:8000/save-message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: currentSession,
          sender: "user",
          message: userMessage,
        }),
      });
    }

    setMessage("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
        }),
      });

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: data.answer,
          citations: data.citations || [],
        },
      ]);

      if (currentSession) {
        await fetch("http://127.0.0.1:8000/save-message", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            session_id: currentSession,
            sender: "ai",
            message: data.answer,
          }),
        });
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: "❌ Backend connection error.",
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex">
      <ChatSidebar
        currentSession={currentSession}
        setCurrentSession={setCurrentSession}
        refresh={refreshChats}
      />
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b shadow-sm px-8 py-5 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">🤖 AI Study Chat</h1>
            <p className="text-slate-500 mt-1">Ask questions about your uploaded documents</p>
          </div>
          <div className="bg-blue-600 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl">
            🤖
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-10 py-8 bg-slate-100">
          <div className="max-w-5xl mx-auto space-y-6">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-3xl rounded-3xl px-6 py-4 shadow-md ${
                    msg.sender === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-white text-slate-800"
                  }`}
                >
                  <p className="whitespace-pre-wrap leading-7">{msg.text}</p>
                  {msg.sender === "ai" && msg.citations?.length > 0 && (
                    <div className="mt-5 border-t pt-4">
                      <h3 className="font-bold mb-2">📚 Sources</h3>
                      {msg.citations.map((citation, i) => (
                        <div key={i} className="text-sm text-slate-500">
                          📄 {citation.source}
                          {" — Page "}
                          {citation.page}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white rounded-3xl px-6 py-4 shadow">🤖 Thinking...</div>
              </div>
            )}
            <div ref={messagesEndRef}></div>
          </div>
        </div>

        {/* Input */}
        <div className="bg-white border-t p-6">
          <div className="max-w-5xl mx-auto flex gap-4">
            <input
              type="text"
              value={message}
              disabled={loading}
              placeholder="Ask anything about your documents..."
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleSend();
                }
              }}
              className="flex-1 rounded-2xl border p-4 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleSend}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 rounded-2xl font-bold transition disabled:opacity-50"
            >
              {loading ? "Thinking..." : "➤"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Chat;