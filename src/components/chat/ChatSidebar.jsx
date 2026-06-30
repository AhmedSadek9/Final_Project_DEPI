import { useEffect, useState } from "react";

function ChatSidebar({ currentSession, setCurrentSession, refreshMessages, refresh }) {
  const [sessions, setSessions] = useState([]);

  // ==========================
  // Load Chats
  // ==========================

  const loadChats = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/chat-sessions"
      );

      const data = await response.json();

      setSessions(data.sessions);
    } catch (err) {
      console.log(err);
    }
  };

  // ==========================
  // Create New Chat
  // ==========================

  const newChat = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/new-chat",
        {
          method: "POST"
        }
      );
      const data = await response.json();

      await loadChats();

      setCurrentSession(data.session_id);
      
      // استدعاء الدالة لجلب رسائل الجلسة الجديدة فور إنشائها
      if (refreshMessages) {
        refreshMessages();
      }
    }
    catch (err) {
      console.log(err);
    }
  };

  // تعديل الـ useEffect لاستخدام refresh كـ dependency
  useEffect(() => {
    loadChats();
  }, [refresh]);

  return (
    <div className="w-72 bg-slate-900 text-white flex flex-col h-screen">
      <div className="p-5 border-b border-slate-700">
        <button
          onClick={newChat}
          className="w-full bg-blue-600 hover:bg-blue-700 rounded-xl py-3 font-bold"
        >
          + New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {
          sessions.map((chat) => (
            <div
              key={chat.id}
              onClick={() => {
                setCurrentSession(chat.id);
                // استدعاء الدالة عند اختيار جلسة موجودة
                if (refreshMessages) {
                  refreshMessages();
                }
              }}
              className={`cursor-pointer rounded-xl p-3 transition-all duration-200 ${
                currentSession === chat.id
                  ? "bg-blue-600"
                  : "hover:bg-slate-800"
              }`}
            >
              💬 {chat.title}
            </div>
          ))
        }
      </div>
    </div>
  );
}

export default ChatSidebar;