"""Replaceable in-process storage for short conversation histories."""

from threading import Lock


class InMemorySessionStore:
    """Thread-safe session history with bounded sessions and turns."""

    def __init__(self, max_sessions: int = 500, max_turns: int = 10):
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: dict[str, dict] = {}
        self._lock = Lock()

    def history_for(self, session_id: str, dataset_id: str) -> list[dict]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session["dataset_id"] != dataset_id:
                return []
            return list(session["history"])

    def append(self, session_id: str, dataset_id: str, question: str, answer: str) -> None:
        with self._lock:
            if session_id not in self._sessions and len(self._sessions) >= self.max_sessions:
                del self._sessions[next(iter(self._sessions))]
            session = self._sessions.get(session_id)
            if session is None or session["dataset_id"] != dataset_id:
                session = {"dataset_id": dataset_id, "history": []}
                self._sessions[session_id] = session
            session["history"].append({"question": question, "answer": answer})
            session["history"] = session["history"][-self.max_turns:]


