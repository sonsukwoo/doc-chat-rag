import { Navigate, Route, Routes } from "react-router-dom";

import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { ReviewPage } from "./pages/ReviewPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<DocumentsPage />} />
      <Route path="/threads/:threadId/chat" element={<ChatPage />} />
      <Route
        path="/threads/:threadId/documents/:documentId/review"
        element={<ReviewPage />}
      />
      <Route path="/documents/:documentId/review" element={<ReviewPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
