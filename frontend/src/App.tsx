import { Navigate, Route, Routes } from "react-router-dom";

import { DocumentsPage } from "./pages/DocumentsPage";
import { ReviewPage } from "./pages/ReviewPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<DocumentsPage />} />
      <Route path="/documents/:documentId/review" element={<ReviewPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
