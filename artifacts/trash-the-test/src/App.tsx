import { Switch, Route, Router as WouterRouter, Redirect } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import LandingPage from "@/pages/LandingPage";
import Login from "@/pages/Login";
import Register from "@/pages/Register";
import Dashboard from "@/pages/Dashboard";
import ActiveStudy from "@/pages/ActiveStudy";
import StudyMode from "@/pages/StudyMode";
import Upload from "@/pages/Upload";
import AITutor from "@/pages/AITutor";
import Settings from "@/pages/Settings";
import Profile from "@/pages/Profile";
import { lazy, Suspense } from "react";
import Flashcards from "@/pages/Flashcards";
// PDF viewer is lazy-loaded: pdf.js is heavy and only needed on /documents/:id.
// Code-splitting keeps the main bundle at its pre-Phase-B size.
const DocumentViewer = lazy(() => import("@/pages/DocumentViewer"));
import Academic from "@/pages/Academic";
import Website from "@/pages/Website";
import PublicWebsite from "@/pages/PublicWebsite";
import ProtectedRoute from "@/components/ProtectedRoute";
import NotFound from "@/pages/not-found";

// Import API services to initialize configuration on app startup
import "@/lib/api-services";

const queryClient = new QueryClient();

function Router() {
  return (
    <Switch>
      <Route path="/" component={LandingPage} />
      <Route path="/login" component={Login} />
      <Route path="/register" component={Register} />
      {/* Domain-agnostic public share route: published websites only. */}
      <Route path="/website/:slug">
        {(params) => <PublicWebsite params={params} />}
      </Route>
      <Route path="/dashboard">
        {() => (
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/study">
        {() => (
          <ProtectedRoute>
            <ActiveStudy />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/study/:id">
        {(params) => (
          <ProtectedRoute>
            <StudyMode params={params} />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/deep/:id">
        {(params) => <Redirect to={`/study/${params?.id ?? ""}`} />}
      </Route>
      <Route path="/settings">
        {() => (
          <ProtectedRoute>
            <Settings />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/profile">
        {() => (
          <ProtectedRoute>
            <Profile />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/upload">
        {() => (
          <ProtectedRoute>
            <Upload />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/tutor">
        {() => (
          <ProtectedRoute>
            <AITutor />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/documents/:id">
        {(params) => (
          <ProtectedRoute>
            <Suspense fallback={<div className="p-10" style={{ color: "#acabaa" }}>Loading document viewer...</div>}>
              <DocumentViewer params={params} />
            </Suspense>
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/flashcards">
        {() => (
          <ProtectedRoute>
            <Flashcards />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/academic">
        {() => (
          <ProtectedRoute>
            <Academic />
          </ProtectedRoute>
        )}
      </Route>
      <Route path="/website">
        {() => (
          <ProtectedRoute>
            <Website />
          </ProtectedRoute>
        )}
      </Route>
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
        <div className="dark">
          <Router />
        </div>
      </WouterRouter>
    </QueryClientProvider>
  );
}

export default App;
