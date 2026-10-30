import type { ReactNode } from "react";
import { Redirect } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { getCurrentUser } from "@/lib/api-services";
import { SpinnerIcon } from "@/components/icons";

interface ProtectedRouteProps {
  children: ReactNode;
}

/** Route guard: redirects to /login when no valid JWT session exists. */
export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { data: user, isLoading } = useQuery({
    queryKey: ["me"],
    queryFn: getCurrentUser,
    retry: false,
    staleTime: 60_000,
  });

  // NOTE: never call navigate() during render (React warns "Cannot update
  // a component (Route) while rendering"). Declarative <Redirect> lets
  // wouter navigate as an effect instead; no new router introduced.
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <SpinnerIcon size={32} color="#acabaa" />
      </div>
    );
  }

  if (!user) {
    return <Redirect to="/login" />;
  }

  return <>{children}</>;
}
