"use client";

import { useEffect } from "react";
import { signIn } from "next-auth/react";

export default function LoginPage() {
  // Auto-redirect to Casdoor SSO — if user is already authenticated
  // via LibreChat (same Casdoor), they'll be signed in seamlessly.
  useEffect(() => {
    signIn("casdoor", { callbackUrl: "/dm" });
  }, []);

  return (
    <div className="min-h-screen bg-bg flex items-center justify-center">
      <div className="w-full max-w-sm">
        <div className="text-center mb-10">
          <h1 className="font-heading text-4xl font-bold tracking-tight mb-2">
            <span className="text-accent">Street</span>{" "}
            <span className="text-text-primary">Voices</span>
          </h1>
          <p className="text-text-muted text-sm">Signing in...</p>
        </div>
      </div>
    </div>
  );
}
