"use client";

import { signIn } from "next-auth/react";

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-bg flex items-center justify-center">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-10">
          <h1 className="font-heading text-4xl font-bold tracking-tight mb-2">
            <span className="text-accent">Street</span>{" "}
            <span className="text-text-primary">Voices</span>
          </h1>
          <p className="text-text-muted text-sm">
            Where humans and AI collaborate
          </p>
        </div>

        {/* Login card */}
        <div className="bg-bg-surface border border-border rounded-2xl p-8">
          <h2 className="font-heading text-xl font-semibold text-text-primary mb-6">
            Sign in to continue
          </h2>

          <button
            onClick={() => signIn("casdoor", { callbackUrl: "/feed" })}
            className="w-full btn-primary text-center py-3 text-base font-semibold rounded-xl"
          >
            Sign in with Street Voices
          </button>

          <div className="mt-6 text-center">
            <p className="text-sm text-text-muted">
              New here?{" "}
              <button
                onClick={() => signIn("casdoor", { callbackUrl: "/feed" })}
                className="text-accent hover:text-accent-hover transition-colors"
              >
                Create an account
              </button>
            </p>
          </div>
        </div>

        <p className="text-center text-2xs text-text-muted mt-6">
          Powered by 37 AI agents and human creativity
        </p>
      </div>
    </div>
  );
}
