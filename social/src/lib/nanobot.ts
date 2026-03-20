const API_URL = process.env.NANOBOT_API_URL || "http://localhost:18790/v1";
const API_KEY = process.env.NANOBOT_API_KEY || "nanobot";

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export async function invokeAgent(
  model: string,
  messages: ChatMessage[],
): Promise<string> {
  const res = await fetch(`${API_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({ model, messages, stream: false }),
  });

  if (!res.ok) {
    throw new Error(`Nanobot API error: ${res.status} ${res.statusText}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content || "";
}

/**
 * Stream an agent invocation — calls nanobot SSE endpoint and fires
 * callbacks for progress hints and the final response.
 */
export async function invokeAgentStreaming(
  model: string,
  messages: ChatMessage[],
  onProgress: (text: string) => void,
): Promise<{ content: string; toolsUsed: string[] }> {
  const res = await fetch(`${API_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({ model, messages, stream: true }),
  });

  if (!res.ok) {
    throw new Error(`Nanobot API error: ${res.status} ${res.statusText}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let finalContent = "";
  const toolsUsed: string[] = [];
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines
    const lines = buffer.split("\n");
    buffer = lines.pop() || ""; // Keep incomplete line in buffer

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;

      try {
        const chunk = JSON.parse(data);
        const delta = chunk.choices?.[0]?.delta;
        if (!delta) continue;

        const content = delta.content || "";

        // Progress hints from nanobot use various emoji prefixes
        // ⏳ = tool calls, 🤖 = agent working, 🔀 = routing, ⚡ = delegation
        const isProgress = /^[\s\n]*(⏳|🤖|🔀|⚡|📋)/.test(content) ||
          content.includes("⏳") || content.includes("🤖") ||
          content.includes("🔀") || content.includes("⚡");

        if (isProgress) {
          const progressLines = content.split("\n").filter((l: string) => l.trim());
          for (const pl of progressLines) {
            onProgress(pl.trim());
          }
        } else if (content && !delta.role) {
          // Final content chunk
          finalContent += content;
        }

        // Extract tool calls if present
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            if (tc?.function?.name) {
              toolsUsed.push(tc.function.name);
            }
          }
        }
      } catch {
        // Skip unparseable chunks
      }
    }
  }

  return { content: finalContent, toolsUsed };
}

export async function getAgentModels(): Promise<
  { id: string; name: string }[]
> {
  const res = await fetch(`${API_URL}/models`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });
  if (!res.ok) return [];
  const data = await res.json();
  return (data.data || [])
    .filter((m: { id: string }) => m.id.startsWith("agent/"))
    .map((m: { id: string; name?: string }) => ({
      id: m.id,
      name: m.name || m.id.replace("agent/", ""),
    }));
}
