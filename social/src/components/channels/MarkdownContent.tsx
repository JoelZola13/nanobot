"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

const components: Components = {
  p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-text-primary">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  del: ({ children }) => <del className="line-through text-text-muted">{children}</del>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
      {children}
    </a>
  ),
  code: ({ className, children, ...props }) => {
    const isBlock = className?.includes("language-");
    if (isBlock) {
      return (
        <code className={`block bg-bg-elevated rounded-lg px-4 py-3 my-2 text-xs font-mono overflow-x-auto border border-border text-text-primary ${className || ""}`} {...props}>
          {children}
        </code>
      );
    }
    return (
      <code className="bg-bg-elevated px-1.5 py-0.5 rounded text-xs font-mono text-accent border border-border" {...props}>
        {children}
      </code>
    );
  },
  pre: ({ children }) => <pre className="my-1">{children}</pre>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-accent pl-3 my-1 text-text-secondary italic">
      {children}
    </blockquote>
  ),
  ul: ({ children }) => <ul className="list-disc pl-5 my-1 space-y-0.5">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 my-1 space-y-0.5">{children}</ol>,
  li: ({ children }) => <li className="text-text-primary/90">{children}</li>,
  h1: ({ children }) => <h1 className="text-lg font-heading font-bold mt-2 mb-1">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-heading font-bold mt-2 mb-1">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-heading font-semibold mt-1 mb-0.5">{children}</h3>,
  hr: () => <hr className="border-border my-2" />,
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="text-xs border border-border rounded">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-bg-elevated">{children}</thead>,
  th: ({ children }) => <th className="px-3 py-1.5 text-left font-medium text-text-secondary border-b border-border">{children}</th>,
  td: ({ children }) => <td className="px-3 py-1.5 border-b border-border">{children}</td>,
};

export default function MarkdownContent({ content }: { content: string }) {
  // Simple messages without markdown syntax — render as plain text for performance
  if (!/[*_`~#\[\]>|\-\d+\.]/.test(content)) {
    return <span>{content}</span>;
  }

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {content}
    </ReactMarkdown>
  );
}
