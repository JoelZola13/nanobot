"use client";

import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Heart, MessageCircle, Share2, Send, ImagePlus } from "lucide-react";
import type { FeedPostData } from "@/types";

interface FeedViewProps {
  initialPosts: FeedPostData[];
  userId: string;
}

export default function FeedView({ initialPosts, userId: _userId }: FeedViewProps) {
  const [posts, setPosts] = useState(initialPosts);
  const [newPost, setNewPost] = useState("");
  const [posting, setPosting] = useState(false);

  const handlePost = async () => {
    if (!newPost.trim() || posting) return;
    setPosting(true);
    try {
      const res = await fetch("/api/feed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: newPost }),
      });
      if (res.ok) {
        const post = await res.json();
        setPosts([post, ...posts]);
        setNewPost("");
      }
    } finally {
      setPosting(false);
    }
  };

  const handleLike = async (postId: string) => {
    const res = await fetch(`/api/feed/${postId}/like`, { method: "POST" });
    if (res.ok) {
      setPosts(
        posts.map((p) =>
          p.id === postId
            ? {
                ...p,
                userLiked: !p.userLiked,
                likeCount: p.userLiked ? p.likeCount - 1 : p.likeCount + 1,
              }
            : p,
        ),
      );
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto py-6 px-4 space-y-6">
        {/* Composer */}
        <div className="bg-bg-surface border border-border rounded-xl p-4">
          <textarea
            value={newPost}
            onChange={(e) => setNewPost(e.target.value)}
            placeholder="What's on your mind?"
            rows={3}
            className="w-full bg-transparent text-text-primary placeholder-text-muted text-sm resize-none focus:outline-none leading-relaxed"
          />
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-border">
            <button className="btn-ghost flex items-center gap-1.5 text-sm">
              <ImagePlus size={16} />
              <span>Media</span>
            </button>
            <button
              onClick={handlePost}
              disabled={!newPost.trim() || posting}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                newPost.trim()
                  ? "bg-accent text-white hover:bg-accent-hover"
                  : "bg-bg-elevated text-text-muted"
              }`}
            >
              <Send size={14} />
              <span>Post</span>
            </button>
          </div>
        </div>

        {/* Posts */}
        {posts.length === 0 && (
          <div className="text-center py-16">
            <div className="text-4xl mb-3">
              <span role="img" aria-label="sparkles">&#10024;</span>
            </div>
            <h3 className="font-heading text-lg font-semibold text-text-primary mb-1">
              The feed is empty
            </h3>
            <p className="text-sm text-text-muted">
              Be the first to share something with the community.
            </p>
          </div>
        )}

        {posts.map((post) => (
          <article
            key={post.id}
            className="bg-bg-surface border border-border rounded-xl overflow-hidden hover:border-border-subtle transition-colors"
          >
            <div className="p-4">
              {/* Author */}
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 avatar text-sm bg-accent-muted text-accent">
                  {post.author.displayName[0]?.toUpperCase()}
                </div>
                <div>
                  <div className="font-medium text-sm text-text-primary">
                    {post.author.displayName}
                  </div>
                  <div className="text-2xs text-text-muted">
                    @{post.author.username} ·{" "}
                    {formatDistanceToNow(new Date(post.createdAt), {
                      addSuffix: true,
                    })}
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="text-sm text-text-primary/90 whitespace-pre-wrap leading-relaxed mb-3">
                {post.content}
              </div>

              {/* Media */}
              {post.media.length > 0 && (
                <div className="rounded-lg overflow-hidden mb-3 border border-border">
                  {post.media.map((m, i) => (
                    <img
                      key={i}
                      src={m.url}
                      alt=""
                      className="w-full object-cover"
                    />
                  ))}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center gap-1 pt-2 border-t border-border">
                <button
                  onClick={() => handleLike(post.id)}
                  className={`btn-ghost flex items-center gap-1.5 text-sm ${
                    post.userLiked ? "text-accent" : ""
                  }`}
                >
                  <Heart
                    size={16}
                    fill={post.userLiked ? "currentColor" : "none"}
                  />
                  <span>{post.likeCount || ""}</span>
                </button>
                <button className="btn-ghost flex items-center gap-1.5 text-sm">
                  <MessageCircle size={16} />
                  <span>{post.commentCount || ""}</span>
                </button>
                <button className="btn-ghost flex items-center gap-1.5 text-sm ml-auto">
                  <Share2 size={16} />
                </button>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
