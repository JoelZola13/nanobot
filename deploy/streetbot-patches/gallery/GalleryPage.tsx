import React, { useEffect, useState, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Filter,
  Search,
  Layers,
  Tag,
  DollarSign,
  Heart,
  ExternalLink,
  Share2,
  Copy,
  Eye,
  MessageCircle,
  Loader2,
  BarChart3,
  Grid,
  X,
  Maximize2,
  ArrowLeft,
  Upload,
  Image,
  ToggleLeft,
  ToggleRight,
  TrendingUp,
  FolderOpen,
  Inbox,
  Clock,
  Wallet,
  PenTool,
} from "lucide-react";
import { SB_API_BASE } from "~/components/streetbot/shared/apiConfig";
import { useGlassStyles } from "../shared/useGlassStyles";
import { GlassBackground } from "../shared/GlassBackground";
import { useAuthContext } from "~/hooks/AuthContext";
import { getOrCreateUserId } from "@/lib/userId";

// ── Types (inlined from gallery types) ──────────────────────────────────────

interface ArtistProfile {
  user_id: string;
  username: string;
  display_name: string;
  avatar_url: string | null;
  is_verified: boolean;
  is_featured: boolean;
  primary_roles: string[];
  city?: string;
  country?: string;
}

interface Artwork {
  id: string;
  artist_id: string | null;
  artist_name: string | null;
  artist_profile?: ArtistProfile | null;
  title: string;
  description: string | null;
  medium: string | null;
  style: string | null;
  year_created: number | null;
  image_url: string;
  thumbnail_url: string | null;
  full_resolution_url: string | null;
  is_featured: boolean;
  is_public: boolean;
  display_order: number | null;
  is_for_sale: boolean;
  price: number | null;
  currency: string;
  is_sold: boolean;
  sold_at: string | null;
  accepts_commissions: boolean;
  commission_info: string | null;
  tags: string[];
  collection_name: string | null;
  view_count: number;
  favorite_count: number;
  comment_count: number;
  share_count: number;
  is_nsfw: boolean;
  is_approved: boolean;
  additional_images: string[];
  created_at: string;
  updated_at: string;
}

interface GalleryUpload {
  id: string;
  user_id: string;
  title: string;
  image_url: string;
  created_at: string;
}

interface MediumOption {
  value: string;
  label: string;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const GALLERY_API_URL = `${SB_API_BASE}/gallery`;

// ── Sub-components ──────────────────────────────────────────────────────────

/** Simple <img> with onError fallback replacing SmartImage */
const GalleryImage = ({ src, alt }: { src: string; alt: string }) => {
  const [imgSrc, setImgSrc] = useState(src);

  useEffect(() => {
    setImgSrc(src);
  }, [src]);

  return (
    <img
      src={imgSrc}
      alt={alt}
      loading="lazy"
      decoding="async"
      onError={() => setImgSrc("/assets/gallery/cyberpunk.png")}
      style={{
        width: "100%",
        height: "100%",
        objectFit: "cover",
        display: "block",
      }}
    />
  );
};

/** Inline ProfileBadge replacement: simple avatar + name span */
const InlineProfileBadge = ({
  userId,
  username,
  displayName,
  avatarUrl,
  isVerified,
}: {
  userId: string;
  username: string;
  displayName: string;
  avatarUrl?: string;
  isVerified: boolean;
}) => {
  const navigate = useNavigate();
  const { colors } = useGlassStyles();
  return (
    <span
      onClick={(e) => {
        e.stopPropagation();
        navigate(`/gallery/artist/${userId}`);
      }}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        cursor: "pointer",
        fontSize: "0.85rem",
        color: colors.textSecondary,
      }}
    >
      {avatarUrl ? (
        <img
          src={avatarUrl}
          alt={displayName || username}
          style={{
            width: 20,
            height: 20,
            borderRadius: "50%",
            objectFit: "cover",
          }}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />
      ) : (
        <span
          style={{
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "rgba(255, 214, 0, 0.3)",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "0.65rem",
            fontWeight: "bold",
            color: colors.accent,
          }}
        >
          {(displayName || username || "?").charAt(0).toUpperCase()}
        </span>
      )}
      <span>{displayName || username}</span>
      {isVerified && (
        <span title="Verified" style={{ color: colors.accent, fontSize: "0.75rem" }}>
          &#10003;
        </span>
      )}
    </span>
  );
};

/** Simple fullscreen lightbox overlay */
const SimpleLightbox = ({
  artworks,
  initialIndex,
  onClose,
  onFavorite,
  favoriteIds,
}: {
  artworks: Artwork[];
  initialIndex: number;
  onClose: () => void;
  onFavorite: (id: string) => void;
  favoriteIds: Set<string>;
}) => {
  const [index, setIndex] = useState(initialIndex);
  const art = artworks[index];

  const goNext = useCallback(() => {
    setIndex((prev) => (prev + 1) % artworks.length);
  }, [artworks.length]);

  const goPrev = useCallback(() => {
    setIndex((prev) => (prev - 1 + artworks.length) % artworks.length);
  }, [artworks.length]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") goNext();
      if (e.key === "ArrowLeft") goPrev();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, goNext, goPrev]);

  if (!art) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(0, 0, 0, 0.92)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      {/* Close button */}
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: 20,
          right: 20,
          background: "rgba(255,255,255,0.15)",
          border: "1px solid rgba(255,255,255,0.2)",
          borderRadius: "50%",
          width: 44,
          height: 44,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
          cursor: "pointer",
          zIndex: 10,
        }}
        aria-label="Close"
      >
        <X size={22} />
      </button>

      {/* Previous button */}
      {artworks.length > 1 && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            goPrev();
          }}
          style={{
            position: "absolute",
            left: 20,
            top: "50%",
            transform: "translateY(-50%)",
            background: "rgba(255,255,255,0.12)",
            border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: "50%",
            width: 48,
            height: 48,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
            cursor: "pointer",
            fontSize: "1.5rem",
            zIndex: 10,
          }}
          aria-label="Previous"
        >
          &#8249;
        </button>
      )}

      {/* Next button */}
      {artworks.length > 1 && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            goNext();
          }}
          style={{
            position: "absolute",
            right: 20,
            top: "50%",
            transform: "translateY(-50%)",
            background: "rgba(255,255,255,0.12)",
            border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: "50%",
            width: 48,
            height: 48,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
            cursor: "pointer",
            fontSize: "1.5rem",
            zIndex: 10,
          }}
          aria-label="Next"
        >
          &#8250;
        </button>
      )}

      {/* Image */}
      <img
        src={art.full_resolution_url || art.image_url}
        alt={art.title}
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: "90vw",
          maxHeight: "80vh",
          objectFit: "contain",
          borderRadius: "12px",
          boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        }}
        onError={(e) => {
          (e.currentTarget as HTMLImageElement).src = art.image_url;
        }}
      />

      {/* Info bar */}
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          marginTop: 16,
          display: "flex",
          alignItems: "center",
          gap: 16,
          color: "#fff",
          fontSize: "0.95rem",
        }}
      >
        <span style={{ fontWeight: "bold", fontSize: "1.1rem" }}>{art.title}</span>
        {art.artist_name && (
          <span style={{ color: "rgba(255,255,255,0.6)" }}>by {art.artist_name}</span>
        )}
        <button
          onClick={() => onFavorite(art.id)}
          style={{
            background: "transparent",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 4,
            color: favoriteIds.has(art.id) ? "#FFD700" : "#fff",
          }}
          aria-label={favoriteIds.has(art.id) ? "Remove from favorites" : "Add to favorites"}
        >
          <Heart
            size={18}
            color={favoriteIds.has(art.id) ? "#FFD700" : "#fff"}
            fill={favoriteIds.has(art.id) ? "#FFD700" : "none"}
          />
        </button>
        <span style={{ color: "rgba(255,255,255,0.4)", fontSize: "0.8rem" }}>
          {index + 1} / {artworks.length}
        </span>
      </div>
    </div>
  );
};

// ── Main Component ──────────────────────────────────────────────────────────

// ── Artwork Detail View ──────────────────────────────────────────────────────

/* ─── Artwork Image Slideshow with slow fade ─── */
function ArtworkSlideshow({ artwork, glassCard }: { artwork: Artwork; glassCard: React.CSSProperties }) {
  const allImages = [
    artwork.full_resolution_url || artwork.image_url,
    ...(artwork.additional_images || []),
  ].filter(Boolean) as string[];
  const [currentIdx, setCurrentIdx] = useState(0);
  const [fade, setFade] = useState(1);
  const hasMultiple = allImages.length > 1;

  // Auto-advance every 5 seconds with slow crossfade
  useEffect(() => {
    if (!hasMultiple) return;
    const interval = setInterval(() => {
      setFade(0); // fade out
      setTimeout(() => {
        setCurrentIdx((prev) => (prev + 1) % allImages.length);
        setFade(1); // fade in
      }, 800); // 800ms fade-out, then switch and fade in
    }, 5000);
    return () => clearInterval(interval);
  }, [allImages.length, hasMultiple]);

  const goTo = (idx: number) => {
    if (idx === currentIdx) return;
    setFade(0);
    setTimeout(() => { setCurrentIdx(idx); setFade(1); }, 600);
  };

  return (
    <div style={{ flex: '1 1 500px', minWidth: 300 }}>
      <div style={{
        ...glassCard, borderRadius: 16, overflow: 'hidden', position: 'relative',
      }}>
        <img
          src={allImages[currentIdx]}
          alt={`${artwork.title} - image ${currentIdx + 1}`}
          style={{
            width: '100%', display: 'block', maxHeight: '70vh', objectFit: 'contain',
            background: '#000', opacity: fade,
            transition: 'opacity 0.8s ease-in-out',
          }}
        />
        {artwork.is_for_sale && (
          <div style={{
            position: 'absolute', top: 16, left: 16, padding: '6px 14px',
            background: '#FFD600', color: '#000', fontWeight: 800, fontSize: '0.75rem',
            borderRadius: 8, textTransform: 'uppercase',
          }}>
            {artwork.is_sold ? 'SOLD' : 'FOR SALE'}
          </div>
        )}
        {/* Navigation arrows */}
        {hasMultiple && (
          <>
            <button
              onClick={() => goTo((currentIdx - 1 + allImages.length) % allImages.length)}
              style={{
                position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                background: 'rgba(0,0,0,0.5)', border: 'none', color: '#fff', cursor: 'pointer',
                borderRadius: '50%', width: 40, height: 40, fontSize: '1.2rem',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                backdropFilter: 'blur(8px)',
              }}
            >‹</button>
            <button
              onClick={() => goTo((currentIdx + 1) % allImages.length)}
              style={{
                position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)',
                background: 'rgba(0,0,0,0.5)', border: 'none', color: '#fff', cursor: 'pointer',
                borderRadius: '50%', width: 40, height: 40, fontSize: '1.2rem',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                backdropFilter: 'blur(8px)',
              }}
            >›</button>
          </>
        )}
      </div>
      {/* Dot indicators */}
      {hasMultiple && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 12 }}>
          {allImages.map((_, i) => (
            <button
              key={i}
              onClick={() => goTo(i)}
              style={{
                width: i === currentIdx ? 24 : 10, height: 10, borderRadius: 5,
                background: i === currentIdx ? '#FFD600' : 'rgba(255,255,255,0.3)',
                border: 'none', cursor: 'pointer', transition: 'all 0.3s ease',
                padding: 0,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ArtworkDetailView({ artworkId, onBack, onSelectTag }: { artworkId: string; onBack: () => void; onSelectTag?: (tag: string) => void }) {
  const { colors, glassCard, glassSurface } = useGlassStyles();
  const { user: authUser } = useAuthContext();
  const [artwork, setArtwork] = useState<Artwork | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFav, setIsFav] = useState(false);
  const [comments, setComments] = useState<Array<{id: string; user_id: string; user_name: string; user_avatar: string; body: string; created_at: string; parent_id: string | null; edited?: boolean}>>([]);
  const [newComment, setNewComment] = useState('');
  const [postingComment, setPostingComment] = useState(false);
  const [replyTo, setReplyTo] = useState<string | null>(null);

  const loadComments = () => {
    fetch(`${GALLERY_API_URL}/comments?artwork_id=${encodeURIComponent(artworkId)}`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setComments(data); })
      .catch(() => {});
  };

  useEffect(() => { if (artworkId) loadComments(); }, [artworkId]);

  const handlePostComment = () => {
    if (!newComment.trim()) return;
    setPostingComment(true);
    const userId = getOrCreateUserId(authUser?.id);
    const userName = authUser?.name || authUser?.username || 'Anonymous';
    const userAvatar = authUser?.avatar || '';
    fetch(`${GALLERY_API_URL}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ artwork_id: artworkId, user_id: userId, user_name: userName, user_avatar: userAvatar, body: newComment.trim(), parent_id: replyTo }),
    })
      .then(r => r.json())
      .then(() => { setNewComment(''); setReplyTo(null); loadComments(); if (artwork) setArtwork({ ...artwork, comments: (artwork.comments || 0) + 1 }); })
      .catch(() => {})
      .finally(() => setPostingComment(false));
  };

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editBody, setEditBody] = useState('');

  const handleEditComment = (commentId: string) => {
    const userId = getOrCreateUserId(authUser?.id);
    fetch(`${GALLERY_API_URL}/comments`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: commentId, user_id: userId, body: editBody.trim() }),
    })
      .then(r => r.json())
      .then(() => { setEditingId(null); setEditBody(''); loadComments(); })
      .catch(() => {});
  };

  const handleDeleteComment = (commentId: string) => {
    if (!confirm('Delete this comment?')) return;
    const userId = getOrCreateUserId(authUser?.id);
    fetch(`${GALLERY_API_URL}/comments?id=${encodeURIComponent(commentId)}&user_id=${encodeURIComponent(userId)}`, { method: 'DELETE' })
      .then(r => r.json())
      .then(() => { loadComments(); if (artwork) setArtwork({ ...artwork, comments: Math.max((artwork.comments || 1) - 1, 0) }); })
      .catch(() => {});
  };

  useEffect(() => {
    setLoading(true);
    fetch(`${GALLERY_API_URL}/artworks/${encodeURIComponent(artworkId)}`)
      .then(r => { if (!r.ok) throw new Error('Not found'); return r.json(); })
      .then(data => { setArtwork(data); setError(null); })
      .catch(() => setError('Artwork not found'))
      .finally(() => setLoading(false));
  }, [artworkId]);

  // Check if favorited
  useEffect(() => {
    if (!artworkId) return;
    const userId = getOrCreateUserId(authUser?.id);
    fetch(`${GALLERY_API_URL}/users/${encodeURIComponent(userId)}/favorites`)
      .then(r => r.json())
      .then(data => {
        if (Array.isArray(data)) {
          const ids = data.map((f: { artwork_id?: string }) => f?.artwork_id ?? '').filter(Boolean);
          setIsFav(ids.includes(artworkId));
        }
      })
      .catch(() => {});
  }, [artworkId, authUser?.id]);

  const toggleFav = async () => {
    const userId = getOrCreateUserId(authUser?.id);
    try {
      await fetch(`${GALLERY_API_URL}/artworks/${encodeURIComponent(artworkId)}/favorites?user_id=${encodeURIComponent(userId)}`, {
        method: isFav ? 'DELETE' : 'POST',
      });
      setIsFav(!isFav);
    } catch {}
  };

  if (loading) {
    return (
      <div style={{ position: 'relative', minHeight: '100vh' }}>
        <GlassBackground />
        <div style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', height: '80vh' }}>
          <Loader2 size={32} style={{ animation: 'spin 1s linear infinite', color: colors.accent }} />
        </div>
      </div>
    );
  }

  if (error || !artwork) {
    return (
      <div style={{ position: 'relative', minHeight: '100vh' }}>
        <GlassBackground />
        <div style={{ position: 'relative', zIndex: 1, padding: '40px 20px', maxWidth: 800, margin: '0 auto', textAlign: 'center' }}>
          <button onClick={onBack} style={{ background: 'none', border: 'none', color: colors.accent, cursor: 'pointer', fontSize: '0.9rem', marginBottom: 20 }}>
            ← Back to Gallery
          </button>
          <h2 style={{ color: colors.text }}>{error || 'Artwork not found'}</h2>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: 'relative', minHeight: '100vh' }}>
      <GlassBackground />
      <div style={{ position: 'relative', zIndex: 1, padding: '24px 20px 60px', maxWidth: 1100, margin: '0 auto' }}>
        {/* Back button */}
        <button onClick={onBack} style={{
          background: 'none', border: 'none', color: colors.accent, cursor: 'pointer',
          fontSize: '0.85rem', fontWeight: 600, marginBottom: 20, display: 'flex', alignItems: 'center', gap: 6,
        }}>
          ← Back to Gallery
        </button>

        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
          {/* Left: Artwork image slideshow */}
          <ArtworkSlideshow artwork={artwork} glassCard={glassCard} />

          {/* Right: Details */}
          <div style={{ flex: '1 1 350px', minWidth: 280 }}>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800, color: colors.text, margin: '0 0 8px' }}>
              {artwork.title}
            </h1>
            <p style={{ fontSize: '0.95rem', color: colors.textMuted, margin: '0 0 20px' }}>
              by <span style={{ color: colors.accent, fontWeight: 600 }}>{artwork.artist_name || 'Unknown Artist'}</span>
              {artwork.year_created && <span> · {artwork.year_created}</span>}
            </p>

            {/* Price */}
            {artwork.is_for_sale && artwork.price != null && (
              <div style={{
                ...glassSurface, padding: '16px 20px', borderRadius: 12, marginBottom: 20,
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              }}>
                <div>
                  <div style={{ fontSize: '0.65rem', color: colors.textMuted, textTransform: 'uppercase', fontWeight: 700, marginBottom: 4 }}>Price</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#FFD600' }}>
                    {new Intl.NumberFormat("en-CA", { style: "currency", currency: artwork.currency || "CAD" }).format(Number(artwork.price))}
                  </div>
                </div>
                {artwork.is_sold ? (
                  <span style={{ padding: '8px 18px', borderRadius: 10, background: 'rgba(239,68,68,0.15)', color: '#ef4444', fontWeight: 700, fontSize: '0.8rem' }}>Sold</span>
                ) : (
                  <button style={{
                    padding: '10px 24px', borderRadius: 10, border: 'none',
                    background: '#FFD600', color: '#000', fontWeight: 700, fontSize: '0.85rem', cursor: 'pointer',
                  }}>
                    Purchase
                  </button>
                )}
              </div>
            )}

            {/* Description */}
            {artwork.description && (
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ fontSize: '0.75rem', fontWeight: 700, color: colors.textMuted, textTransform: 'uppercase', marginBottom: 8 }}>Description</h3>
                <p style={{ fontSize: '0.85rem', color: colors.text, lineHeight: 1.7 }}>{artwork.description}</p>
              </div>
            )}

            {/* Details grid */}
            <div style={{ ...glassSurface, padding: '16px 20px', borderRadius: 12, marginBottom: 20 }}>
              <h3 style={{ fontSize: '0.75rem', fontWeight: 700, color: colors.textMuted, textTransform: 'uppercase', marginBottom: 12 }}>Details</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 20px' }}>
                {artwork.medium && (
                  <div>
                    <div style={{ fontSize: '0.62rem', color: colors.textMuted, textTransform: 'uppercase' }}>Medium</div>
                    <div style={{ fontSize: '0.82rem', color: colors.text, fontWeight: 500 }}>{artwork.medium}</div>
                  </div>
                )}
                {artwork.style && (
                  <div>
                    <div style={{ fontSize: '0.62rem', color: colors.textMuted, textTransform: 'uppercase' }}>Style</div>
                    <div style={{ fontSize: '0.82rem', color: colors.text, fontWeight: 500 }}>{artwork.style}</div>
                  </div>
                )}
                {artwork.year_created && (
                  <div>
                    <div style={{ fontSize: '0.62rem', color: colors.textMuted, textTransform: 'uppercase' }}>Year</div>
                    <div style={{ fontSize: '0.82rem', color: colors.text, fontWeight: 500 }}>{artwork.year_created}</div>
                  </div>
                )}
                {artwork.collection_name && (
                  <div>
                    <div style={{ fontSize: '0.62rem', color: colors.textMuted, textTransform: 'uppercase' }}>Collection</div>
                    <div style={{ fontSize: '0.82rem', color: colors.text, fontWeight: 500 }}>{artwork.collection_name}</div>
                  </div>
                )}
              </div>
            </div>

            {/* Tags */}
            {artwork.tags && artwork.tags.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ fontSize: '0.75rem', fontWeight: 700, color: colors.textMuted, textTransform: 'uppercase', marginBottom: 8 }}>Tags</h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {artwork.tags.map(tag => (
                    <span key={tag} onClick={() => onSelectTag ? onSelectTag(tag) : null} style={{
                      padding: '4px 12px', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600,
                      background: 'rgba(255,214,0,0.1)', color: '#FFD600', border: '1px solid rgba(255,214,0,0.2)',
                      cursor: onSelectTag ? 'pointer' : 'default', transition: 'all 0.2s',
                    }}>{tag}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Stats & actions */}
            <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: colors.textMuted, fontSize: '0.8rem' }}>
                <Eye size={16} /> {artwork.view_count}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: colors.textMuted, fontSize: '0.8rem' }}>
                <Heart size={16} fill={isFav ? '#ef4444' : 'none'} color={isFav ? '#ef4444' : colors.textMuted} /> {artwork.favorite_count}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: colors.textMuted, fontSize: '0.8rem' }}>
                <MessageCircle size={16} /> {artwork.comment_count}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: colors.textMuted, fontSize: '0.8rem' }}>
                <Share2 size={16} /> {artwork.share_count}
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <button onClick={toggleFav} style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '10px 20px', borderRadius: 10,
                border: `1px solid ${isFav ? 'rgba(239,68,68,0.3)' : colors.border}`,
                background: isFav ? 'rgba(239,68,68,0.1)' : 'rgba(255,255,255,0.04)',
                color: isFav ? '#ef4444' : colors.text, cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600,
              }}>
                <Heart size={16} fill={isFav ? '#ef4444' : 'none'} /> {isFav ? 'Favorited' : 'Favorite'}
              </button>
              <button onClick={async () => {
                const shareUrl = window.location.href;
                const shareData = { title: artwork.title, text: `Check out "${artwork.title}" by ${artwork.artist_name} on Street Gallery`, url: shareUrl };
                try {
                  if (navigator.share && /Mobi|Android/i.test(navigator.userAgent)) {
                    await navigator.share(shareData);
                  } else {
                    await navigator.clipboard.writeText(shareUrl);
                    const btn = document.getElementById('share-link-btn');
                    if (btn) {
                      btn.textContent = '✓ Link Copied!';
                      btn.style.borderColor = 'rgba(34,197,94,0.5)';
                      btn.style.color = '#22c55e';
                      setTimeout(() => { btn.textContent = ''; btn.style.borderColor = ''; btn.style.color = ''; }, 2000);
                    }
                  }
                } catch { await navigator.clipboard.writeText(shareUrl); }
              }} id="share-link-btn-wrapper" style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '10px 20px', borderRadius: 10,
                border: `1px solid ${colors.border}`, background: 'rgba(255,255,255,0.04)',
                color: colors.text, cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600,
                transition: 'all 0.2s',
              }}>
                <Copy size={16} /> <span id="share-link-btn">Share Link</span>
              </button>
            </div>

            {/* Comments Section */}
            <div style={{ ...glassSurface, padding: '20px', borderRadius: 12, marginTop: 20 }}>
              <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: colors.accent, textTransform: 'uppercase', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <MessageCircle size={16} /> Comments ({comments.length})
              </h3>

              {/* Comment input */}
              <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
                <div style={{ width: 32, height: 32, borderRadius: '50%', background: 'linear-gradient(135deg, #eab308, #f59e0b)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', fontWeight: 700, color: '#000', flexShrink: 0 }}>
                  {(authUser?.name || authUser?.username || 'A').charAt(0).toUpperCase()}
                </div>
                <div style={{ flex: 1 }}>
                  {replyTo && (
                    <div style={{ fontSize: '0.7rem', color: colors.accent, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                      Replying to comment <button onClick={() => setReplyTo(null)} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '0.7rem' }}>✕ Cancel</button>
                    </div>
                  )}
                  <textarea
                    value={newComment}
                    onChange={e => setNewComment(e.target.value)}
                    placeholder="Add a comment..."
                    rows={2}
                    style={{ width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 8, padding: '8px 12px', color: colors.text, fontSize: '0.82rem', resize: 'vertical', outline: 'none', fontFamily: 'inherit' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 6 }}>
                    <button
                      onClick={handlePostComment}
                      disabled={postingComment || !newComment.trim()}
                      style={{ background: newComment.trim() ? colors.accent : 'rgba(255,255,255,0.1)', color: newComment.trim() ? '#000' : colors.textMuted, border: 'none', borderRadius: 20, padding: '6px 18px', fontSize: '0.75rem', fontWeight: 700, cursor: newComment.trim() ? 'pointer' : 'default', transition: 'all 0.2s' }}
                    >
                      {postingComment ? 'Posting...' : 'Post'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Comment list */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {comments.length === 0 && (
                  <p style={{ fontSize: '0.8rem', color: colors.textMuted, textAlign: 'center', padding: '10px 0' }}>No comments yet. Be the first!</p>
                )}
                {comments.filter(c => !c.parent_id).map(comment => (
                  <div key={comment.id}>
                    <div style={{ display: 'flex', gap: 10 }}>
                      <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'rgba(255,255,255,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.6rem', fontWeight: 700, color: colors.text, flexShrink: 0 }}>
                        {(comment.user_name || 'A').charAt(0).toUpperCase()}
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                          <span style={{ fontSize: '0.78rem', fontWeight: 600, color: colors.text }}>{comment.user_name || 'Anonymous'}</span>
                          <span style={{ fontSize: '0.62rem', color: colors.textMuted }}>{new Date(comment.created_at).toLocaleDateString()}</span>
                        </div>
                        {editingId === comment.id ? (
                          <div style={{ margin: '4px 0' }}>
                            <textarea value={editBody} onChange={e => setEditBody(e.target.value)} rows={2} style={{ width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 8, padding: '6px 10px', color: colors.text, fontSize: '0.8rem', resize: 'vertical', outline: 'none', fontFamily: 'inherit' }} />
                            <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                              <button onClick={() => handleEditComment(comment.id)} style={{ background: colors.accent, color: '#000', border: 'none', borderRadius: 14, padding: '4px 14px', fontSize: '0.68rem', fontWeight: 700, cursor: 'pointer' }}>Save</button>
                              <button onClick={() => { setEditingId(null); setEditBody(''); }} style={{ background: 'rgba(255,255,255,0.1)', color: colors.text, border: 'none', borderRadius: 14, padding: '4px 14px', fontSize: '0.68rem', cursor: 'pointer' }}>Cancel</button>
                            </div>
                          </div>
                        ) : (
                          <p style={{ fontSize: '0.8rem', color: colors.text, lineHeight: 1.5, margin: '4px 0' }}>
                            {comment.body}
                            {comment.edited && <span style={{ fontSize: '0.6rem', color: colors.textMuted, marginLeft: 6 }}>(edited)</span>}
                          </p>
                        )}
                        <div style={{ display: 'flex', gap: 12, marginTop: 2 }}>
                          <button onClick={() => { setReplyTo(comment.id); }} style={{ background: 'none', border: 'none', color: colors.textMuted, fontSize: '0.65rem', cursor: 'pointer', padding: 0 }}>Reply</button>
                          {comment.user_id === getOrCreateUserId(authUser?.id) && editingId !== comment.id && (
                            <>
                              <button onClick={() => { setEditingId(comment.id); setEditBody(comment.body); }} style={{ background: 'none', border: 'none', color: colors.accent, fontSize: '0.65rem', cursor: 'pointer', padding: 0 }}>Edit</button>
                              <button onClick={() => handleDeleteComment(comment.id)} style={{ background: 'none', border: 'none', color: '#ef4444', fontSize: '0.65rem', cursor: 'pointer', padding: 0 }}>Delete</button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    {/* Replies */}
                    {comments.filter(r => r.parent_id === comment.id).map(reply => (
                      <div key={reply.id} style={{ display: 'flex', gap: 10, marginLeft: 38, marginTop: 8 }}>
                        <div style={{ width: 24, height: 24, borderRadius: '50%', background: 'rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.55rem', fontWeight: 700, color: colors.text, flexShrink: 0 }}>
                          {(reply.user_name || 'A').charAt(0).toUpperCase()}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: colors.text }}>{reply.user_name || 'Anonymous'}</span>
                            <span style={{ fontSize: '0.6rem', color: colors.textMuted }}>{new Date(reply.created_at).toLocaleDateString()}</span>
                          </div>
                          {editingId === reply.id ? (
                            <div style={{ margin: '3px 0' }}>
                              <textarea value={editBody} onChange={e => setEditBody(e.target.value)} rows={2} style={{ width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 8, padding: '6px 10px', color: colors.text, fontSize: '0.78rem', resize: 'vertical', outline: 'none', fontFamily: 'inherit' }} />
                              <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                                <button onClick={() => handleEditComment(reply.id)} style={{ background: colors.accent, color: '#000', border: 'none', borderRadius: 14, padding: '4px 14px', fontSize: '0.68rem', fontWeight: 700, cursor: 'pointer' }}>Save</button>
                                <button onClick={() => { setEditingId(null); setEditBody(''); }} style={{ background: 'rgba(255,255,255,0.1)', color: colors.text, border: 'none', borderRadius: 14, padding: '4px 14px', fontSize: '0.68rem', cursor: 'pointer' }}>Cancel</button>
                              </div>
                            </div>
                          ) : (
                            <p style={{ fontSize: '0.78rem', color: colors.text, lineHeight: 1.5, margin: '3px 0' }}>
                              {reply.body}
                              {reply.edited && <span style={{ fontSize: '0.6rem', color: colors.textMuted, marginLeft: 6 }}>(edited)</span>}
                            </p>
                          )}
                          {reply.user_id === getOrCreateUserId(authUser?.id) && editingId !== reply.id && (
                            <div style={{ display: 'flex', gap: 12, marginTop: 2 }}>
                              <button onClick={() => { setEditingId(reply.id); setEditBody(reply.body); }} style={{ background: 'none', border: 'none', color: colors.accent, fontSize: '0.65rem', cursor: 'pointer', padding: 0 }}>Edit</button>
                              <button onClick={() => handleDeleteComment(reply.id)} style={{ background: 'none', border: 'none', color: '#ef4444', fontSize: '0.65rem', cursor: 'pointer', padding: 0 }}>Delete</button>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>

            {/* Commission info */}
            {artwork.accepts_commissions && artwork.commission_info && (
              <div style={{ ...glassSurface, padding: '16px 20px', borderRadius: 12, marginTop: 20 }}>
                <h3 style={{ fontSize: '0.75rem', fontWeight: 700, color: '#22c55e', textTransform: 'uppercase', marginBottom: 8 }}>
                  ✨ Commissions Open
                </h3>
                <p style={{ fontSize: '0.82rem', color: colors.text, lineHeight: 1.6, margin: 0 }}>{artwork.commission_info}</p>
              </div>
            )}
          </div>
        </div>

        {/* More by this Artist */}
        {artwork.artist_id && <MoreByArtist artistId={artwork.artist_id} artistName={artwork.artist_name || 'this artist'} currentArtworkId={artwork.id} glassCard={glassCard} colors={colors} />}

      </div>
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

/* ─── More by this Artist section ─── */
function MoreByArtist({ artistId, artistName, currentArtworkId, glassCard, colors }: {
  artistId: string; artistName: string; currentArtworkId: string;
  glassCard: React.CSSProperties; colors: Record<string, string>;
}) {
  const [artworks, setArtworks] = useState<Artwork[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetch(`${GALLERY_API_URL}/artworks`)
      .then(r => r.json())
      .then((data: Artwork[]) => {
        const byArtist = data.filter(a => a.artist_id === artistId && a.id !== currentArtworkId);
        setArtworks(byArtist);
      })
      .catch(() => {});
  }, [artistId, currentArtworkId]);

  if (artworks.length === 0) return null;

  return (
    <div style={{ marginTop: 40, paddingBottom: 40 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.3rem', fontWeight: 800, color: colors.text, margin: 0 }}>
          More by <span style={{ color: '#FFD600' }}>{artistName}</span>
        </h2>
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
        gap: 20,
      }}>
        {artworks.slice(0, 6).map(art => (
          <div
            key={art.id}
            onClick={() => navigate(`/gallery/artwork/${art.id}`)}
            style={{
              ...glassCard, borderRadius: 16, overflow: 'hidden', cursor: 'pointer',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-6px)'; e.currentTarget.style.boxShadow = '0 12px 30px rgba(0,0,0,0.4)'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
          >
            <div style={{ aspectRatio: '4/3', background: '#000', position: 'relative' }}>
              <img
                src={art.thumbnail_url || art.image_url}
                alt={art.title}
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
              />
              {art.is_for_sale && (
                <div style={{
                  position: 'absolute', top: 10, left: 10, padding: '4px 10px',
                  background: '#FFD600', color: '#000', fontWeight: 800, fontSize: '0.65rem',
                  borderRadius: 6, textTransform: 'uppercase',
                }}>
                  {art.is_sold ? 'SOLD' : 'FOR SALE'}
                </div>
              )}
            </div>
            <div style={{ padding: '12px 14px' }}>
              <div style={{ fontWeight: 700, color: colors.text, fontSize: '0.9rem', marginBottom: 4 }}>{art.title}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: 10, fontSize: '0.72rem', color: colors.textMuted }}>
                  <span>👁 {art.view_count}</span>
                  <span>❤ {art.favorite_count}</span>
                </div>
                {art.is_for_sale && art.price && (
                  <span style={{ color: '#FFD600', fontWeight: 700, fontSize: '0.82rem' }}>
                    CA${art.price.toLocaleString()}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
