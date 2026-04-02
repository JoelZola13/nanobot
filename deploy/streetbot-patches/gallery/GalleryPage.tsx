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

// ── Style Collections ────────────────────────────────────────────────────────

const MEDIUM_COLLECTIONS = [
  {
    id: "photography",
    name: "Photography",
    description: "Captured moments — street scenes, portraits, landscapes, and raw visual storytelling.",
    cover: "https://picsum.photos/seed/col-photo/600/400",
    gradient: "linear-gradient(135deg, #232526 0%, #414345 100%)",
  },
  {
    id: "digital-art",
    name: "Digital Art",
    description: "Born on screen — digital illustrations, 3D renders, generative art, and pixel craft.",
    cover: "https://picsum.photos/seed/col-digital/600/400",
    gradient: "linear-gradient(135deg, #0ff 0%, #f0f 50%, #ff0 100%)",
  },
  {
    id: "oil-painting",
    name: "Oil Painting",
    description: "Rich textures and deep colour — the timeless tradition of oil on canvas.",
    cover: "https://picsum.photos/seed/col-oil/600/400",
    gradient: "linear-gradient(135deg, #8B4513 0%, #D2691E 100%)",
  },
  {
    id: "watercolor",
    name: "Watercolor",
    description: "Soft washes, translucent layers, and flowing pigment on paper.",
    cover: "https://picsum.photos/seed/col-watercolor/600/400",
    gradient: "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)",
  },
  {
    id: "mixed-media",
    name: "Mixed Media",
    description: "Collage, assemblage, and boundary-breaking combinations of materials and technique.",
    cover: "https://picsum.photos/seed/col-mixed/600/400",
    gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
  },
  {
    id: "street-art",
    name: "Street Art",
    description: "Murals, stencils, wheat-paste, and graffiti — art that lives on walls and in public spaces.",
    cover: "https://picsum.photos/seed/col-street/600/400",
    gradient: "linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%)",
  },
  {
    id: "acrylic",
    name: "Acrylic",
    description: "Versatile and vibrant — fast-drying acrylics from bold impasto to smooth glazes.",
    cover: "https://picsum.photos/seed/col-acrylic/600/400",
    gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  },
  {
    id: "charcoal",
    name: "Charcoal",
    description: "Dramatic contrasts and raw mark-making — the power of black and white.",
    cover: "https://picsum.photos/seed/col-charcoal/600/400",
    gradient: "linear-gradient(135deg, #434343 0%, #000000 100%)",
  },
  {
    id: "ink",
    name: "Ink",
    description: "Bold lines, delicate washes, and the expressive flow of ink on paper.",
    cover: "https://picsum.photos/seed/col-ink/600/400",
    gradient: "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
  },
];

function CollectionsView({ onBack, onSelectMedium }: { onBack: () => void; onSelectMedium: (medium: string) => void }) {
  const { colors, glassCard } = useGlassStyles();
  const { user: authUser } = useAuthContext();
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [savedIds, setSavedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Fetch artwork counts per medium
    fetch(`${GALLERY_API_URL}/artworks`)
      .then((r) => r.json())
      .then((artworks: Artwork[]) => {
        const mediumCounts: Record<string, number> = {};
        artworks.forEach((a) => {
          if (a.medium) {
            mediumCounts[a.medium] = (mediumCounts[a.medium] || 0) + 1;
          }
        });
        setCounts(mediumCounts);
      })
      .catch(() => {});
  }, []);

  // Load saved collections for this user
  useEffect(() => {
    if (!authUser?.id) return;
    fetch(`${GALLERY_API_URL}/collections/saved?user_id=${authUser.id}`)
      .then((r) => r.json())
      .then((rows: { collection_id: string }[]) => {
        setSavedIds(new Set(rows.map((r) => r.collection_id)));
      })
      .catch(() => {});
  }, [authUser?.id]);

  const toggleSave = async (collectionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!authUser?.id) return;
    const isSaved = savedIds.has(collectionId);
    const method = isSaved ? "DELETE" : "POST";
    try {
      await fetch(
        `${GALLERY_API_URL}/collections/save?user_id=${authUser.id}&collection_id=${collectionId}`,
        { method },
      );
      setSavedIds((prev) => {
        const next = new Set(prev);
        if (isSaved) next.delete(collectionId);
        else next.add(collectionId);
        return next;
      });
    } catch {}
  };

  return (
    <div style={{ background: colors.bg, minHeight: "100vh", position: "relative" }}>
      <GlassBackground />

      <div style={{ position: "relative", zIndex: 1, maxWidth: 1200, margin: "0 auto", padding: "32px 20px" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 32 }}>
          <button
            onClick={onBack}
            style={{
              background: "rgba(255,255,255,0.08)",
              border: `1px solid ${colors.border}`,
              borderRadius: "50%",
              width: 42,
              height: 42,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              color: colors.text,
              transition: "all 0.2s",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,214,0,0.15)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.08)"; }}
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: "1.8rem", fontWeight: 800, color: "#FFD700" }}>
              Collections
            </h1>
            <p style={{ margin: "4px 0 0", color: colors.textMuted, fontSize: "0.9rem" }}>
              Browse artwork by medium
            </p>
          </div>
        </div>

        {/* Collections Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: "24px",
          }}
        >
          {MEDIUM_COLLECTIONS.map((col) => {
            const count = counts[col.name] || 0;
            return (
              <div
                key={col.id}
                onClick={() => onSelectMedium(col.name)}
                style={{
                  ...glassCard,
                  overflow: "hidden",
                  cursor: "pointer",
                  transition: "all 0.3s ease",
                  position: "relative",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "translateY(-4px)";
                  e.currentTarget.style.boxShadow = "0 12px 40px rgba(255, 214, 0, 0.15)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.boxShadow = "";
                }}
              >
                {/* Cover image with gradient overlay */}
                <div
                  style={{
                    height: 180,
                    backgroundImage: `${col.gradient}, url(${col.cover})`,
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundBlendMode: "overlay",
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      bottom: 0,
                      left: 0,
                      right: 0,
                      height: "60%",
                      background: `linear-gradient(transparent, ${colors.bg})`,
                    }}
                  />
                  {/* Heart bookmark button */}
                  <button
                    onClick={(e) => toggleSave(col.id, e)}
                    style={{
                      position: "absolute",
                      top: 12,
                      left: 12,
                      background: savedIds.has(col.id) ? "rgba(255, 60, 80, 0.85)" : "rgba(0,0,0,0.5)",
                      backdropFilter: "blur(10px)",
                      border: "none",
                      borderRadius: "50%",
                      width: 38,
                      height: 38,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      cursor: "pointer",
                      transition: "all 0.25s ease",
                      boxShadow: savedIds.has(col.id) ? "0 0 12px rgba(255,60,80,0.4)" : "none",
                    }}
                    onMouseEnter={(e) => {
                      if (!savedIds.has(col.id)) {
                        e.currentTarget.style.background = "rgba(255, 60, 80, 0.6)";
                        e.currentTarget.style.transform = "scale(1.15)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!savedIds.has(col.id)) {
                        e.currentTarget.style.background = "rgba(0,0,0,0.5)";
                      }
                      e.currentTarget.style.transform = "scale(1)";
                    }}
                  >
                    <Heart
                      size={18}
                      fill={savedIds.has(col.id) ? "#fff" : "none"}
                      color="#fff"
                      strokeWidth={2}
                    />
                  </button>

                </div>

                {/* Info */}
                <div style={{ padding: "16px 20px 12px" }}>
                  <h3 style={{ margin: "0 0 8px", fontSize: "1.2rem", fontWeight: 700, color: colors.text }}>
                    {col.name}
                  </h3>
                  <p style={{ margin: 0, fontSize: "0.85rem", color: colors.textMuted, lineHeight: 1.5 }}>
                    {col.description}
                  </p>
                </div>

                {/* Footer bar */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 20px", borderTop: "1px solid rgba(255,255,255,0.08)", marginTop: "auto" }}>
                  <span style={{ fontSize: "0.78rem", color: colors.textMuted, fontWeight: 500 }}>
                    {count} {count === 1 ? "piece" : "pieces"}
                  </span>
                  <span style={{ fontSize: "0.78rem", color: colors.accent, fontWeight: 700, display: "flex", alignItems: "center", gap: 4 }}>
                    View →
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Saved Collections View ──────────────────────────────────────────────────

function SavedCollectionsView({ onBack, onSelectMedium }: { onBack: () => void; onSelectMedium: (medium: string) => void }) {
  const { colors, glassCard } = useGlassStyles();
  const { user: authUser } = useAuthContext();
  const [savedIds, setSavedIds] = useState<Set<string>>(new Set());
  const [counts, setCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    fetch(`${GALLERY_API_URL}/artworks`)
      .then((r) => r.json())
      .then((artworks: Artwork[]) => {
        const mediumCounts: Record<string, number> = {};
        artworks.forEach((a) => {
          if (a.medium) mediumCounts[a.medium] = (mediumCounts[a.medium] || 0) + 1;
        });
        setCounts(mediumCounts);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!authUser?.id) return;
    fetch(`${GALLERY_API_URL}/collections/saved?user_id=${authUser.id}`)
      .then((r) => r.json())
      .then((rows: { collection_id: string }[]) => {
        setSavedIds(new Set(rows.map((r) => r.collection_id)));
      })
      .catch(() => {});
  }, [authUser?.id]);

  const unsave = async (collectionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!authUser?.id) return;
    try {
      await fetch(
        `${GALLERY_API_URL}/collections/save?user_id=${authUser.id}&collection_id=${collectionId}`,
        { method: "DELETE" },
      );
      setSavedIds((prev) => {
        const next = new Set(prev);
        next.delete(collectionId);
        return next;
      });
    } catch {}
  };

  const savedCollections = MEDIUM_COLLECTIONS.filter((col) => savedIds.has(col.id));

  return (
    <div style={{ background: colors.bg, minHeight: "100vh", position: "relative" }}>
      <GlassBackground />
      <div style={{ position: "relative", zIndex: 1, maxWidth: 1200, margin: "0 auto", padding: "32px 20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 32 }}>
          <button
            onClick={onBack}
            style={{
              background: "rgba(255,255,255,0.08)",
              border: `1px solid ${colors.border}`,
              borderRadius: "50%",
              width: 42,
              height: 42,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              color: colors.text,
              transition: "all 0.2s",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,214,0,0.15)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.08)"; }}
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: "1.8rem", fontWeight: 800, color: "#FFD700" }}>
              <Heart size={24} style={{ marginRight: 8, verticalAlign: "middle" }} fill="#ff3c50" color="#ff3c50" />
              Saved Collections
            </h1>
            <p style={{ margin: "4px 0 0", color: colors.textMuted, fontSize: "0.9rem" }}>
              Your bookmarked medium collections
            </p>
          </div>
        </div>

        {savedCollections.length === 0 ? (
          <div
            style={{
              textAlign: "center",
              padding: "80px 20px",
              ...glassCard,
            }}
          >
            <Heart size={48} color={colors.textMuted} style={{ marginBottom: 16 }} />
            <h3 style={{ color: colors.text, margin: "0 0 8px" }}>No saved collections yet</h3>
            <p style={{ color: colors.textMuted, margin: 0 }}>
              Browse collections and click the heart to bookmark your favourites.
            </p>
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: "24px",
            }}
          >
            {savedCollections.map((col) => {
              const count = counts[col.name] || 0;
              return (
                <div
                  key={col.id}
                  onClick={() => onSelectMedium(col.name)}
                  style={{
                    ...glassCard,
                    overflow: "hidden",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                    position: "relative",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-4px)";
                    e.currentTarget.style.boxShadow = "0 12px 40px rgba(255, 60, 80, 0.15)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "";
                  }}
                >
                  <div
                    style={{
                      height: 180,
                      backgroundImage: `${col.gradient}, url(${col.cover})`,
                      backgroundSize: "cover",
                      backgroundPosition: "center",
                      backgroundBlendMode: "overlay",
                      position: "relative",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: "60%",
                        background: `linear-gradient(transparent, ${colors.bg})`,
                      }}
                    />
                    {/* Remove button */}
                    <button
                      onClick={(e) => unsave(col.id, e)}
                      title="Remove from saved"
                      style={{
                        position: "absolute",
                        top: 12,
                        left: 12,
                        background: "rgba(255, 60, 80, 0.85)",
                        border: "none",
                        borderRadius: "50%",
                        width: 38,
                        height: 38,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        cursor: "pointer",
                        transition: "all 0.25s ease",
                        boxShadow: "0 0 12px rgba(255,60,80,0.4)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = "rgba(200, 30, 50, 1)";
                        e.currentTarget.style.transform = "scale(1.15)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "rgba(255, 60, 80, 0.85)";
                        e.currentTarget.style.transform = "scale(1)";
                      }}
                    >
                      <Heart size={18} fill="#fff" color="#fff" strokeWidth={2} />
                    </button>
                  </div>
                  <div style={{ padding: "16px 20px 12px" }}>
                    <h3 style={{ margin: "0 0 8px", fontSize: "1.2rem", fontWeight: 700, color: colors.text }}>
                      {col.name}
                    </h3>
                    <p style={{ margin: 0, fontSize: "0.85rem", color: colors.textMuted, lineHeight: 1.5 }}>
                      {col.description}
                    </p>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 20px", borderTop: "1px solid rgba(255,255,255,0.08)", marginTop: "auto" }}>
                    <span style={{ fontSize: "0.78rem", color: colors.textMuted, fontWeight: 500 }}>
                      {count} {count === 1 ? "piece" : "pieces"}
                    </span>
                    <span style={{ fontSize: "0.78rem", color: colors.accent, fontWeight: 700, display: "flex", alignItems: "center", gap: 4 }}>
                      View →
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Medium & style options for the upload form ──────────────────────────────

const MEDIUM_OPTIONS = [
  "Oil Painting",
  "Digital Art",
  "Photography",
  "Watercolor",
  "Mixed Media",
  "Street Art",
  "Acrylic",
  "Charcoal",
  "Ink",
];

const STYLE_OPTIONS = [
  "Abstract",
  "Contemporary",
  "Realism",
  "Pop Art",
  "Minimalist",
  "Cyberpunk",
  "Impressionism",
  "Surrealism",
  "Expressionism",
  "Urban",
];

// Medium-specific sub-styles for the filter sidebar
const MEDIUM_STYLES: Record<string, string[]> = {
  Photography: [
    "Street Photography",
    "Film Photography",
    "Portrait Photography",
    "Landscape Photography",
    "Documentary",
    "Black & White",
    "Abstract Photography",
    "Fashion Photography",
  ],
  "Digital Art": [
    "3D Render",
    "Pixel Art",
    "Generative Art",
    "Digital Illustration",
    "Concept Art",
    "Cyberpunk",
    "Vaporwave",
    "AI-Assisted",
  ],
  "Oil Painting": [
    "Realism",
    "Impressionism",
    "Abstract",
    "Expressionism",
    "Surrealism",
    "Classical",
    "Plein Air",
    "Figurative",
  ],
  Watercolor: [
    "Botanical",
    "Landscape",
    "Loose / Wet-on-Wet",
    "Illustrative",
    "Abstract",
    "Portraiture",
    "Urban Sketching",
  ],
  "Mixed Media": [
    "Collage",
    "Assemblage",
    "Found Object",
    "Textile Art",
    "Contemporary",
    "Experimental",
    "Layered",
  ],
  "Street Art": [
    "Graffiti",
    "Mural",
    "Stencil",
    "Wheat-Paste",
    "Tagging",
    "Wildstyle",
    "Political",
    "Pop Art",
  ],
  Acrylic: [
    "Abstract",
    "Pop Art",
    "Impasto",
    "Minimalist",
    "Figurative",
    "Geometric",
    "Contemporary",
  ],
  Charcoal: [
    "Portrait",
    "Figure Drawing",
    "Landscape",
    "Hyperrealism",
    "Gestural",
    "Abstract",
    "Architectural",
  ],
  Ink: [
    "Line Art",
    "Sumi-e / Brush",
    "Cross-Hatching",
    "Calligraphy",
    "Illustrative",
    "Abstract",
    "Comic / Manga",
  ],
};

// ── Upload Art View ─────────────────────────────────────────────────────────

function UploadArtView({ onBack }: { onBack: () => void }) {
  const { colors, glassCard, glassSurface } = useGlassStyles();
  const { user: authUser } = useAuthContext();
  const navigate = useNavigate();

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [medium, setMedium] = useState("");
  const [style, setStyle] = useState("");
  const [tags, setTags] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isForSale, setIsForSale] = useState(false);
  const [price, setPrice] = useState("");
  const [currency, setCurrency] = useState("CAD");
  const [acceptsCommissions, setAcceptsCommissions] = useState(false);
  const [commissionInfo, setCommissionInfo] = useState("");
  const [yearCreated, setYearCreated] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !imageFile) {
      setError("Title and image are required.");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const userId = getOrCreateUserId(authUser?.id);
      const formData = new FormData();
      formData.append("title", title.trim());
      formData.append("artist_id", userId);
      if (description.trim()) formData.append("description", description.trim());
      if (medium) formData.append("medium", medium);
      if (style) formData.append("style", style);
      if (tags.trim()) formData.append("tags", tags.trim());
      if (yearCreated) formData.append("year_created", yearCreated);
      formData.append("is_for_sale", String(isForSale));
      if (isForSale && price) {
        formData.append("price", price);
        formData.append("currency", currency);
      }
      formData.append("accepts_commissions", String(acceptsCommissions));
      if (acceptsCommissions && commissionInfo.trim()) {
        formData.append("commission_info", commissionInfo.trim());
      }
      formData.append("image", imageFile);

      const resp = await fetch(`${GALLERY_API_URL}/artworks`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.error || `Upload failed (${resp.status})`);
      }

      setSuccess(true);
      setTimeout(() => navigate("/gallery"), 1500);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "12px 16px",
    background: "rgba(255,255,255,0.05)",
    border: `1px solid ${colors.border}`,
    borderRadius: "10px",
    color: colors.text,
    fontSize: "0.95rem",
    outline: "none",
    transition: "border-color 0.2s",
    boxSizing: "border-box",
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    marginBottom: "6px",
    fontWeight: 600,
    fontSize: "0.85rem",
    color: colors.textMuted,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  };

  return (
    <div style={{ background: colors.bg, minHeight: "100vh", position: "relative" }}>
      <GlassBackground />

      <div style={{ position: "relative", zIndex: 1, maxWidth: 720, margin: "0 auto", padding: "32px 20px" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 32 }}>
          <button
            onClick={onBack}
            style={{
              background: "rgba(255,255,255,0.08)",
              border: `1px solid ${colors.border}`,
              borderRadius: "50%",
              width: 42,
              height: 42,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              color: colors.text,
              transition: "all 0.2s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(255,214,0,0.15)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(255,255,255,0.08)";
            }}
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: "1.8rem", fontWeight: 800, color: colors.text }}>
              Submit Your Art
            </h1>
            <p style={{ margin: "4px 0 0", color: colors.textMuted, fontSize: "0.9rem" }}>
              Share your artwork with the Street Voices community
            </p>
          </div>
        </div>

        {/* Success message */}
        {success && (
          <div
            style={{
              ...glassCard,
              padding: "24px",
              marginBottom: 24,
              textAlign: "center",
              borderColor: "rgba(34,197,94,0.4)",
            }}
          >
            <p style={{ fontSize: "1.2rem", color: "#22c55e", fontWeight: 700, margin: 0 }}>
              Artwork submitted successfully!
            </p>
            <p style={{ color: colors.textMuted, marginTop: 8, marginBottom: 0 }}>
              Redirecting to gallery...
            </p>
          </div>
        )}

        {/* Form */}
        {!success && (
          <form onSubmit={handleSubmit}>
            <div style={{ ...glassCard, padding: "28px", marginBottom: 24 }}>
              <h2 style={{ margin: "0 0 24px", fontSize: "1.1rem", color: "#FFD700", fontWeight: 700 }}>
                Artwork Details
              </h2>

              {/* Title */}
              <div style={{ marginBottom: 20 }}>
                <label style={labelStyle}>
                  Title <span style={{ color: "#ef4444" }}>*</span>
                </label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Give your artwork a title"
                  required
                  style={inputStyle}
                  onFocus={(e) => { e.currentTarget.style.borderColor = "#FFD700"; }}
                  onBlur={(e) => { e.currentTarget.style.borderColor = colors.border; }}
                />
              </div>

              {/* Description */}
              <div style={{ marginBottom: 20 }}>
                <label style={labelStyle}>Description</label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Tell the story behind your piece..."
                  rows={4}
                  style={{ ...inputStyle, resize: "vertical" }}
                  onFocus={(e) => { e.currentTarget.style.borderColor = "#FFD700"; }}
                  onBlur={(e) => { e.currentTarget.style.borderColor = colors.border; }}
                />
              </div>

              {/* Medium & Style row */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                <div>
                  <label style={labelStyle}>Medium</label>
                  <select
                    value={medium}
                    onChange={(e) => { setMedium(e.target.value); setStyle(""); }}
                    style={{ ...inputStyle, cursor: "pointer" }}
                  >
                    <option value="">Select medium...</option>
                    {MEDIUM_OPTIONS.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>{medium ? `${medium} Style` : "Style"}</label>
                  <select
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                    style={{ ...inputStyle, cursor: "pointer" }}
                  >
                    <option value="">Select style...</option>
                    {(medium && MEDIUM_STYLES[medium] ? MEDIUM_STYLES[medium] : STYLE_OPTIONS).map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Tags & Year row */}
              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 20 }}>
                <div>
                  <label style={labelStyle}>Tags</label>
                  <input
                    type="text"
                    value={tags}
                    onChange={(e) => setTags(e.target.value)}
                    placeholder="abstract, colorful, portrait (comma-separated)"
                    style={inputStyle}
                    onFocus={(e) => { e.currentTarget.style.borderColor = "#FFD700"; }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = colors.border; }}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Year Created</label>
                  <input
                    type="number"
                    value={yearCreated}
                    onChange={(e) => setYearCreated(e.target.value)}
                    placeholder={String(new Date().getFullYear())}
                    min="1900"
                    max={new Date().getFullYear()}
                    style={inputStyle}
                    onFocus={(e) => { e.currentTarget.style.borderColor = "#FFD700"; }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = colors.border; }}
                  />
                </div>
              </div>
            </div>

            {/* Image Upload Card */}
            <div style={{ ...glassCard, padding: "28px", marginBottom: 24 }}>
              <h2 style={{ margin: "0 0 24px", fontSize: "1.1rem", color: "#FFD700", fontWeight: 700 }}>
                <Image size={18} style={{ marginRight: 8, verticalAlign: "middle" }} />
                Upload Image <span style={{ color: "#ef4444" }}>*</span>
              </h2>

              {imagePreview ? (
                <div style={{ position: "relative", marginBottom: 16 }}>
                  <img
                    src={imagePreview}
                    alt="Preview"
                    style={{
                      width: "100%",
                      maxHeight: 400,
                      objectFit: "contain",
                      borderRadius: 12,
                      border: `1px solid ${colors.border}`,
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => { setImageFile(null); setImagePreview(null); }}
                    style={{
                      position: "absolute",
                      top: 10,
                      right: 10,
                      background: "rgba(0,0,0,0.7)",
                      border: "none",
                      borderRadius: "50%",
                      width: 32,
                      height: 32,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      cursor: "pointer",
                      color: "#fff",
                    }}
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <label
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "48px 24px",
                    border: `2px dashed ${colors.border}`,
                    borderRadius: 12,
                    cursor: "pointer",
                    transition: "all 0.2s",
                    textAlign: "center",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "#FFD700";
                    e.currentTarget.style.background = "rgba(255,214,0,0.05)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = colors.border;
                    e.currentTarget.style.background = "transparent";
                  }}
                >
                  <Upload size={40} style={{ color: colors.textMuted, marginBottom: 12 }} />
                  <p style={{ color: colors.text, fontWeight: 600, margin: "0 0 4px" }}>
                    Click to upload your artwork
                  </p>
                  <p style={{ color: colors.textMuted, fontSize: "0.85rem", margin: 0 }}>
                    PNG, JPG, GIF, or WebP (max 10MB)
                  </p>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/gif,image/webp"
                    onChange={handleImageChange}
                    style={{ display: "none" }}
                  />
                </label>
              )}
            </div>

            {/* Pricing & Commissions Card */}
            <div style={{ ...glassCard, padding: "28px", marginBottom: 24 }}>
              <h2 style={{ margin: "0 0 24px", fontSize: "1.1rem", color: "#FFD700", fontWeight: 700 }}>
                Pricing & Commissions
              </h2>

              {/* For Sale toggle */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: isForSale ? 16 : 20 }}>
                <button
                  type="button"
                  onClick={() => setIsForSale(!isForSale)}
                  style={{ background: "none", border: "none", cursor: "pointer", padding: 0, display: "flex" }}
                >
                  {isForSale ? (
                    <ToggleRight size={28} style={{ color: "#FFD700" }} />
                  ) : (
                    <ToggleLeft size={28} style={{ color: colors.textMuted }} />
                  )}
                </button>
                <span style={{ color: colors.text, fontWeight: 600 }}>Available for sale</span>
              </div>

              {isForSale && (
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 20 }}>
                  <div>
                    <label style={labelStyle}>Price</label>
                    <input
                      type="number"
                      value={price}
                      onChange={(e) => setPrice(e.target.value)}
                      placeholder="0.00"
                      min="0"
                      step="0.01"
                      style={inputStyle}
                      onFocus={(e) => { e.currentTarget.style.borderColor = "#FFD700"; }}
                      onBlur={(e) => { e.currentTarget.style.borderColor = colors.border; }}
                    />
                  </div>
                  <div>
                    <label style={labelStyle}>Currency</label>
                    <select
                      value={currency}
                      onChange={(e) => setCurrency(e.target.value)}
                      style={{ ...inputStyle, cursor: "pointer" }}
                    >
                      <option value="CAD">CAD</option>
                      <option value="USD">USD</option>
                      <option value="EUR">EUR</option>
                      <option value="GBP">GBP</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Commissions toggle */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: acceptsCommissions ? 16 : 0 }}>
                <button
                  type="button"
                  onClick={() => setAcceptsCommissions(!acceptsCommissions)}
                  style={{ background: "none", border: "none", cursor: "pointer", padding: 0, display: "flex" }}
                >
                  {acceptsCommissions ? (
                    <ToggleRight size={28} style={{ color: "#FFD700" }} />
                  ) : (
                    <ToggleLeft size={28} style={{ color: colors.textMuted }} />
                  )}
                </button>
                <span style={{ color: colors.text, fontWeight: 600 }}>Open to commissions</span>
              </div>

              {acceptsCommissions && (
                <div style={{ marginTop: 4 }}>
                  <label style={labelStyle}>Commission Details</label>
                  <textarea
                    value={commissionInfo}
                    onChange={(e) => setCommissionInfo(e.target.value)}
                    placeholder="Describe your commission availability, pricing, turnaround time..."
                    rows={3}
                    style={{ ...inputStyle, resize: "vertical" }}
                    onFocus={(e) => { e.currentTarget.style.borderColor = "#FFD700"; }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = colors.border; }}
                  />
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <div
                style={{
                  background: "rgba(239,68,68,0.1)",
                  border: "1px solid rgba(239,68,68,0.3)",
                  borderRadius: 10,
                  padding: "12px 16px",
                  marginBottom: 16,
                  color: "#ef4444",
                  fontSize: "0.9rem",
                }}
              >
                {error}
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              disabled={submitting || !title.trim() || !imageFile}
              style={{
                width: "100%",
                padding: "16px",
                background: submitting || !title.trim() || !imageFile
                  ? "rgba(255,214,0,0.3)"
                  : "#FFD700",
                color: "#000",
                fontWeight: 800,
                fontSize: "1rem",
                textTransform: "uppercase",
                letterSpacing: "1px",
                border: "none",
                borderRadius: "999px",
                cursor: submitting || !title.trim() || !imageFile ? "not-allowed" : "pointer",
                transition: "all 0.2s",
                boxShadow: submitting || !title.trim() || !imageFile
                  ? "none"
                  : "0 4px 14px rgba(255, 214, 0, 0.4)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 10,
              }}
              onMouseEnter={(e) => {
                if (!submitting && title.trim() && imageFile) {
                  e.currentTarget.style.transform = "scale(1.02)";
                  e.currentTarget.style.boxShadow = "0 6px 20px rgba(255, 214, 0, 0.5)";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "scale(1)";
                e.currentTarget.style.boxShadow = submitting || !title.trim() || !imageFile
                  ? "none"
                  : "0 4px 14px rgba(255, 214, 0, 0.4)";
              }}
            >
              {submitting ? (
                <>
                  <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={18} />
                  Submit Artwork
                </>
              )}
            </button>
          </form>
        )}
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

/* ─── Dashboard View ─── */
/* ─── Portfolio Stats Detail Panel ─── */
function PortfolioStatsPanel({ onClose, glassCard }: { onClose: () => void; glassCard: React.CSSProperties }) {
  const [activeTab, setActiveTab] = useState<string>("overview");

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "artworks", label: "Top Artworks" },
    { id: "timeline", label: "Engagement Timeline" },
  ];

  const overviewStats = [
    { label: "Total Views", value: "3,847", change: "+12%", up: true },
    { label: "Total Saves", value: "284", change: "+8%", up: true },
    { label: "Profile Visits", value: "1,203", change: "+23%", up: true },
    { label: "Avg. Time on Page", value: "2m 34s", change: "-3%", up: false },
    { label: "Share Clicks", value: "156", change: "+41%", up: true },
  ];

  const topArtworks = [
    { title: "Urban Pulse", views: 567, saves: 42, comments: 8, medium: "Digital Art" },
    { title: "Morning Light", views: 312, saves: 28, comments: 5, medium: "Watercolor" },
    { title: "Concrete Dreams", views: 234, saves: 18, comments: 3, medium: "Mixed Media" },
    { title: "Neon Reflections", views: 189, saves: 15, comments: 2, medium: "Digital Art" },
    { title: "Golden Hour", views: 145, saves: 12, comments: 4, medium: "Photography" },
  ];

  const timeline = [
    { period: "This Week", views: 892, saves: 67, visits: 234 },
    { period: "Last Week", views: 743, saves: 51, visits: 198 },
    { period: "2 Weeks Ago", views: 681, saves: 48, visits: 187 },
    { period: "3 Weeks Ago", views: 612, saves: 43, visits: 156 },
    { period: "4 Weeks Ago", views: 519, saves: 38, visits: 128 },
    { period: "Last Month", views: 1840, saves: 127, visits: 489 },
  ];

  return (
    <div style={{ ...glassCard, padding: 0, borderRadius: 16, overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "20px 24px", borderBottom: "1px solid rgba(255,255,255,0.08)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <TrendingUp size={24} color="#FFD600" />
          <h2 style={{ color: "#fff", fontSize: 22, fontWeight: 700, margin: 0 }}>Portfolio Stats</h2>
        </div>
        <button onClick={onClose} style={{ background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, color: "#fff", cursor: "pointer", padding: "6px 8px" }}>
          <X size={16} />
        </button>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              flex: 1,
              padding: "14px 16px",
              background: activeTab === tab.id ? "rgba(255,214,0,0.08)" : "transparent",
              border: "none",
              borderBottom: activeTab === tab.id ? "2px solid #FFD600" : "2px solid transparent",
              color: activeTab === tab.id ? "#FFD600" : "rgba(255,255,255,0.5)",
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 600,
              transition: "all 0.2s",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: 24 }}>
        {activeTab === "overview" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 16 }}>
            {overviewStats.map((stat) => (
              <div
                key={stat.label}
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 12,
                  padding: 20,
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,214,0,0.06)"; (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,214,0,0.2)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)"; (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.08)"; }}
              >
                <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 12, marginBottom: 8, fontWeight: 500 }}>{stat.label}</div>
                <div style={{ color: "#fff", fontSize: 28, fontWeight: 800 }}>{stat.value}</div>
                <div style={{ color: stat.up ? "#4ade80" : "#f87171", fontSize: 13, marginTop: 6, fontWeight: 600 }}>
                  {stat.change} <span style={{ color: "rgba(255,255,255,0.3)", fontWeight: 400 }}>vs last month</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === "artworks" && (
          <div>
            {topArtworks.map((art, i) => (
              <div
                key={art.title}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "16px 12px",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  cursor: "pointer",
                  borderRadius: 8,
                  transition: "background 0.2s",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                  <span style={{ color: "#FFD600", fontSize: 18, fontWeight: 800, width: 28 }}>#{i + 1}</span>
                  <div>
                    <div style={{ color: "#fff", fontSize: 15, fontWeight: 600 }}>{art.title}</div>
                    <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>{art.medium}</div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ color: "#fff", fontSize: 16, fontWeight: 700 }}>{art.views}</div>
                    <div style={{ color: "rgba(255,255,255,0.35)", fontSize: 11 }}>views</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ color: "#FFD600", fontSize: 16, fontWeight: 700 }}>{art.saves}</div>
                    <div style={{ color: "rgba(255,255,255,0.35)", fontSize: 11 }}>saves</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ color: "rgba(255,255,255,0.7)", fontSize: 16, fontWeight: 700 }}>{art.comments}</div>
                    <div style={{ color: "rgba(255,255,255,0.35)", fontSize: 11 }}>comments</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === "timeline" && (
          <div>
            {timeline.map((row) => (
              <div
                key={row.period}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "14px 12px",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  cursor: "pointer",
                  borderRadius: 8,
                  transition: "background 0.2s",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <div style={{ color: "#fff", fontSize: 14, fontWeight: 600, minWidth: 120 }}>{row.period}</div>
                <div style={{ display: "flex", gap: 24 }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ color: "#fff", fontSize: 15, fontWeight: 700 }}>{row.views.toLocaleString()}</div>
                    <div style={{ color: "rgba(255,255,255,0.35)", fontSize: 11 }}>views</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ color: "#FFD600", fontSize: 15, fontWeight: 700 }}>{row.saves}</div>
                    <div style={{ color: "rgba(255,255,255,0.35)", fontSize: 11 }}>saves</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ color: "rgba(255,255,255,0.7)", fontSize: 15, fontWeight: 700 }}>{row.visits}</div>
                    <div style={{ color: "rgba(255,255,255,0.35)", fontSize: 11 }}>visits</div>
                  </div>
                </div>
                {/* Mini bar chart */}
                <div style={{ width: 120, height: 8, background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min(100, (row.views / 900) * 100)}%`, height: "100%", background: "linear-gradient(90deg, #FFD600, #ff9800)", borderRadius: 4 }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Dashboard View ─── */
function DashboardView({ onBack }: { onBack: () => void }) {
  const { colors, glassCard } = useGlassStyles();
  const [expandedSection, setExpandedSection] = useState<string | null>(null);
  const navigate = useNavigate();

  const sections = [
    {
      id: "portfolio-stats",
      icon: <TrendingUp size={28} color="#FFD600" />,
      title: "Portfolio Stats",
      description: "Views, saves, profile visits, and which artworks get the most engagement.",
      items: [
        { label: "Total Views", value: "3,847" },
        { label: "Total Saves", value: "284" },
        { label: "Profile Visits", value: "1,203" },
        { label: "Top Artwork", value: "Urban Pulse" },
        { label: "Engagement Trend", value: "+18% this month" },
      ],
      expandable: true,
    },
    {
      id: "artwork-manager",
      icon: <PenTool size={28} color="#FFD600" />,
      title: "Artwork Manager",
      description: "Upload, edit, tag by collection/style, and set availability.",
      items: [
        { label: "Upload new artwork", value: "→", action: () => navigate("/gallery/upload") },
        { label: "Edit existing pieces", value: "15 artworks" },
        { label: "Tag by collection & style", value: "8 collections" },
        { label: "For Sale", value: "6 pieces" },
        { label: "Commission", value: "3 open" },
        { label: "Display Only", value: "6 pieces" },
      ],
    },
    {
      id: "collection-assignment",
      icon: <FolderOpen size={28} color="#FFD600" />,
      title: "Collection Assignment",
      description: "Organise your artworks into the style collections.",
      items: [
        { label: "Photography", value: "2 pieces", action: () => navigate("/gallery/collections") },
        { label: "Digital Art", value: "4 pieces", action: () => navigate("/gallery/collections") },
        { label: "Oil Painting", value: "1 piece", action: () => navigate("/gallery/collections") },
        { label: "Watercolor", value: "2 pieces", action: () => navigate("/gallery/collections") },
        { label: "Mixed Media", value: "2 pieces", action: () => navigate("/gallery/collections") },
        { label: "Street Art", value: "1 piece", action: () => navigate("/gallery/collections") },
      ],
    },
    {
      id: "inquiry-inbox",
      icon: <Inbox size={28} color="#FFD600" />,
      title: "Inquiry Inbox",
      description: "Messages from buyers and collaborators routed through the platform.",
      items: [
        { label: "New inquiries", value: "3 unread" },
        { label: "From: Sarah M.", value: "Interested in 'Urban Pulse' — CA$450" },
        { label: "From: David K.", value: "Commission request — portrait" },
        { label: "From: Gallery XYZ", value: "Exhibition opportunity" },
        { label: "Archived", value: "12 conversations" },
      ],
    },
    {
      id: "commission-tracker",
      icon: <Clock size={28} color="#FFD600" />,
      title: "Commission Status Tracker",
      description: "Track open requests, in-progress work, and delivered commissions.",
      items: [
        { label: "Open Requests", value: "2" },
        { label: "In Progress", value: "1" },
        { label: "Delivered", value: "4" },
        { label: "Total Revenue", value: "CA$3,200" },
        { label: "Avg. Completion", value: "12 days" },
      ],
    },
    {
      id: "earnings-summary",
      icon: <Wallet size={28} color="#FFD600" />,
      title: "Earnings Summary",
      description: "Revenue by artwork and period — ready when transactions go live.",
      items: [
        { label: "Total Earnings", value: "CA$5,450" },
        { label: "This Month", value: "CA$1,200" },
        { label: "Last Month", value: "CA$890" },
        { label: "Top Seller", value: "Concrete Dreams — CA$1,200" },
        { label: "Pending Payouts", value: "CA$450" },
      ],
    },
  ];

  return (
    <div style={{ background: colors.bg, minHeight: "100vh", position: "relative" }}>
      <GlassBackground />
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "40px 24px", position: "relative", zIndex: 1 }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 12 }}>
          <button
            onClick={onBack}
            style={{
              background: "rgba(255,255,255,0.08)",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: "50%",
              width: 42,
              height: 42,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              color: "#fff",
            }}
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 style={{ color: "#FFD600", fontSize: 32, fontWeight: 800, margin: 0 }}>Artist Dashboard</h1>
            <p style={{ color: "rgba(255,255,255,0.5)", fontSize: 14, margin: "4px 0 0" }}>
              Manage your gallery, track engagement, and handle inquiries
            </p>
          </div>
        </div>

        {/* Expanded Portfolio Stats Panel */}
        {expandedSection === "portfolio-stats" && (
          <div style={{ marginTop: 24, marginBottom: 24 }}>
            <PortfolioStatsPanel onClose={() => setExpandedSection(null)} glassCard={glassCard} />
          </div>
        )}

        {/* Dashboard Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
            gap: 24,
            marginTop: 32,
          }}
        >
          {sections.map((section) => (
            <div
              key={section.title}
              onClick={() => {
                if (section.expandable) {
                  setExpandedSection(expandedSection === section.id ? null : section.id);
                }
              }}
              style={{
                ...glassCard,
                padding: 28,
                borderRadius: 16,
                cursor: "pointer",
                transition: "transform 0.2s, box-shadow 0.2s",
                border: expandedSection === section.id ? "1px solid rgba(255,214,0,0.3)" : (glassCard as any).border,
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.transform = "translateY(-4px)";
                (e.currentTarget as HTMLElement).style.boxShadow = "0 12px 40px rgba(255,214,0,0.1)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                (e.currentTarget as HTMLElement).style.boxShadow = "none";
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 12 }}>
                <div
                  style={{
                    background: "rgba(255,214,0,0.1)",
                    borderRadius: 12,
                    width: 48,
                    height: 48,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {section.icon}
                </div>
                <h2 style={{ color: "#fff", fontSize: 20, fontWeight: 700, margin: 0 }}>{section.title}</h2>
                {section.expandable && (
                  <span style={{ marginLeft: "auto", color: "#FFD600", fontSize: 12, fontWeight: 600 }}>
                    {expandedSection === section.id ? "▲ Collapse" : "▼ Expand"}
                  </span>
                )}
              </div>
              <p style={{ color: "rgba(255,255,255,0.55)", fontSize: 14, lineHeight: 1.5, margin: "0 0 16px" }}>
                {section.description}
              </p>
              <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
                {section.items.map((item) => (
                  <li
                    key={item.label}
                    onClick={(e) => {
                      e.stopPropagation();
                      if ((item as any).action) (item as any).action();
                    }}
                    style={{
                      color: "rgba(255,255,255,0.7)",
                      fontSize: 13,
                      padding: "10px 8px",
                      borderBottom: "1px solid rgba(255,255,255,0.05)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 8,
                      cursor: (item as any).action ? "pointer" : "default",
                      borderRadius: 6,
                      transition: "background 0.15s",
                    }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)"; }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                  >
                    <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ color: "#FFD600", fontSize: 8 }}>●</span>
                      {item.label}
                    </span>
                    <span style={{ color: "#FFD600", fontSize: 13, fontWeight: 600 }}>{item.value}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function GalleryPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isDark, colors, glassCard, glassSurface, glassTag, cardHoverHandlers, buttonHoverHandlers } = useGlassStyles();
  const { user: authUser } = useAuthContext();
  const [filterOpen, setFilterOpen] = useState(false);
  const [favoriteIds, setFavoriteIds] = useState<Set<string>>(new Set());

  // Check if we're on an artwork detail page
  const artworkMatch = location.pathname.match(/\/gallery\/artwork\/(.+)/);

  // Artworks state
  const [artworks, setArtworks] = useState<Artwork[]>([]);
  const [artworksLoading, setArtworksLoading] = useState(true);
  const [artworksError, setArtworksError] = useState<string | null>(null);

  // Filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMedium, setSelectedMedium] = useState<string>("");
  const [selectedStyle, setSelectedStyle] = useState<string>("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [showForSale, setShowForSale] = useState(false);
  const [priceSort, setPriceSort] = useState<"" | "low-high" | "high-low">("");
  const [showHotOnly, setShowHotOnly] = useState(false);
  const [mediums, setMediums] = useState<MediumOption[]>([]);
  const [popularTags, setPopularTags] = useState<{ tag: string; count: number }[]>([]);

  // Lightbox state
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  // User uploads state
  const [uploads, setUploads] = useState<GalleryUpload[]>([]);
  const [uploadsLoading, setUploadsLoading] = useState(false);
  const [uploadsError, setUploadsError] = useState<string | null>(null);
  const [uploadsMessage, setUploadsMessage] = useState<string | null>(null);

  // Artist profiles for enrichment (keyed by user_id)
  const [artistProfiles, setArtistProfiles] = useState<Record<string, ArtistProfile>>({});

  // Load available mediums
  const loadMediums = useCallback(async () => {
    try {
      const resp = await fetch(`${GALLERY_API_URL}/artworks/mediums`);
      if (resp.ok) {
        const data = await resp.json();
        setMediums(data);
      }
    } catch {
      // Ignore - mediums are optional
    }
  }, []);

  // Load popular tags
  const loadPopularTags = useCallback(async () => {
    try {
      const resp = await fetch(`${GALLERY_API_URL}/tags?limit=20`);
      if (resp.ok) {
        const data = await resp.json();
        setPopularTags(Array.isArray(data) ? data : []);
      }
    } catch {
      // Ignore - tags are optional
    }
  }, []);

  // Load artworks from API
  const loadArtworks = useCallback(async () => {
    setArtworksLoading(true);
    setArtworksError(null);
    try {
      const params = new URLSearchParams();
      if (searchQuery) params.set("search", searchQuery);
      if (selectedMedium) params.set("medium", selectedMedium);
      if (selectedStyle) params.set("style", selectedStyle);
      if (selectedTags.length > 0) params.set("tags", selectedTags.join(","));
      if (showForSale) params.set("is_for_sale", "true");

      const url = `${GALLERY_API_URL}/artworks${params.toString() ? `?${params}` : ""}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Failed to load artworks (${resp.status})`);
      const rawData = await resp.json();
      let data: Artwork[] = Array.isArray(rawData) ? rawData : [];

      // Hot products filter (high engagement: views + favorites)
      if (showHotOnly) {
        data = data.filter(a => (a.view_count || 0) + (a.favorites || 0) >= 10);
      }

      // Price sort
      if (priceSort === "low-high") {
        data.sort((a, b) => (a.price || 0) - (b.price || 0));
      } else if (priceSort === "high-low") {
        data.sort((a, b) => (b.price || 0) - (a.price || 0));
      }

      setArtworks(data);
    } catch (error) {
      setArtworksError(error instanceof Error ? error.message : "Failed to load artworks");
      setArtworks([]);
    } finally {
      setArtworksLoading(false);
    }
  }, [searchQuery, selectedMedium, selectedStyle, selectedTags, showForSale, priceSort, showHotOnly]);

  // Load user's artwork favorites
  const loadFavorites = useCallback(async () => {
    try {
      const userId = getOrCreateUserId(authUser?.id);
      const resp = await fetch(
        `${GALLERY_API_URL}/user/${encodeURIComponent(userId)}/artwork-favorites`,
      );
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data)) {
        const ids = data
          .map((fav: { artwork_id?: string }) => fav?.artwork_id ?? "")
          .filter(Boolean);
        setFavoriteIds(new Set(ids));
      }
    } catch {
      setFavoriteIds(new Set());
    }
  }, []);

  // Load user's uploads (legacy)
  const loadUploads = useCallback(async () => {
    setUploadsLoading(true);
    setUploadsError(null);
    try {
      const userId = getOrCreateUserId(authUser?.id);
      const resp = await fetch(
        `${GALLERY_API_URL}/uploads?user_id=${encodeURIComponent(userId)}`,
      );
      if (!resp.ok) throw new Error(`Failed to load uploads (${resp.status})`);
      const data = await resp.json();
      setUploads(Array.isArray(data) ? data : []);
    } catch (error) {
      setUploadsError(error instanceof Error ? error.message : "Failed to load uploads");
      setUploads([]);
    } finally {
      setUploadsLoading(false);
    }
  }, []);

  // Batch load artist profiles for artworks
  const loadArtistProfiles = useCallback(async (artworkList: Artwork[]) => {
    const artistIds = [
      ...new Set(
        artworkList
          .map((a) => a.artist_id)
          .filter((id): id is string => id !== null && id !== ""),
      ),
    ];

    if (artistIds.length === 0) return;

    try {
      const resp = await fetch(`${SB_API_BASE}/street-profiles/batch-lookup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_ids: artistIds }),
      });
      if (!resp.ok) return;
      const profiles = await resp.json();
      setArtistProfiles(profiles);
    } catch {
      // Profiles are optional, don't error
    }
  }, []);

  useEffect(() => {
    loadMediums();
    loadPopularTags();
    loadFavorites();
    loadUploads();
  }, [loadMediums, loadPopularTags, loadFavorites, loadUploads]);

  useEffect(() => {
    loadArtworks();
  }, [loadArtworks]);

  // Fetch artist profiles when artworks change
  useEffect(() => {
    if (artworks.length > 0) {
      loadArtistProfiles(artworks);
    }
  }, [artworks, loadArtistProfiles]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      loadArtworks();
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery, loadArtworks]);

  const toggleFavorite = useCallback(
    async (artworkId: string) => {
      const userId = getOrCreateUserId(authUser?.id);
      const isFav = favoriteIds.has(artworkId);
      try {
        const resp = await fetch(
          `${GALLERY_API_URL}/artworks/${encodeURIComponent(artworkId)}/favorites?user_id=${encodeURIComponent(userId)}`,
          { method: isFav ? "DELETE" : "POST" },
        );
        if (!resp.ok) throw new Error();
        setFavoriteIds((prev) => {
          const next = new Set(prev);
          if (isFav) next.delete(artworkId);
          else next.add(artworkId);
          return next;
        });
      } catch {
        console.error("Failed to toggle favorite");
      }
    },
    [favoriteIds],
  );

  const handleShareOrCopy = useCallback(async (url: string) => {
    setUploadsMessage(null);
    try {
      if (typeof navigator !== "undefined" && "share" in navigator) {
        await (navigator as Navigator).share({ url });
        setUploadsMessage("Shared!");
        setTimeout(() => setUploadsMessage(null), 2000);
        return;
      }
    } catch {
      // fall back to clipboard
    }

    try {
      await navigator.clipboard.writeText(url);
      setUploadsMessage("Link copied!");
    } catch {
      setUploadsMessage("Failed to copy link");
    }
    setTimeout(() => setUploadsMessage(null), 2000);
  }, []);

  const formatPrice = (price: number | null, currency: string) => {
    if (price === null) return null;
    return new Intl.NumberFormat("en-CA", {
      style: "currency",
      currency: currency || "CAD",
    }).format(price);
  };

  // Get unique styles from artworks for filter
  // When a medium is selected, show medium-specific sub-styles; otherwise show styles from artworks
  const availableStyles = selectedMedium && MEDIUM_STYLES[selectedMedium]
    ? MEDIUM_STYLES[selectedMedium]
    : [...new Set(artworks.map((a) => a.style).filter(Boolean))];

  // Page-specific accent text color (not in shared hook)
  const accentText = isDark ? "#FFD700" : "#000";

  // Artwork detail view — checked after all hooks
  if (artworkMatch) {
    return <ArtworkDetailView artworkId={artworkMatch[1]} onBack={() => navigate('/gallery')} onSelectTag={(tag) => { setSelectedTags([tag]); navigate('/gallery'); }} />;
  }

  // Upload page sub-route
  const isUploadPage = location.pathname === '/gallery/upload';
  if (isUploadPage) {
    return <UploadArtView onBack={() => navigate('/gallery')} />;
  }

  // Saved collections page sub-route
  const isSavedPage = location.pathname === '/gallery/saved';
  if (isSavedPage) {
    return (
      <SavedCollectionsView
        onBack={() => navigate('/gallery')}
        onSelectMedium={(medium) => {
          setSelectedMedium(medium);
          navigate('/gallery');
        }}
      />
    );
  }

  // Dashboard page sub-route
  const isDashboardPage = location.pathname === '/gallery/dashboard';
  if (isDashboardPage) {
    return <DashboardView onBack={() => navigate('/gallery')} />;
  }

  // Collections page sub-route
  const isCollectionsPage = location.pathname === '/gallery/collections';
  if (isCollectionsPage) {
    return (
      <CollectionsView
        onBack={() => navigate('/gallery')}
        onSelectMedium={(medium) => {
          setSelectedMedium(medium);
          navigate('/gallery');
        }}
      />
    );
  }

  return (
    <div style={{ background: colors.bg, minHeight: "100vh", position: "relative" }}>
      <GlassBackground />

      {/* Hero Section - Enhanced with glass effect */}
      <div
        style={{
          height: "320px",
          background: "transparent",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          position: "relative",
          overflow: "hidden",
          zIndex: 1,
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background:
              "radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.15) 0%, transparent 70%)",
          }}
        />
        <div style={{ position: "relative", zIndex: 2 }}>
          <h1
            style={{
              fontSize: "clamp(2.5rem, 5vw, 3.5rem)",
              marginBottom: "12px",
              textTransform: "uppercase",
              letterSpacing: "3px",
              color: colors.accent,
              textShadow: isDark ? "0 0 20px rgba(255, 215, 0, 0.5)" : "none",
              fontWeight: 800,
            }}
          >
            Street Gallery
          </h1>
          <p
            style={{
              color: colors.textSecondary,
              marginBottom: "24px",
              fontSize: "1.2rem",
            }}
          >
            Curated digital artifacts from the underground.
          </p>
          <div
            style={{
              display: "flex",
              gap: "15px",
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            <button
              onClick={() => navigate("/gallery/upload")}
              style={{
                background: "#FFD700",
                color: "#000",
                fontWeight: "bold",
                padding: "12px 24px",
                borderRadius: "999px",
                transition: "all 0.2s",
                textTransform: "uppercase",
                letterSpacing: "0.5px",
                fontSize: "0.9rem",
                border: "none",
                cursor: "pointer",
                boxShadow: "0 4px 14px rgba(255, 214, 0, 0.4)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "scale(1.05)";
                e.currentTarget.style.boxShadow = "0 6px 20px rgba(255, 214, 0, 0.5)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "scale(1)";
                e.currentTarget.style.boxShadow = "0 4px 14px rgba(255, 214, 0, 0.4)";
              }}
            >
              Submit Art
            </button>
            <button
              onClick={() => navigate("/gallery/collections")}
              style={{
                border: "1px solid #FFD700",
                color: "#FFD700",
                padding: "12px 24px",
                borderRadius: "999px",
                fontWeight: "bold",
                textTransform: "uppercase",
                transition: "all 0.2s",
                background: "rgba(255, 214, 0, 0.1)",
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(255, 215, 0, 0.2)";
                e.currentTarget.style.transform = "scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(255, 214, 0, 0.1)";
                e.currentTarget.style.transform = "scale(1)";
              }}
            >
              <Grid size={16} /> Collections
            </button>
            <button
              onClick={() => navigate("/gallery/dashboard")}
              style={{
                border: `1px solid ${colors.border}`,
                color: colors.textSecondary,
                padding: "12px 24px",
                borderRadius: "999px",
                fontWeight: "bold",
                textTransform: "uppercase",
                transition: "all 0.2s",
                background: colors.surface,
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = colors.surfaceHover;
                e.currentTarget.style.transform = "scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = colors.surface;
                e.currentTarget.style.transform = "scale(1)";
              }}
            >
              <BarChart3 size={16} /> Dashboard
            </button>
            <button
              onClick={() => navigate("/gallery/saved")}
              style={{
                border: `1px solid ${colors.border}`,
                color: colors.textSecondary,
                padding: "12px 24px",
                borderRadius: "999px",
                fontWeight: "bold",
                textTransform: "uppercase",
                transition: "all 0.2s",
                background: colors.surface,
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(255, 60, 80, 0.15)";
                e.currentTarget.style.transform = "scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = colors.surface;
                e.currentTarget.style.transform = "scale(1)";
              }}
            >
              <Heart size={16} /> Saved
            </button>
          </div>
        </div>
      </div>


      <div
        style={{
          display: "flex",
          minHeight: "calc(100vh - 320px)",
          maxWidth: "1600px",
          margin: "0 auto",
          width: "100%",
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Sidebar Filters - GLASSMORPHISM */}
        <aside
          style={{
            width: filterOpen ? "280px" : "0",
            borderRight: filterOpen ? `1px solid ${colors.border}` : "none",
            padding: filterOpen ? "20px" : "20px 0",
            background: colors.surface,
            backdropFilter: "blur(24px) saturate(180%)",
            WebkitBackdropFilter: "blur(24px) saturate(180%)",
            transition: "width 0.3s, padding 0.3s, opacity 0.3s",
            overflow: "hidden",
            opacity: filterOpen ? 1 : 0,
          }}
        >
          <div
            style={{
              marginBottom: "20px",
              paddingBottom: "10px",
              borderBottom: `1px solid ${colors.border}`,
            }}
          >
            <h3
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                margin: 0,
                color: colors.text,
              }}
            >
              <Filter size={18} /> Filters
            </h3>
          </div>

          <div style={{ marginBottom: "25px" }}>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontWeight: "bold",
                marginBottom: "10px",
                color: colors.textSecondary,
                fontSize: "0.9rem",
              }}
            >
              <Layers size={14} /> Medium
            </label>
            <select
              value={selectedMedium}
              onChange={(e) => { setSelectedMedium(e.target.value); setSelectedStyle(""); }}
              style={{
                width: "100%",
                background: colors.cardBg,
                backdropFilter: "blur(8px)",
                border: `1px solid ${colors.border}`,
                padding: "10px 12px",
                borderRadius: "10px",
                color: colors.text,
              }}
            >
              <option value="">All Mediums</option>
              {mediums.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          <div style={{ marginBottom: "25px" }}>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontWeight: "bold",
                marginBottom: "10px",
                color: colors.textSecondary,
                fontSize: "0.9rem",
              }}
            >
              <Tag size={14} /> {selectedMedium ? `${selectedMedium} Styles` : "Style"}
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {availableStyles.slice(0, 8).map((style) => (
                <label
                  key={style}
                  style={{
                    fontWeight: "normal",
                    color: colors.text,
                    cursor: "pointer",
                    padding: "6px 8px",
                    borderRadius: "8px",
                    background:
                      selectedStyle === style ? "rgba(255, 214, 0, 0.1)" : "transparent",
                  }}
                >
                  <input
                    type="radio"
                    name="style"
                    checked={selectedStyle === style}
                    onChange={() =>
                      setSelectedStyle(selectedStyle === style ? "" : style!)
                    }
                    style={{ marginRight: "8px", accentColor: colors.accent }}
                  />
                  {style}
                </label>
              ))}
              {selectedStyle && (
                <button
                  onClick={() => setSelectedStyle("")}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: accentText,
                    cursor: "pointer",
                    textAlign: "left",
                    padding: 0,
                    fontSize: "0.85rem",
                  }}
                >
                  Clear style filter
                </button>
              )}
            </div>
          </div>

          {/* Tags Filter */}
          {popularTags.length > 0 && (
            <div style={{ marginBottom: "25px" }}>
              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  fontWeight: "bold",
                  marginBottom: "10px",
                  color: colors.textSecondary,
                  fontSize: "0.9rem",
                }}
              >
                <Tag size={14} /> Tags
              </label>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {popularTags.slice(0, 12).map((tagItem) => {
                  const isSelected = selectedTags.includes(tagItem.tag);
                  return (
                    <button
                      key={tagItem.tag}
                      type="button"
                      onClick={() => {
                        if (isSelected) {
                          setSelectedTags(
                            selectedTags.filter((t) => t !== tagItem.tag),
                          );
                        } else {
                          setSelectedTags([...selectedTags, tagItem.tag]);
                        }
                      }}
                      style={{
                        background: isSelected
                          ? "rgba(255, 214, 0, 0.2)"
                          : "transparent",
                        border: `1px solid ${isSelected ? colors.accent : colors.border}`,
                        borderRadius: "12px",
                        padding: "4px 10px",
                        fontSize: "0.75rem",
                        color: isSelected ? accentText : colors.textSecondary,
                        cursor: "pointer",
                        transition: "all 0.15s",
                      }}
                    >
                      {tagItem.tag}
                      <span style={{ marginLeft: 4, opacity: 0.6 }}>
                        ({tagItem.count})
                      </span>
                    </button>
                  );
                })}
              </div>
              {selectedTags.length > 0 && (
                <button
                  onClick={() => setSelectedTags([])}
                  style={{
                    marginTop: 8,
                    background: "transparent",
                    border: "none",
                    color: accentText,
                    cursor: "pointer",
                    textAlign: "left",
                    padding: 0,
                    fontSize: "0.85rem",
                  }}
                >
                  Clear tags ({selectedTags.length})
                </button>
              )}
            </div>
          )}

          <div style={{ marginBottom: "25px" }}>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontWeight: "bold",
                marginBottom: "10px",
                color: colors.textSecondary,
                fontSize: "0.9rem",
              }}
            >
              <DollarSign size={14} /> For Sale
            </label>
            <label
              style={{
                fontWeight: "normal",
                color: colors.text,
                cursor: "pointer",
                padding: "6px 8px",
                borderRadius: "8px",
                background: showForSale ? "rgba(255, 214, 0, 0.1)" : "transparent",
                display: "flex",
                alignItems: "center",
              }}
            >
              <input
                type="checkbox"
                checked={showForSale}
                onChange={(e) => setShowForSale(e.target.checked)}
                style={{ marginRight: "8px", accentColor: colors.accent }}
              />
              Show only for sale
            </label>
          </div>

          {/* Price Sort */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ fontWeight: "bold", fontSize: "0.85rem", color: colors.accent, display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
              <DollarSign size={14} /> Price
            </label>
            {[
              { value: "", label: "Default" },
              { value: "low-high", label: "Lowest to Highest" },
              { value: "high-low", label: "Highest to Lowest" },
            ].map(opt => (
              <label key={opt.value} style={{ display: "flex", alignItems: "center", padding: "6px 8px", borderRadius: 8, cursor: "pointer", background: priceSort === opt.value ? "rgba(255,214,0,0.1)" : "transparent", color: colors.text, fontWeight: "normal", marginBottom: 2 }}>
                <input type="radio" name="priceSort" checked={priceSort === opt.value} onChange={() => setPriceSort(opt.value as "" | "low-high" | "high-low")} style={{ marginRight: 8, accentColor: colors.accent }} />
                {opt.label}
              </label>
            ))}
          </div>

          {/* Hot Products */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ fontWeight: "bold", fontSize: "0.85rem", color: "#ef4444", display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
              🔥 Most Popular
            </label>
            <label style={{ display: "flex", alignItems: "center", padding: "6px 8px", borderRadius: 8, cursor: "pointer", background: showHotOnly ? "rgba(239,68,68,0.1)" : "transparent", color: colors.text, fontWeight: "normal" }}>
              <input type="checkbox" checked={showHotOnly} onChange={(e) => setShowHotOnly(e.target.checked)} style={{ marginRight: 8, accentColor: "#ef4444" }} />
              Show hot products only
            </label>
          </div>

          <button
            onClick={() => {
              setSelectedMedium("");
              setSelectedStyle("");
              setSelectedTags([]);
              setShowForSale(false);
              setPriceSort("");
              setShowHotOnly(false);
              setSearchQuery("");
            }}
            style={{
              width: "100%",
              padding: "12px",
              ...glassSurface,
              borderRadius: "12px",
              color: colors.text,
              cursor: "pointer",
              fontWeight: "bold",
              transition: "all 0.2s",
            }}
            {...buttonHoverHandlers}
          >
            Clear All Filters
          </button>
        </aside>

        {/* Main Grid */}
        <div style={{ flex: 1, padding: "20px" }}>
          {/* Search and Filter Toggle - GLASSMORPHISM */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: "24px",
              gap: "16px",
            }}
          >
            <div
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                gap: "12px",
                ...glassSurface,
                padding: "12px 16px",
                borderRadius: "14px",
              }}
            >
              <Search size={18} color={colors.textMuted} />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search artworks, artists, tags..."
                style={{
                  width: "100%",
                  background: "transparent",
                  border: "none",
                  outline: "none",
                  color: colors.text,
                  fontSize: "15px",
                }}
              />
            </div>
            <button
              onClick={() => setFilterOpen(!filterOpen)}
              style={{
                color: colors.textSecondary,
                fontSize: "0.9rem",
                ...glassSurface,
                borderRadius: "14px",
                padding: "12px 20px",
                cursor: "pointer",
                fontWeight: 500,
                transition: "all 0.2s",
              }}
              {...buttonHoverHandlers}
            >
              {filterOpen ? "Hide Filters" : "Show Filters"}
            </button>
          </div>

          {/* Artworks Grid */}
          {artworksLoading ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: "60px",
                color: colors.textMuted,
              }}
            >
              <Loader2
                size={24}
                className="animate-spin"
                style={{ marginRight: 10 }}
              />
              Loading artworks...
            </div>
          ) : artworksError ? (
            <div style={{ padding: "40px", textAlign: "center", color: colors.error }}>
              {artworksError}
              <button
                onClick={loadArtworks}
                style={{
                  display: "block",
                  margin: "16px auto 0",
                  padding: "8px 16px",
                  background: colors.surface,
                  border: `1px solid ${colors.border}`,
                  borderRadius: "8px",
                  color: colors.text,
                  cursor: "pointer",
                }}
              >
                Try Again
              </button>
            </div>
          ) : artworks.length === 0 ? (
            <div style={{ padding: "60px", textAlign: "center", color: colors.textMuted }}>
              <p style={{ fontSize: "1.2rem", marginBottom: 16 }}>No artworks found</p>
              <p>Be the first to submit your art to the gallery!</p>
              <button
                onClick={() => navigate("/gallery/upload")}
                style={{
                  marginTop: 16,
                  padding: "10px 20px",
                  background: colors.accent,
                  border: "none",
                  borderRadius: "999px",
                  color: "#000",
                  fontWeight: "bold",
                  cursor: "pointer",
                }}
              >
                Submit Art
              </button>
            </div>
          ) : (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(420px, 1fr))",
                gap: "24px",
              }}
            >
              {artworks.map((art) => (
                <div
                  key={art.id}
                  onClick={() => navigate(`/gallery/artwork/${art.id}`)}
                  style={{
                    ...glassCard,
                    borderRadius: "20px",
                    overflow: "hidden",
                    transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                    cursor: "pointer",
                  }}
                  {...cardHoverHandlers}
                >
                  <div
                    style={{
                      aspectRatio: "16/9",
                      position: "relative",
                      background: "#222",
                    }}
                  >
                    <GalleryImage
                      src={art.thumbnail_url || art.image_url}
                      alt={art.title}
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleFavorite(art.id);
                      }}
                      style={{
                        position: "absolute",
                        top: 12,
                        right: 12,
                        width: 36,
                        height: 36,
                        borderRadius: "50%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "1px solid rgba(255,255,255,0.18)",
                        background: "rgba(0,0,0,0.45)",
                        cursor: "pointer",
                        zIndex: 3,
                      }}
                      aria-label={
                        favoriteIds.has(art.id)
                          ? "Remove from favorites"
                          : "Add to favorites"
                      }
                    >
                      <Heart
                        size={18}
                        color={favoriteIds.has(art.id) ? "#FFD700" : "#ffffff"}
                        fill={favoriteIds.has(art.id) ? "#FFD700" : "none"}
                      />
                    </button>
                    {art.is_for_sale && !art.is_sold && (
                      <div
                        style={{
                          position: "absolute",
                          top: 12,
                          left: 12,
                          background: "#FFD700",
                          color: "#000",
                          padding: "4px 8px",
                          borderRadius: "4px",
                          fontSize: "0.75rem",
                          fontWeight: "bold",
                        }}
                      >
                        FOR SALE
                      </div>
                    )}
                    {art.is_sold && (
                      <div
                        style={{
                          position: "absolute",
                          top: 12,
                          left: 12,
                          background: "#ef4444",
                          color: "#fff",
                          padding: "4px 8px",
                          borderRadius: "4px",
                          fontSize: "0.75rem",
                          fontWeight: "bold",
                        }}
                      >
                        SOLD
                      </div>
                    )}
                    <div
                      className="art-overlay"
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: "rgba(0,0,0,0.4)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 10,
                        opacity: 0,
                        transition: "opacity 0.2s",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.opacity = "1";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.opacity = "0";
                      }}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          const idx = artworks.findIndex((a) => a.id === art.id);
                          if (idx !== -1) setLightboxIndex(idx);
                        }}
                        style={{
                          background: "rgba(255,255,255,0.2)",
                          backdropFilter: "blur(8px)",
                          color: "#fff",
                          padding: "10px",
                          borderRadius: "50%",
                          fontWeight: "bold",
                          border: "1px solid rgba(255,255,255,0.3)",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                        title="Quick View"
                      >
                        <Maximize2 size={18} />
                      </button>
                      <button
                        style={{
                          background: "#FFD700",
                          color: "#000",
                          padding: "8px 16px",
                          borderRadius: "20px",
                          fontWeight: "bold",
                          border: "none",
                          cursor: "pointer",
                        }}
                      >
                        View Details
                      </button>
                    </div>
                  </div>
                  <div style={{ padding: "16px" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        marginBottom: "6px",
                      }}
                    >
                      <h4
                        style={{
                          fontSize: "1rem",
                          fontWeight: "bold",
                          color: colors.text,
                          margin: 0,
                        }}
                      >
                        {art.title}
                      </h4>
                      {art.is_for_sale && art.price && (
                        <span
                          style={{
                            fontSize: "0.9rem",
                            fontWeight: "bold",
                            color: accentText,
                            textShadow: isDark
                              ? "0 0 10px rgba(255, 215, 0, 0.5)"
                              : "none",
                          }}
                        >
                          {formatPrice(art.price, art.currency)}
                        </span>
                      )}
                    </div>
                    {/* Artist Profile Badge (inlined) */}
                    {art.artist_id && artistProfiles[art.artist_id] ? (
                      <div
                        style={{ marginBottom: "10px" }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <InlineProfileBadge
                          userId={art.artist_id}
                          username={artistProfiles[art.artist_id].username}
                          displayName={artistProfiles[art.artist_id].display_name}
                          avatarUrl={
                            artistProfiles[art.artist_id].avatar_url || undefined
                          }
                          isVerified={artistProfiles[art.artist_id].is_verified}
                        />
                      </div>
                    ) : art.artist_name ? (
                      <span
                        style={{
                          display: "block",
                          fontSize: "0.85rem",
                          color: colors.textSecondary,
                          marginBottom: "10px",
                        }}
                      >
                        by {art.artist_name}
                      </span>
                    ) : null}
                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: "6px",
                        marginBottom: "12px",
                      }}
                    >
                      {art.medium && (
                        <span style={glassTag}>
                          {art.medium.replace("_", " ")}
                        </span>
                      )}
                      {art.style && (
                        <span style={glassTag}>
                          {art.style}
                        </span>
                      )}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        gap: "14px",
                        color: colors.textMuted,
                        fontSize: "0.8rem",
                        alignItems: "center",
                      }}
                    >
                      <span
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "4px",
                        }}
                      >
                        <Eye size={14} /> {art.view_count}
                      </span>
                      <span
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "4px",
                        }}
                      >
                        <Heart size={14} /> {art.favorite_count}
                      </span>
                      <span
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "4px",
                        }}
                      >
                        <MessageCircle size={14} /> {art.comment_count}
                      </span>
                      {/* Message Artist Button (simple replacement) */}
                      {art.artist_id && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(
                              `/messages?to=${encodeURIComponent(art.artist_id!)}&context=${encodeURIComponent(`Hi! I love your artwork "${art.title}"!`)}`,
                            );
                          }}
                          style={{
                            marginLeft: "auto",
                            background: "transparent",
                            border: "none",
                            cursor: "pointer",
                            padding: "4px 6px",
                            borderRadius: "6px",
                            display: "inline-flex",
                            alignItems: "center",
                            justifyContent: "center",
                            color: colors.textMuted,
                            transition: "all 0.15s",
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background =
                              "rgba(255, 255, 255, 0.08)";
                            e.currentTarget.style.color = colors.text;
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = "transparent";
                            e.currentTarget.style.color = colors.textMuted;
                          }}
                          title={`Message ${art.artist_name || "Artist"}`}
                        >
                          <MessageCircle size={16} />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Lightbox Modal (simple fullscreen overlay) */}
      {lightboxIndex !== null && artworks.length > 0 && (
        <SimpleLightbox
          artworks={artworks}
          initialIndex={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
          onFavorite={toggleFavorite}
          favoriteIds={favoriteIds}
        />
      )}
    </div>
  );
}
