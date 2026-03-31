import React, { useEffect, useState, useCallback, lazy, Suspense } from "react";
import { useNavigate, useLocation } from "react-router-dom";

const SubmitArtPage = lazy(() => import("./SubmitArtPage"));
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

function ArtworkDetailView({ artworkId, onBack }: { artworkId: string; onBack: () => void }) {
  const { colors, glassCard, glassSurface } = useGlassStyles();
  const { user: authUser } = useAuthContext();
  const [artwork, setArtwork] = useState<Artwork | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFav, setIsFav] = useState(false);

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
          {/* Left: Artwork image */}
          <div style={{ flex: '1 1 500px', minWidth: 300 }}>
            <div style={{
              ...glassCard, borderRadius: 16, overflow: 'hidden', position: 'relative',
            }}>
              <img
                src={artwork.full_resolution_url || artwork.image_url}
                alt={artwork.title}
                style={{ width: '100%', display: 'block', maxHeight: '70vh', objectFit: 'contain', background: '#000' }}
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
            </div>
          </div>

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
                    ${Number(artwork.price).toFixed(2)} <span style={{ fontSize: '0.7rem', color: colors.textMuted }}>{artwork.currency}</span>
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
                    <span key={tag} style={{
                      padding: '4px 12px', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600,
                      background: 'rgba(255,214,0,0.1)', color: '#FFD600', border: '1px solid rgba(255,214,0,0.2)',
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
              <button onClick={() => { navigator.clipboard.writeText(window.location.href); }} style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '10px 20px', borderRadius: 10,
                border: `1px solid ${colors.border}`, background: 'rgba(255,255,255,0.04)',
                color: colors.text, cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600,
              }}>
                <Copy size={16} /> Share Link
              </button>
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
      </div>
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
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
  const isUploadPage = location.pathname === "/gallery/upload";

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
      const data = await resp.json();
      setArtworks(Array.isArray(data) ? data : []);
    } catch (error) {
      setArtworksError(error instanceof Error ? error.message : "Failed to load artworks");
      setArtworks([]);
    } finally {
      setArtworksLoading(false);
    }
  }, [searchQuery, selectedMedium, selectedStyle, selectedTags, showForSale]);

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
  const availableStyles = [...new Set(artworks.map((a) => a.style).filter(Boolean))];

  // Page-specific accent text color (not in shared hook)
  const accentText = isDark ? "#FFD700" : "#000";

  // Artwork detail view — checked after all hooks
  if (artworkMatch) {
    return <ArtworkDetailView artworkId={artworkMatch[1]} onBack={() => navigate('/gallery')} />;
  }

  // Render full-page upload if on /gallery/upload
  if (isUploadPage) {
    return (
      <Suspense fallback={<div style={{ padding: 40, textAlign: "center" }}>Loading...</div>}>
        <SubmitArtPage />
      </Suspense>
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
              onChange={(e) => setSelectedMedium(e.target.value)}
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
              <Tag size={14} /> Style
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {availableStyles.slice(0, 6).map((style) => (
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

          <button
            onClick={() => {
              setSelectedMedium("");
              setSelectedStyle("");
              setSelectedTags([]);
              setShowForSale(false);
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
          {/* User Uploads Section - GLASSMORPHISM */}
          <div
            style={{
              ...glassCard,
              padding: "20px",
              marginBottom: "24px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 12,
                flexWrap: "wrap",
              }}
            >
              <h3 style={{ margin: 0, color: accentText, fontSize: "1.1rem" }}>
                Your Uploaded Art
              </h3>
              <button
                type="button"
                onClick={() => navigate("/gallery/upload")}
                style={{
                  background: "rgba(255, 214, 0, 0.1)",
                  backdropFilter: "blur(10px)",
                  border: "1px solid rgba(255, 214, 0, 0.3)",
                  color: accentText,
                  padding: "8px 16px",
                  borderRadius: "999px",
                  fontWeight: "bold",
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(255, 214, 0, 0.2)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(255, 214, 0, 0.1)";
                }}
              >
                Manage uploads
              </button>
            </div>

            {uploadsMessage && (
              <p
                style={{
                  marginTop: 10,
                  marginBottom: 0,
                  color: accentText,
                  fontSize: "0.85rem",
                }}
              >
                {uploadsMessage}
              </p>
            )}

            <div style={{ marginTop: 14 }}>
              {uploadsLoading ? (
                <p style={{ margin: 0, color: colors.textMuted }}>Loading uploads...</p>
              ) : uploadsError ? (
                <p style={{ margin: 0, color: colors.error }}>{uploadsError}</p>
              ) : uploads.length === 0 ? (
                <p style={{ margin: 0, color: colors.textMuted }}>
                  No uploads yet. Click &quot;Manage uploads&quot; to submit your first
                  piece.
                </p>
              ) : (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                    gap: 16,
                  }}
                >
                  {uploads.slice(0, 6).map((upload) => (
                    <div
                      key={upload.id}
                      style={{
                        background: colors.surface,
                        backdropFilter: "blur(16px) saturate(180%)",
                        WebkitBackdropFilter: "blur(16px) saturate(180%)",
                        borderRadius: "16px",
                        overflow: "hidden",
                        border: `1px solid ${colors.border}`,
                        position: "relative",
                        transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = "translateY(-4px)";
                        e.currentTarget.style.borderColor = colors.borderHover;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = "translateY(0)";
                        e.currentTarget.style.borderColor = colors.border;
                      }}
                    >
                      <div
                        style={{
                          aspectRatio: "1",
                          background: "rgba(0,0,0,0.3)",
                          position: "relative",
                        }}
                      >
                        <img
                          loading="lazy"
                          decoding="async"
                          src={upload.image_url}
                          alt={upload.title}
                          style={{
                            width: "100%",
                            height: "100%",
                            objectFit: "cover",
                            display: "block",
                          }}
                          onError={(e) => {
                            (e.currentTarget as HTMLImageElement).src =
                              "/assets/gallery/cyberpunk.png";
                          }}
                        />
                      </div>
                      <div style={{ padding: "12px" }}>
                        <div
                          style={{
                            fontWeight: "bold",
                            color: colors.text,
                            marginBottom: 8,
                          }}
                        >
                          {upload.title}
                        </div>
                        <div style={{ display: "flex", gap: 10 }}>
                          <a
                            href={upload.image_url}
                            target="_blank"
                            rel="noreferrer"
                            style={{
                              flex: 1,
                              display: "inline-flex",
                              alignItems: "center",
                              justifyContent: "center",
                              gap: 8,
                              borderRadius: "10px",
                              border: `1px solid ${colors.border}`,
                              background: "transparent",
                              color: colors.text,
                              padding: "8px 10px",
                              fontWeight: "bold",
                              textDecoration: "none",
                              transition: "all 0.2s",
                            }}
                          >
                            <ExternalLink size={16} />
                            View
                          </a>
                          <button
                            type="button"
                            onClick={() => handleShareOrCopy(upload.image_url)}
                            style={{
                              flex: 1,
                              display: "inline-flex",
                              alignItems: "center",
                              justifyContent: "center",
                              gap: 8,
                              borderRadius: "10px",
                              border: `1px solid ${colors.border}`,
                              background: "transparent",
                              color: colors.text,
                              padding: "8px 10px",
                              fontWeight: "bold",
                              cursor: "pointer",
                              transition: "all 0.2s",
                            }}
                          >
                            {typeof navigator !== "undefined" && "share" in navigator ? (
                              <Share2 size={16} />
                            ) : (
                              <Copy size={16} />
                            )}
                            Share
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

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
                gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
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
                      aspectRatio: "1",
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
