import React, { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Upload, Image, Loader2, X } from "lucide-react";
import { useGlassStyles } from "../shared/useGlassStyles";
import { GlassBackground } from "../shared/GlassBackground";
import { useAuthContext } from "~/hooks/AuthContext";
import { getOrCreateUserId } from "@/lib/userId";
import { SB_API_BASE } from "~/components/streetbot/shared/apiConfig";

const MEDIUMS = [
  "Oil Painting", "Acrylic", "Watercolor", "Digital Art", "Photography",
  "Mixed Media", "Sculpture", "Illustration", "Collage", "Charcoal",
  "Pastel", "Ink", "Spray Paint", "Textile", "Ceramic", "Other",
];

const STYLES = [
  "Abstract", "Realism", "Street Art", "Pop Art", "Impressionism",
  "Expressionism", "Minimalism", "Surrealism", "Contemporary",
  "Figurative", "Landscape", "Portrait", "Conceptual", "Folk Art", "Other",
];

export default function SubmitArtPage() {
  const navigate = useNavigate();
  const { user: authUser } = useAuthContext();
  const userId = getOrCreateUserId(authUser?.id);
  const { isDark, colors, glassCard } = useGlassStyles();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [artistName, setArtistName] = useState(authUser?.name || authUser?.username || "");
  const [medium, setMedium] = useState("");
  const [style, setStyle] = useState("");
  const [tags, setTags] = useState("");
  const [yearCreated, setYearCreated] = useState("");
  const [isForSale, setIsForSale] = useState(false);
  const [price, setPrice] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please select an image file (JPG, PNG, GIF, WebP)");
      return;
    }
    if (f.size > 50 * 1024 * 1024) {
      setError("Image must be under 50 MB");
      return;
    }
    setFile(f);
    setError(null);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) {
      setFile(f);
      setError(null);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(f);
    }
  }, []);

  const removeImage = () => {
    setFile(null);
    setPreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please select an image");
      return;
    }
    if (!title.trim()) {
      setError("Please enter a title");
      return;
    }

    setSubmitting(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("title", title.trim());
    formData.append("description", description.trim());
    formData.append("artist_name", artistName.trim() || "Anonymous");
    formData.append("medium", medium);
    formData.append("style", style);
    formData.append("tags", tags);
    formData.append("year_created", yearCreated);
    formData.append("user_id", userId);
    formData.append("is_for_sale", String(isForSale));
    if (isForSale && price) formData.append("price", price);

    try {
      const resp = await fetch(`${SB_API_BASE}/gallery/upload`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.error || `Upload failed (${resp.status})`);
      }
      setSuccess(true);
      setTimeout(() => navigate("/gallery"), 2000);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setSubmitting(false);
    }
  };

  const accent = "#FFD700";

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "14px 18px",
    borderRadius: "14px",
    border: `1px solid ${isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)"}`,
    background: isDark ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.8)",
    color: colors.text,
    fontSize: "1rem",
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.2s",
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    marginBottom: "8px",
    fontWeight: 600,
    fontSize: "0.85rem",
    color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.6)",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  };

  if (success) {
    return (
      <div style={{ position: "relative", minHeight: "100vh", overflow: "hidden" }}>
        <GlassBackground />
        <div style={{
          position: "relative", zIndex: 1,
          display: "flex", alignItems: "center", justifyContent: "center",
          minHeight: "100vh", padding: "40px 20px",
        }}>
          <div style={{
            ...glassCard, borderRadius: "28px", padding: "60px 40px",
            textAlign: "center", maxWidth: "500px",
          }}>
            <div style={{ fontSize: "4rem", marginBottom: 20 }}>&#10004;</div>
            <h2 style={{ color: accent, marginBottom: 12, fontSize: "1.8rem" }}>Art Submitted!</h2>
            <p style={{ color: colors.textMuted, fontSize: "1.1rem" }}>
              Your artwork is now live in the gallery.
            </p>
            <p style={{ color: colors.textMuted, fontSize: "0.9rem", marginTop: 16 }}>
              Redirecting to gallery...
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: "relative", minHeight: "100vh", overflow: "hidden" }}>
      <GlassBackground />

      <div style={{
        position: "relative", zIndex: 1,
        maxWidth: "720px", margin: "0 auto",
        padding: "40px 20px 80px",
      }}>
        {/* Back arrow */}
        <button
          onClick={() => navigate("/gallery")}
          style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            background: "none", border: "none",
            color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.6)",
            cursor: "pointer", padding: "8px 0", marginBottom: "24px",
            fontSize: "0.95rem", fontWeight: 500,
          }}
        >
          <ArrowLeft size={20} />
          Back to Gallery
        </button>

        {/* Header */}
        <div style={{ marginBottom: "32px" }}>
          <h1 style={{
            fontSize: "2.2rem", fontWeight: 800,
            background: `linear-gradient(135deg, ${accent}, #FFA500)`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            margin: "0 0 8px",
          }}>
            Submit Your Art
          </h1>
          <p style={{ color: colors.textMuted, fontSize: "1.05rem", margin: 0 }}>
            Share your work with the Street Voices community
          </p>
        </div>

        {/* Form card */}
        <div style={{
          ...glassCard,
          borderRadius: "24px",
          padding: "32px",
        }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>

            {/* Image upload area */}
            <div>
              <label style={labelStyle}>Artwork Image *</label>
              <div
                onClick={() => !preview && fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                style={{
                  border: `2px dashed ${preview ? accent : (isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.15)")}`,
                  borderRadius: "20px",
                  padding: preview ? "0" : "48px 24px",
                  textAlign: "center",
                  cursor: preview ? "default" : "pointer",
                  overflow: "hidden",
                  transition: "all 0.3s",
                  minHeight: "220px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  position: "relative",
                  background: isDark
                    ? "linear-gradient(135deg, rgba(128,0,255,0.08), rgba(75,0,130,0.12))"
                    : "linear-gradient(135deg, rgba(128,0,255,0.04), rgba(75,0,130,0.06))",
                }}
              >
                {preview ? (
                  <>
                    <img src={preview} alt="Preview" style={{
                      width: "100%", maxHeight: "400px", objectFit: "contain",
                    }} />
                    <button
                      onClick={(e) => { e.stopPropagation(); removeImage(); }}
                      style={{
                        position: "absolute", top: 12, right: 12,
                        background: "rgba(0,0,0,0.6)", border: "none",
                        borderRadius: "50%", width: 36, height: 36,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        cursor: "pointer", color: "#fff",
                      }}
                    >
                      <X size={18} />
                    </button>
                  </>
                ) : (
                  <div>
                    <div style={{
                      width: 72, height: 72, borderRadius: "50%",
                      background: "linear-gradient(135deg, rgba(128,0,255,0.2), rgba(255,214,0,0.2))",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      margin: "0 auto 16px",
                    }}>
                      <Image size={32} color={accent} />
                    </div>
                    <p style={{ color: colors.text, margin: 0, fontSize: "1.1rem", fontWeight: 600 }}>
                      Drop your artwork here
                    </p>
                    <p style={{ color: colors.textMuted, margin: "8px 0 0", fontSize: "0.9rem" }}>
                      or click to browse — JPG, PNG, GIF, WebP up to 50 MB
                    </p>
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                />
              </div>
            </div>

            {/* Title */}
            <div>
              <label style={labelStyle}>Title *</label>
              <input
                style={inputStyle}
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Give your artwork a title"
              />
            </div>

            {/* Artist name */}
            <div>
              <label style={labelStyle}>Artist Name</label>
              <input
                style={inputStyle}
                value={artistName}
                onChange={(e) => setArtistName(e.target.value)}
                placeholder="Your name or alias"
              />
            </div>

            {/* Description */}
            <div>
              <label style={labelStyle}>Description</label>
              <textarea
                style={{ ...inputStyle, minHeight: "100px", resize: "vertical", fontFamily: "inherit" }}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Tell the story behind this piece..."
              />
            </div>

            {/* Medium + Style row */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              <div>
                <label style={labelStyle}>Medium</label>
                <select style={inputStyle} value={medium} onChange={(e) => setMedium(e.target.value)}>
                  <option value="">Select medium...</option>
                  {MEDIUMS.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label style={labelStyle}>Style</label>
                <select style={inputStyle} value={style} onChange={(e) => setStyle(e.target.value)}>
                  <option value="">Select style...</option>
                  {STYLES.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>

            {/* Tags + Year row */}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "16px" }}>
              <div>
                <label style={labelStyle}>Tags</label>
                <input
                  style={inputStyle}
                  value={tags}
                  onChange={(e) => setTags(e.target.value)}
                  placeholder="street art, mural, toronto (comma separated)"
                />
              </div>
              <div>
                <label style={labelStyle}>Year Created</label>
                <input
                  style={inputStyle}
                  type="number"
                  value={yearCreated}
                  onChange={(e) => setYearCreated(e.target.value)}
                  placeholder={new Date().getFullYear().toString()}
                  min="1900"
                  max={new Date().getFullYear()}
                />
              </div>
            </div>

            {/* For sale toggle */}
            <div style={{
              display: "flex", alignItems: "center", gap: 16,
              padding: "16px 20px", borderRadius: "14px",
              background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.02)",
              border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)"}`,
            }}>
              <label style={{
                display: "flex", alignItems: "center", gap: 10,
                cursor: "pointer", fontWeight: 500, fontSize: "0.95rem",
                color: colors.text, margin: 0,
              }}>
                <input
                  type="checkbox"
                  checked={isForSale}
                  onChange={(e) => setIsForSale(e.target.checked)}
                  style={{ width: 20, height: 20, accentColor: accent }}
                />
                This artwork is for sale
              </label>
              {isForSale && (
                <input
                  style={{ ...inputStyle, width: "140px", margin: 0 }}
                  type="number"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  placeholder="Price (CAD)"
                  min="0"
                  step="0.01"
                />
              )}
            </div>

            {/* Error */}
            {error && (
              <div style={{
                background: "rgba(255, 59, 48, 0.1)",
                border: "1px solid rgba(255, 59, 48, 0.3)",
                borderRadius: "14px",
                padding: "14px 18px",
                color: "#ff3b30",
                fontSize: "0.95rem",
              }}>
                {error}
              </div>
            )}

            {/* Submit button */}
            <button
              onClick={handleSubmit}
              disabled={submitting || !file || !title.trim()}
              style={{
                background: submitting || !file || !title.trim()
                  ? "rgba(255, 214, 0, 0.3)"
                  : `linear-gradient(135deg, ${accent}, #FFA500)`,
                color: "#000",
                fontWeight: 700,
                padding: "16px 32px",
                borderRadius: "999px",
                border: "none",
                cursor: submitting || !file || !title.trim() ? "not-allowed" : "pointer",
                fontSize: "1.05rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 10,
                transition: "all 0.3s",
                boxShadow: submitting || !file || !title.trim()
                  ? "none"
                  : "0 4px 20px rgba(255, 214, 0, 0.4)",
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              {submitting ? (
                <>
                  <Loader2 size={22} style={{ animation: "spin 1s linear infinite" }} />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={22} />
                  Submit Art
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
