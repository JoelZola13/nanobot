import React, { useState, useRef, useCallback } from "react";
import { X, Upload, Image, Loader2 } from "lucide-react";
import { useGlassStyles } from "../shared/useGlassStyles";
import { useAuthContext } from "~/hooks/AuthContext";
import { getOrCreateUserId } from "@/lib/userId";
import { SB_API_BASE } from "~/components/streetbot/shared/apiConfig";

interface SubmitArtModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

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

export default function SubmitArtModal({ open, onClose, onSuccess }: SubmitArtModalProps) {
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
      setTimeout(() => {
        resetForm();
        onClose();
        onSuccess?.();
      }, 1500);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setSubmitting(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setTitle("");
    setDescription("");
    setMedium("");
    setStyle("");
    setTags("");
    setIsForSale(false);
    setPrice("");
    setError(null);
    setSuccess(false);
  };

  if (!open) return null;

  const accent = "#FFD700";
  const overlayStyle: React.CSSProperties = {
    position: "fixed",
    inset: 0,
    background: "rgba(0, 0, 0, 0.7)",
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    zIndex: 9999,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "20px",
  };

  const modalStyle: React.CSSProperties = {
    ...glassCard,
    width: "100%",
    maxWidth: "640px",
    maxHeight: "90vh",
    overflow: "auto",
    borderRadius: "24px",
    padding: "32px",
    position: "relative",
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "12px 16px",
    borderRadius: "12px",
    border: `1px solid ${isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.15)"}`,
    background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.03)",
    color: colors.text,
    fontSize: "0.95rem",
    outline: "none",
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
    <div style={overlayStyle} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={modalStyle}>
        {/* Close button */}
        <button
          onClick={onClose}
          style={{
            position: "absolute", top: 16, right: 16,
            background: "none", border: "none", color: colors.textMuted,
            cursor: "pointer", padding: 8,
          }}
        >
          <X size={24} />
        </button>

        <h2 style={{ margin: "0 0 8px", color: accent, fontSize: "1.5rem" }}>
          Submit Your Art
        </h2>
        <p style={{ margin: "0 0 24px", color: colors.textMuted, fontSize: "0.9rem" }}>
          Share your work with the Street Voices community
        </p>

        {success ? (
          <div style={{ textAlign: "center", padding: "40px 0" }}>
            <div style={{ fontSize: "3rem", marginBottom: 16 }}>&#10004;</div>
            <h3 style={{ color: accent }}>Art Submitted!</h3>
            <p style={{ color: colors.textMuted }}>Your artwork is now live in the gallery.</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* Image upload area */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              style={{
                border: `2px dashed ${preview ? accent : (isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)")}`,
                borderRadius: "16px",
                padding: preview ? "0" : "40px",
                textAlign: "center",
                cursor: "pointer",
                overflow: "hidden",
                transition: "all 0.2s",
                minHeight: "200px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)",
              }}
            >
              {preview ? (
                <img src={preview} alt="Preview" style={{ width: "100%", maxHeight: "300px", objectFit: "contain" }} />
              ) : (
                <div>
                  <Image size={48} color={colors.textMuted} style={{ marginBottom: 12 }} />
                  <p style={{ color: colors.textMuted, margin: 0, fontSize: "1rem" }}>
                    Drop an image here or click to browse
                  </p>
                  <p style={{ color: colors.textMuted, margin: "8px 0 0", fontSize: "0.8rem", opacity: 0.6 }}>
                    JPG, PNG, GIF, WebP — up to 50 MB
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
                style={{ ...inputStyle, minHeight: "80px", resize: "vertical", fontFamily: "inherit" }}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Tell the story behind this piece..."
              />
            </div>

            {/* Medium + Style row */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              <div>
                <label style={labelStyle}>Medium</label>
                <select
                  style={inputStyle}
                  value={medium}
                  onChange={(e) => setMedium(e.target.value)}
                >
                  <option value="">Select medium...</option>
                  {MEDIUMS.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label style={labelStyle}>Style</label>
                <select
                  style={inputStyle}
                  value={style}
                  onChange={(e) => setStyle(e.target.value)}
                >
                  <option value="">Select style...</option>
                  {STYLES.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>

            {/* Tags */}
            <div>
              <label style={labelStyle}>Tags</label>
              <input
                style={inputStyle}
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="street art, mural, toronto (comma separated)"
              />
            </div>

            {/* For sale toggle */}
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <label style={{ ...labelStyle, margin: 0, cursor: "pointer", display: "flex", alignItems: "center", gap: 8 }}>
                <input
                  type="checkbox"
                  checked={isForSale}
                  onChange={(e) => setIsForSale(e.target.checked)}
                  style={{ width: 18, height: 18, accentColor: accent }}
                />
                This artwork is for sale
              </label>
              {isForSale && (
                <input
                  style={{ ...inputStyle, width: "120px" }}
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
                borderRadius: "12px",
                padding: "12px 16px",
                color: "#ff3b30",
                fontSize: "0.9rem",
              }}>
                {error}
              </div>
            )}

            {/* Submit button */}
            <button
              onClick={handleSubmit}
              disabled={submitting || !file || !title.trim()}
              style={{
                background: submitting || !file || !title.trim() ? "rgba(255, 214, 0, 0.3)" : accent,
                color: "#000",
                fontWeight: "bold",
                padding: "14px 24px",
                borderRadius: "999px",
                border: "none",
                cursor: submitting || !file || !title.trim() ? "not-allowed" : "pointer",
                fontSize: "1rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                transition: "all 0.2s",
              }}
            >
              {submitting ? (
                <>
                  <Loader2 size={20} style={{ animation: "spin 1s linear infinite" }} />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={20} />
                  Submit Art
                </>
              )}
            </button>
          </div>
        )}
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
