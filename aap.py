"""
Underwater Image Enhancement — Streamlit Testing App
=====================================================
Run:  streamlit run app.py
"""

import io
import os
import time
import tempfile
import numpy as np
import cv2
import torch
import streamlit as st
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_msssim import ssim as compute_ssim
import torch.nn.functional as F

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AquaEnhance · Underwater Image Restoration",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #040d14;
    color: #e0eaf2;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #060f18 !important;
    border-right: 1px solid #0d2233;
}
section[data-testid="stSidebar"] * {
    color: #a8c4d8 !important;
}

/* Header strip */
.header-strip {
    background: linear-gradient(90deg, #03212f 0%, #041a28 60%, #030e18 100%);
    border-bottom: 1px solid #0e3348;
    padding: 1.4rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-title {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #e8f4fd;
}
.header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #3a8fb5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}
.badge {
    background: #0a3a52;
    border: 1px solid #1a6a8a;
    color: #5bc4f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
    margin-left: auto;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1.5rem 0;
}
.metric-card {
    flex: 1;
    background: #060f18;
    border: 1px solid #0d2233;
    border-top: 2px solid #1a7aaa;
    padding: 1rem 1.2rem;
    border-radius: 4px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3a8fb5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8f4fd;
    line-height: 1;
}
.metric-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #3a8fb5;
    margin-left: 4px;
}

/* Image panels */
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3a8fb5;
    margin-bottom: 6px;
    padding-left: 2px;
}
.panel-label.enhanced {
    color: #22d3a8;
}

/* Upload zone */
.upload-zone {
    border: 1px dashed #1a4a62;
    background: #060f18;
    border-radius: 6px;
    padding: 2rem;
    text-align: center;
}

/* Section divider */
.divider {
    border: none;
    border-top: 1px solid #0d2233;
    margin: 1.5rem 0;
}

/* Status pill */
.status-ok {
    display: inline-block;
    background: #042a1a;
    border: 1px solid #0d5c3a;
    color: #22d3a8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
}
.status-warn {
    display: inline-block;
    background: #1a1200;
    border: 1px solid #5c3d00;
    color: #f5a623;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
}

/* Inference time bar */
.time-bar-wrap {
    background: #060f18;
    border: 1px solid #0d2233;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
}
.time-bar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #3a8fb5;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.time-bar-track {
    background: #0a1e2c;
    border-radius: 2px;
    height: 4px;
    width: 100%;
}
.time-bar-fill {
    background: linear-gradient(90deg, #1a7aaa, #22d3a8);
    height: 4px;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* Streamlit overrides */
.stButton > button {
    background: #0a3a52 !important;
    border: 1px solid #1a6a8a !important;
    color: #5bc4f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 3px !important;
    padding: 0.5rem 1.5rem !important;
    width: 100%;
}
.stButton > button:hover {
    background: #0f4d6b !important;
    border-color: #5bc4f0 !important;
}
div[data-testid="stFileUploader"] {
    border: 1px dashed #1a4a62 !important;
    background: #060f18 !important;
    border-radius: 6px !important;
}
.stSlider > div > div > div > div {
    background: #1a7aaa !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 256
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str):
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 3,
        activation      = "tanh",
    )
    # torch.load default may be weights_only=True in newer PyTorch versions.
    # For full checkpoint files that include metadata, we need weights_only=False.
    try:
        with torch.serialization.safe_globals(["numpy._core.multiarray.scalar"]):
            ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except Exception:
        # Fallback for older PyTorch versions or if safe_globals context not available
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and all(k.startswith("encoder") or k.startswith("decoder") or k.endswith("weight") for k in ckpt.keys()):
        # Heuristic for plain state-dict checkpoint
        state_dict = ckpt
    else:
        raise ValueError("Checkpoint does not contain a valid model_state dictionary.")
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    meta = {
        "epoch"   : ckpt.get("epoch", "—") if isinstance(ckpt, dict) else "—",
        "val_loss": ckpt.get("val_loss", None) if isinstance(ckpt, dict) else None,
        "val_ssim": ckpt.get("val_ssim", None) if isinstance(ckpt, dict) else None,
        "val_psnr": ckpt.get("val_psnr", None) if isinstance(ckpt, dict) else None,
    }
    return model, meta


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def enhance(img_np: np.ndarray, model) -> tuple[np.ndarray, float]:
    """
    img_np : HxWx3 uint8 RGB
    returns (enhanced HxWx3 uint8 RGB, inference_ms)
    """
    orig_h, orig_w = img_np.shape[:2]
    tensor = INFER_TRANSFORM(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(tensor)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    arr      = out.cpu().clamp(-1, 1).permute(1, 2, 0).numpy()
    enhanced = ((arr + 1.0) / 2.0 * 255).astype(np.uint8)
    enhanced = cv2.resize(enhanced, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    return enhanced, elapsed_ms


def compute_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """Compute PSNR and SSIM between original and enhanced."""
    o = torch.from_numpy(original.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    e = torch.from_numpy(enhanced.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    # Resize to same size for metric computation
    e_r = F.interpolate(e, size=(o.shape[2], o.shape[3]), mode="bilinear", align_corners=False)
    psnr = 10.0 * np.log10(1.0 / (F.mse_loss(o, e_r).item() + 1e-8))
    ssim = compute_ssim(o, e_r, data_range=1.0, size_average=True).item()
    # Colour shift: average red channel gain (indicator of underwater fix)
    r_gain = enhanced[:, :, 0].mean() / max(original[:, :, 0].mean(), 1.0)
    return {"psnr": psnr, "ssim": ssim, "r_gain": r_gain}


def to_download_bytes(img_np: np.ndarray, fmt: str = "PNG") -> bytes:
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 AquaEnhance")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("**Model Checkpoint**")
    ckpt_file = st.file_uploader(
        "Upload best_model.pth",
        type=["pth", "pt"],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Inference Settings**")

    tile_mode = st.checkbox("Tile large images (>1024px)", value=True,
                            help="Process in overlapping tiles — better quality on high-res images")
    show_diff = st.checkbox("Show difference map", value=False,
                            help="Amplified pixel-level difference between input and output")
    export_fmt = st.selectbox("Export format", ["PNG", "JPEG"], index=0)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Device**")
    device_label = "CUDA" if torch.cuda.is_available() else "CPU"
    color        = "status-ok" if torch.cuda.is_available() else "status-warn"
    st.markdown(f'<span class="{color}">{device_label}</span>', unsafe_allow_html=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.caption(gpu_name)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;'
        'color:#1a4a62;letter-spacing:1px;">UNet · ResNet34 · PyTorch</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-strip">
  <div>
    <div class="header-title">AquaEnhance</div>
    <div class="header-sub">Underwater Image Restoration · Deep Learning</div>
  </div>
  <div class="badge">v1.0 · UNet ResNet34</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
DEFAULT_CHECKPOINT_PATH = os.path.join("Model", "best_model.pth")
model = None
meta  = {}
model_loaded = False

if ckpt_file is not None:
    # Save uploaded checkpoint to a temp file in a cross-platform temp dir
    tmp_path = os.path.join(tempfile.gettempdir(), ckpt_file.name)
    with open(tmp_path, "wb") as f:
        f.write(ckpt_file.read())
    with st.spinner("Loading model weights ..."):
        try:
            model, meta = load_model(tmp_path)
            model_loaded = True
            st.markdown(
                f'<span class="status-ok">Model loaded from upload · Epoch {meta["epoch"]}</span>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

if not model_loaded and os.path.exists(DEFAULT_CHECKPOINT_PATH):
    with st.spinner("Loading local checkpoint Model/best_model.pth ..."):
        try:
            model, meta = load_model(DEFAULT_CHECKPOINT_PATH)
            model_loaded = True
            st.markdown(
                f'<span class="status-ok">Model loaded from Model/best_model.pth · Epoch {meta.get("epoch", "—")}</span>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Failed to load local checkpoint: {e}")

if model_loaded and meta.get("val_loss") is not None:
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
        f'color:#3a8fb5;margin-top:6px;'>
        f'val_loss={meta["val_loss"]:.4f} &nbsp;|&nbsp; '
        f'SSIM={meta["val_ssim"]:.4f} &nbsp;|&nbsp; '
        f'PSNR={meta["val_psnr"]:.2f} dB</div>',
        unsafe_allow_html=True,
    )

if not model_loaded:
    st.warning(
        "No checkpoint loaded. Upload `best_model.pth` in the sidebar or place one at `Model/best_model.pth`."
    )


# ─────────────────────────────────────────────
#  IMAGE UPLOAD & PROCESSING
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("#### Upload Underwater Image")

uploaded_img = st.file_uploader(
    "Drop an underwater image here",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)

if uploaded_img and model is None:
    st.warning("Please upload and load the model checkpoint first in the sidebar.")

if uploaded_img and model is not None:
    # Load image
    pil_img  = Image.open(uploaded_img).convert("RGB")
    img_np   = np.array(pil_img)
    h, w     = img_np.shape[:2]

    # Run enhancement
    with st.spinner("Enhancing ..."):
        if tile_mode and max(h, w) > 1024:
            # Tile processing for large images
            tile_sz   = 512
            overlap   = 64
            enhanced  = np.zeros_like(img_np, dtype=np.float32)
            weight_map = np.zeros((h, w), dtype=np.float32)

            for y in range(0, h, tile_sz - overlap):
                for x in range(0, w, tile_sz - overlap):
                    y2, x2 = min(y + tile_sz, h), min(x + tile_sz, w)
                    tile   = img_np[y:y2, x:x2]
                    enh_t, _ = enhance(tile, model)
                    enh_t    = cv2.resize(enh_t, (x2 - x, y2 - y))
                    enhanced[y:y2, x:x2] += enh_t.astype(np.float32)
                    weight_map[y:y2, x:x2] += 1.0

            weight_map = np.maximum(weight_map, 1.0)
            enhanced   = np.clip(enhanced / weight_map[:, :, None], 0, 255).astype(np.uint8)
            elapsed_ms = 0.0
        else:
            enhanced, elapsed_ms = enhance(img_np, model)

    metrics = compute_metrics(img_np, enhanced)

    # ── Metrics row ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">PSNR</div>
        <div class="metric-value">{metrics['psnr']:.2f}<span class="metric-unit">dB</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-label">SSIM</div>
        <div class="metric-value">{metrics['ssim']:.4f}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Red Channel Gain</div>
        <div class="metric-value">{metrics['r_gain']:.2f}<span class="metric-unit">×</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Resolution</div>
        <div class="metric-value">{w}<span class="metric-unit">×{h}</span></div>
      </div>
      {"" if tile_mode and max(h,w) > 1024 else f'''
      <div class="metric-card">
        <div class="metric-label">Inference</div>
        <div class="metric-value">{elapsed_ms:.1f}<span class="metric-unit">ms</span></div>
      </div>
      '''}
    </div>
    """, unsafe_allow_html=True)

    # ── Image columns ────────────────────────────────────────────────
    cols = [2, 2, 2] if show_diff else [2, 2]
    display_cols = st.columns(cols)

    with display_cols[0]:
        st.markdown('<div class="panel-label">Input · Degraded</div>', unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)

    with display_cols[1]:
        st.markdown('<div class="panel-label enhanced">Output · Enhanced</div>', unsafe_allow_html=True)
        st.image(enhanced, use_container_width=True)

    if show_diff:
        with display_cols[2]:
            st.markdown('<div class="panel-label">Difference Map ×5</div>', unsafe_allow_html=True)
            orig_r  = cv2.resize(img_np, (enhanced.shape[1], enhanced.shape[0]))
            diff    = np.abs(enhanced.astype(np.int16) - orig_r.astype(np.int16))
            diff_amp = np.clip(diff * 5, 0, 255).astype(np.uint8)
            st.image(diff_amp, use_container_width=True)

    # ── Channel histogram ────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    with st.expander("📊 RGB Channel Histogram", expanded=False):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams.update({
                "figure.facecolor": "#040d14",
                "axes.facecolor":   "#060f18",
                "axes.edgecolor":   "#0d2233",
                "text.color":       "#a8c4d8",
                "xtick.color":      "#3a8fb5",
                "ytick.color":      "#3a8fb5",
                "axes.spines.top":  False,
                "axes.spines.right":False,
            })
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            colors    = ["#e05252", "#52e09a", "#5282e0"]
            names     = ["Red", "Green", "Blue"]
            for i, (c, n) in enumerate(zip(colors, names)):
                axes[0].plot(cv2.calcHist([img_np],     [i], None, [256], [0, 256]).flatten(), color=c, linewidth=1.2, alpha=0.85, label=n)
                axes[1].plot(cv2.calcHist([enhanced],   [i], None, [256], [0, 256]).flatten(), color=c, linewidth=1.2, alpha=0.85)
            axes[0].set_title("Input",    color="#a8c4d8", fontsize=10)
            axes[1].set_title("Enhanced", color="#a8c4d8", fontsize=10)
            axes[0].legend(loc="upper right", fontsize=8)
            st.pyplot(fig)
        except ModuleNotFoundError:
            st.info("Install matplotlib to view the RGB histogram: `pip install matplotlib`.")
        except Exception as e:
            st.warning(f"Unable to render histogram: {e}")
        else:
            axes[1].set_title("Enhanced", color="#22d3a8", fontsize=10)
            axes[0].legend(facecolor="#060f18", labelcolor="#a8c4d8", fontsize=8)
            for ax in axes:
                ax.set_xlim(0, 255)
                ax.set_ylim(bottom=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Download ─────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    dl_bytes = to_download_bytes(enhanced, export_fmt)
    fname    = f"enhanced_{uploaded_img.name.rsplit('.', 1)[0]}.{export_fmt.lower()}"
    st.download_button(
        label    = f"⬇  Download Enhanced Image ({export_fmt})",
        data     = dl_bytes,
        file_name= fname,
        mime     = f"image/{'jpeg' if export_fmt == 'JPEG' else 'png'}",
    )

elif uploaded_img and model is None:
    st.warning("Please upload a model checkpoint in the sidebar first.")


# ─────────────────────────────────────────────
#  BATCH MODE
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
with st.expander("📁 Batch Processing", expanded=False):
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
        'color:#3a8fb5;margin-bottom:1rem;">Upload multiple images and download a ZIP</div>',
        unsafe_allow_html=True,
    )
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if batch_files and model is not None:
        if st.button("Run Batch Enhancement"):
            import zipfile, tempfile, os
            progress = st.progress(0)
            status   = st.empty()
            results  = {}

            for i, f in enumerate(batch_files):
                status.markdown(
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:0.7rem;color:#3a8fb5;">Processing {f.name} ...</span>',
                    unsafe_allow_html=True,
                )
                img  = np.array(Image.open(f).convert("RGB"))
                enh, _ = enhance(img, model)
                results[f"enhanced_{f.name}"] = to_download_bytes(enh, "PNG")
                progress.progress((i + 1) / len(batch_files))

            # Pack into ZIP
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in results.items():
                    zf.writestr(fname, data)

            status.empty()
            progress.empty()
            st.success(f"✓ {len(batch_files)} images enhanced")
            st.download_button(
                "⬇  Download All (ZIP)",
                data      = zip_buf.getvalue(),
                file_name = "aquaenhance_batch.zip",
                mime      = "application/zip",
            )
    elif batch_files and model is None:
        st.warning("Upload a model checkpoint first.")