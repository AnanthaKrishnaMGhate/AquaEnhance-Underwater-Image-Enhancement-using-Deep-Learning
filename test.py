"""
AquaEnhance Pro — Maximum Clarity Underwater Reconstruction + Groq AI
======================================================================
Based on the stable production pipeline, enhanced with:

  SAFE BLENDING  — every formula stage uses blend weight, no blowout
  NEW FILTERS    — Median, Gaussian, Laplacian, High-Boost, Sobel, Homomorphic
  GROQ AI        — analyses image, explains stages, suggests parameters

Pipeline order:
  [01] Physics Color Correction   Beer-Lambert attenuation
  [02] White Balance              SoG(p=6) + Gray World fusion
  [03] Median Filter              Salt-and-pepper noise removal
  [04] Gaussian Denoise           Additive Gaussian noise smoothing
  [05] CLAHE (LAB)                Local adaptive contrast
  [06] Dark Channel Prior         Guided-filter refined dehazing
  [07] Homomorphic Filter         Illumination/reflectance separation (freq)
  [08] DFT Butterworth Bandpass   Frequency haze+noise removal
  [09] Wiener Deconvolution       PSF inversion — scatter reversal
  [10] NLM Denoising              Patch-based optimal denoising
  [11] Vignette Correction        Radial illumination fix
  [12] MSRCP Retinex              Multi-scale log illumination
  [13] Bilateral + Guided         Edge-preserving final smoothing
  [14] Laplacian Sharpening       2nd-order edge enhancement
  [15] Unsharp Masking            Gaussian detail recovery
  [16] High-Boost Filter          Amplified high-frequency boost
  [17] Sobel Edge Enhance         Directional gradient sharpening
  [18] Gamma Correction           Power-law brightness
  [19] Saturation Boost           HSV chroma restoration
  [20] Color Balance Stretch      Percentile bias removal
       UNet ResNet34              Deep learning global pass
  [→]  LAB Weighted Fusion        DL ⊕ formula in perceptual space

Run: streamlit run app.py
"""

import io
import os
import time
import warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_msssim import ssim as compute_ssim
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaEnhance Pro",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS  (from the stable version you approved, DM Sans theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&display=swap');

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#07080a;color:#c9d8e4;}

section[data-testid="stSidebar"]{background:#080b0e!important;border-right:1px solid #151e26;}
section[data-testid="stSidebar"] *{color:#7a9db5!important;}
section[data-testid="stSidebar"] label{color:#9ab8cc!important;}

.hdr{padding:1.6rem 0 1.4rem;border-bottom:1px solid #151e26;margin-bottom:2rem;}
.hdr-eye{font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:3px;
  color:#1e6a8a;text-transform:uppercase;margin-bottom:.5rem;}
.hdr-title{font-size:2rem;font-weight:600;color:#e8f4fc;letter-spacing:-1px;line-height:1.1;}
.hdr-title span{color:#1aa8d8;}
.hdr-desc{font-size:.82rem;color:#4a7a92;margin-top:.5rem;line-height:1.5;}

.ribbon{display:flex;gap:0;margin:1.2rem 0;border:1px solid #151e26;border-radius:4px;overflow:hidden;flex-wrap:wrap;}
.rb{flex:1;min-width:42px;padding:6px 4px;text-align:center;background:#0a0d10;border-right:1px solid #151e26;}
.rb:last-child{border-right:none;}
.rb.on{background:#071420;border-top:2px solid #1aa8d8;}
.rb.final{background:#071a0f;border-top:2px solid #1dd4a0;}
.rb-n{font-family:'DM Mono',monospace;font-size:.48rem;color:#1e6a8a;letter-spacing:1px;}
.rb-l{font-size:.55rem;font-weight:500;color:#6a96ae;margin-top:1px;}

.mrow{display:flex;gap:8px;margin:1.2rem 0;flex-wrap:wrap;}
.mc{flex:1;min-width:100px;background:#0a0d10;border:1px solid #151e26;
  border-radius:4px;padding:.85rem 1rem;}
.mc.hi{border-left:3px solid #1aa8d8;}
.mc.best{border-left:3px solid #1dd4a0;}
.ml{font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:2px;
  color:#1e6a8a;text-transform:uppercase;margin-bottom:4px;}
.mv{font-size:1.5rem;font-weight:600;color:#e8f4fc;line-height:1;}
.mu{font-family:'DM Mono',monospace;font-size:.62rem;color:#2a7fa8;margin-left:3px;}

.sec{font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:3px;
  text-transform:uppercase;color:#1e6a8a;margin:1.6rem 0 .7rem;
  display:flex;align-items:center;gap:8px;}
.sec::after{content:'';flex:1;height:1px;background:#151e26;}

.fwrap{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:.5rem 0;}
.fc{background:#0a0d10;border:1px solid #151e26;border-left:3px solid #1e4a62;
  border-radius:3px;padding:.9rem 1rem;}
.fc.key{border-left-color:#1aa8d8;}
.ft{font-size:.78rem;font-weight:600;color:#1aa8d8;margin-bottom:.3rem;}
.fd{font-size:.72rem;color:#4a7a92;line-height:1.55;}
.fm{font-family:'DM Mono',monospace;font-size:.62rem;color:#1dd4a0;
  background:#060809;padding:5px 8px;border-radius:2px;
  display:block;margin-top:.4rem;white-space:pre-wrap;}

.pl{font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:2px;
  text-transform:uppercase;color:#1e6a8a;margin-bottom:5px;}
.pl.out{color:#1dd4a0;font-weight:500;}

.ok{display:inline-block;background:#071a0f;border:1px solid #0d4a26;
  color:#1dd4a0;font-family:'DM Mono',monospace;font-size:.55rem;
  letter-spacing:1.5px;padding:3px 10px;border-radius:2px;text-transform:uppercase;}
.warn{display:inline-block;background:#130d00;border:1px solid #4a2e00;
  color:#f5a623;font-family:'DM Mono',monospace;font-size:.55rem;
  letter-spacing:1.5px;padding:3px 10px;border-radius:2px;text-transform:uppercase;}

/* Groq chat */
.chat-wrap{background:#0a0d10;border:1px solid #151e26;border-radius:5px;
  padding:.9rem;max-height:400px;overflow-y:auto;margin:.5rem 0;}
.msg-u{background:#071420;border:1px solid #1e3a52;border-radius:3px;
  padding:.55rem .85rem;margin-bottom:.55rem;font-size:.8rem;color:#a8c8e0;}
.msg-u::before{content:'You  ';font-family:'DM Mono',monospace;font-size:.48rem;
  letter-spacing:2px;color:#1aa8d8;text-transform:uppercase;display:block;margin-bottom:2px;}
.msg-a{background:#08100a;border:1px solid #0d3020;border-radius:3px;
  padding:.55rem .85rem;margin-bottom:.55rem;font-size:.8rem;color:#b8d8c0;line-height:1.6;}
.msg-a::before{content:'Groq  ';font-family:'DM Mono',monospace;font-size:.48rem;
  letter-spacing:2px;color:#1dd4a0;text-transform:uppercase;display:block;margin-bottom:2px;}

.stButton>button{background:#0a1520!important;border:1px solid #1e4a62!important;
  color:#1aa8d8!important;font-family:'DM Sans',sans-serif!important;
  font-weight:500!important;letter-spacing:.3px!important;border-radius:4px!important;
  width:100%;}
.stButton>button:hover{background:#0f1e2e!important;border-color:#1aa8d8!important;}
div[data-testid="stFileUploader"]{border:1px dashed #1e3a4e!important;
  background:#0a0d10!important;border-radius:4px!important;}
.stProgress>div>div{background:linear-gradient(90deg,#1aa8d8,#1dd4a0)!important;}
div[data-testid="stExpander"]{background:#0a0d10!important;
  border:1px solid #151e26!important;border-radius:4px!important;}
textarea{background:#070a0e!important;color:#c9d8e4!important;
  border:1px solid #1e3a52!important;border-radius:3px!important;font-size:.82rem!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 256
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_TF = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────────────────────────────────────
#  CORE UTILITY — safe_blend prevents blowout from any single stage
# ─────────────────────────────────────────────────────────────────────────────
def blend(original: np.ndarray, processed: np.ndarray, w: float) -> np.ndarray:
    """
    Alpha-blend processed result with original input.
    out = clip(w·processed + (1-w)·original, 0, 255)
    w=1.0 → fully processed result
    w=0.5 → equal mix
    Guarantees pixel values never exceed [0,255] regardless of stage output.
    """
    return np.clip(
        w * processed.astype(np.float32) + (1.0 - w) * original.astype(np.float32),
        0, 255
    ).astype(np.uint8)


def safe_norm(img_f: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile stretch per channel. Prevents blowout after frequency-domain ops."""
    out = np.zeros_like(img_f)
    for c in range(3):
        p_lo = np.percentile(img_f[:, :, c], lo)
        p_hi = np.percentile(img_f[:, :, c], hi)
        if p_hi > p_lo:
            out[:, :, c] = np.clip((img_f[:, :, c] - p_lo) / (p_hi - p_lo) * 255, 0, 255)
        else:
            out[:, :, c] = np.clip(img_f[:, :, c], 0, 255)
    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=3, classes=3, activation="tanh")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.to(DEVICE).eval()
    return m, {k: ck.get(k) for k in ["epoch", "val_loss", "val_ssim", "val_psnr"]}


def dl_enhance(img: np.ndarray, model) -> np.ndarray:
    oh, ow = img.shape[:2]
    t = INFER_TF(image=img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)[0]
    arr = out.cpu().clamp(-1, 1).permute(1, 2, 0).numpy()
    return cv2.resize(((arr + 1.0) / 2.0 * 255).astype(np.uint8),
                      (ow, oh), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
#  FORMULA STAGES
#  Each returns uint8 RGB.  Blend weight applied in pipeline runner.
# ─────────────────────────────────────────────────────────────────────────────

def f_physics_cc(img: np.ndarray) -> np.ndarray:
    """
    [01] Beer-Lambert Physics Color Correction
    J(c) = I(c)/exp(-β_c·d_est)  β=[0.30,0.06,0.02] R,G,B
    d_est = 1 − min_c(I(c)/255)  (dark pixel → deep water proxy)
    Percentile stretch [0.5,99.5] clips outlier pixels.
    """
    f     = img.astype(np.float32) / 255.0
    d_est = np.clip(1.0 - np.min(f, axis=2), 0.05, 0.95)
    beta  = np.array([0.30, 0.06, 0.02], np.float32)
    out   = np.zeros_like(f)
    for c in range(3):
        out[:, :, c] = np.clip(f[:, :, c] / np.maximum(np.exp(-beta[c]*d_est), 0.08), 0, 1)
    for c in range(3):
        lo = np.percentile(out[:, :, c], 0.5)
        hi = np.percentile(out[:, :, c], 99.5)
        out[:, :, c] = np.clip((out[:, :, c]-lo)/(hi-lo+1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def f_white_balance(img: np.ndarray) -> np.ndarray:
    """
    [02] White Balance — SoG(p=6) + Gray World fusion
    SoG: scale[c] = μ_all / ||I^c||_p   (p=6 Minkowski)
    GW:  scale[c] = μ_all / μ[c]
    Blend: 0.6·SoG + 0.4·GW, clip [0.5, 3.0]
    """
    f  = img.astype(np.float32)
    mu = f.mean(axis=(0, 1)); gray = mu.mean()
    norms = np.power(np.mean(np.power(np.abs(f)+1e-6, 6), axis=(0,1)), 1/6)
    sog   = np.where(mu>0, gray/mu, 1.0)*0.6 + np.where(norms>0, gray/norms, 1.0)*0.4
    return np.clip(f * np.clip(sog, 0.5, 3.0), 0, 255).astype(np.uint8)


def f_median(img: np.ndarray, k: int = 3) -> np.ndarray:
    """
    [03] Median Filter — salt-and-pepper / impulse noise
    I_out(x,y) = median{I(x+i,y+j) | (i,j)∈W_k}
    Non-linear; k=3 removes impulse noise without blurring edges.
    """
    return cv2.medianBlur(img, k)


def f_gaussian(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    [04] Gaussian Smoothing — additive Gaussian noise
    I_out = I * G_σ,  G_σ(x,y) = exp(-(x²+y²)/2σ²) / 2πσ²
    σ=0.8: conservative — smooths noise spikes without visible softening.
    """
    return cv2.GaussianBlur(img, (0, 0), sigma)


def f_clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    [05] CLAHE on L channel in LAB
    Redistributes local histogram per tile; clip=2.0 limits noise amplification.
    Only L channel modified — A,B (colour) preserved exactly.
    """
    lab          = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _guided_filter(guide_u8: np.ndarray, src_f: np.ndarray,
                    r: int = 40, eps: float = 1e-3) -> np.ndarray:
    I  = guide_u8.astype(np.float32)/255.0; p = src_f.astype(np.float32)
    d  = 2*r+1; box = lambda x: cv2.boxFilter(x, -1, (d,d))
    mI = box(I); mp = box(p); varI = box(I*I)-mI*mI; covIp = box(I*p)-mI*mp
    a  = covIp/(varI+eps); b = mp-a*mI
    return box(a)*I + box(b)


def f_dark_channel(img: np.ndarray, patch: int = 15,
                    omega: float = 0.92, t_min: float = 0.1) -> np.ndarray:
    """
    [06] Dark Channel Prior + Guided Filter (He 2009)
    J_dark = min_c min_{Ω(x)} I^c(y)
    A: top 0.1% brightest pixels in dark channel
    t(x) = 1 - ω·J_dark/A  (guided-filter refined, r=40)
    J(x) = (I-A)/max(t,t_min) + A
    """
    f    = img.astype(np.float64)/255.0
    dark = cv2.erode(np.min(f,axis=2).astype(np.float32),
                     cv2.getStructuringElement(cv2.MORPH_RECT,(patch,patch)))
    idx  = np.argsort(dark.flatten())[-max(1,int(0.001*dark.size)):]
    A    = np.clip([f[:,:,c].flatten()[idx].max() for c in range(3)], 0.1, 1.0)
    t    = np.clip(1.0-omega*np.min(f/A,axis=2), t_min, 1.0).astype(np.float32)
    t    = np.clip(_guided_filter(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), t), t_min, 1.0)
    J    = (f-A)/t[:,:,None]+A
    return np.clip(J*255, 0, 255).astype(np.uint8)


def f_homomorphic(img: np.ndarray,
                   gamma_l: float = 0.5, gamma_h: float = 1.5,
                   c: float = 1.0, d0: float = 30.0) -> np.ndarray:
    """
    [07] Homomorphic Filter (frequency domain illumination separation)
    Model: f(x,y) = i(x,y)·r(x,y)  illumination × reflectance
    log domain: z = log(f+1) = log(i+1) + log(r+1)
    H(D) = (γ_H - γ_L)·[1 - exp(-c·D²/D₀²)] + γ_L  (Butterworth-style)
    Boosts high-freq reflectance (detail), suppresses low-freq illumination (haze).
    Output: exp(IDFT(H·DFT(z))) - 1, then percentile norm.
    """
    h, w   = img.shape[:2]
    cy, cx = h//2, w//2
    u      = (np.arange(h)-cy)[:,None].astype(np.float32)
    v      = (np.arange(w)-cx)[None,:].astype(np.float32)
    D2     = u**2 + v**2
    H      = (gamma_h-gamma_l)*(1.0-np.exp(-c*D2/(d0**2+1e-6))) + gamma_l

    out = np.zeros_like(img, dtype=np.float32)
    for ch in range(3):
        log_img = np.log1p(img[:,:,ch].astype(np.float32))
        Fsh     = np.fft.fftshift(np.fft.fft2(log_img))
        filt    = np.abs(np.fft.ifft2(np.fft.ifftshift(Fsh*H)))
        out[:,:,ch] = np.expm1(filt)
    return safe_norm(out)


def f_butterworth(img: np.ndarray,
                   hp_d0: float = 25.0, hp_n: int = 2,
                   lp_d0: float = 70.0, lp_n: int = 4) -> np.ndarray:
    """
    [08] DFT Butterworth Bandpass per channel
    H_HP(D) = 1 - 1/[1+(D/D₀_hp)^2n]   removes low-freq haze
    H_LP(D) = 1/[1+(D/D₀_lp)^2n]        removes high-freq noise
    H = H_HP · H_LP  applied to shifted DFT, output safe_norm.
    """
    h, w   = img.shape[:2]
    cy, cx = h//2, w//2
    u = (np.arange(h)-cy)[:,None].astype(np.float32)
    v = (np.arange(w)-cx)[None,:].astype(np.float32)
    D = np.sqrt(u**2+v**2)+1e-6
    H = (1.0-1.0/(1.0+(D/hp_d0)**(2*hp_n))) * (1.0/(1.0+(D/lp_d0)**(2*lp_n)))
    out = np.zeros_like(img, dtype=np.float32)
    for ch in range(3):
        Fsh = np.fft.fftshift(np.fft.fft2(img[:,:,ch].astype(np.float32)))
        out[:,:,ch] = np.abs(np.fft.ifft2(np.fft.ifftshift(Fsh*H)))
    return safe_norm(out)


def f_wiener(img: np.ndarray, k: float = 0.005,
              sigma: float = 1.2) -> np.ndarray:
    """
    [09] Wiener Deconvolution (frequency domain)
    PSF model: Gaussian(σ) approximates water scatter blur
    W(u,v) = H*(u,v) / (|H(u,v)|² + K)
    Ĵ = IDFT(W · DFT(I))
    K=0.005 mild; lower K = sharper but more ringing risk.
    Output safe_norm prevents blowout.
    """
    h, w = img.shape[:2]
    psz  = max(3, min(h,w)//8); psz += (psz%2==0)
    psf  = cv2.getGaussianKernel(psz, sigma)
    psf  = (psf@psf.T).astype(np.float32); psf/=psf.sum()
    pad  = np.zeros((h,w),np.float32); pad[:psz,:psz]=psf
    H    = np.fft.fft2(pad)
    W    = np.conj(H)/(np.abs(H)**2+k)
    out  = np.zeros_like(img, dtype=np.float32)
    for ch in range(3):
        out[:,:,ch] = np.abs(np.fft.ifft2(W*np.fft.fft2(img[:,:,ch].astype(np.float32))))
    return safe_norm(out)


def f_nlm(img: np.ndarray, h: float = 5.0) -> np.ndarray:
    """
    [10] Non-Local Means Denoising (optimal patch method)
    NL[u](x) = Σ_y w(x,y)·u(y) / Σ_y w(x,y)
    w(x,y) = exp(-||B_x-B_y||²/h²)
    Best texture preservation vs Gaussian/median. h=5 conservative.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)


def f_vignette(img: np.ndarray, sigma_frac: float = 0.6) -> np.ndarray:
    """
    [11] Vignette Correction — Gaussian radial model
    G(r) = exp(-r²/2σ²),  σ = sigma_frac·min(H,W)
    I_out = I / max(G, 0.4),  clipped [1.0, 2.2]
    Cap at 2.2× prevents peripheral noise explosion.
    """
    h, w = img.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    G    = np.exp(-((X-w/2)**2+(Y-h/2)**2)/(2*(sigma_frac*min(h,w))**2))
    mask = np.clip(1.0/np.maximum(G/G.max(), 0.4), 1.0, 2.2).astype(np.float32)
    return np.clip(img.astype(np.float32)*mask[:,:,None], 0, 255).astype(np.uint8)


def f_retinex(img: np.ndarray,
               sigmas: tuple = (15, 80, 250)) -> np.ndarray:
    """
    [12] MSRCP Retinex (Jobson 1997)
    SSR_s(x) = log(I+1) - log(G_σs*I+1)
    MSR = Σ(1/S)·SSR_s
    CP: I_out = MSR·(I/I_gray)  preserves chromaticity.
    Percentile stretch [0.5,99.5].
    """
    f  = img.astype(np.float32)+1.0; R = np.zeros_like(f)
    for s in sigmas:
        R += (np.log(f)-np.log(np.maximum(cv2.GaussianBlur(f,(0,0),s),1.0)))/len(sigmas)
    out = R * (f/np.maximum(f.mean(axis=2,keepdims=True),1.0))
    res = np.zeros_like(f)
    for c in range(3):
        a,b = np.percentile(out[:,:,c],0.5), np.percentile(out[:,:,c],99.5)
        res[:,:,c] = np.clip((out[:,:,c]-a)/(b-a+1e-6)*255, 0, 255)
    return res.astype(np.uint8)


def f_bilateral(img: np.ndarray,
                 sigma_c: float = 20.0, sigma_s: float = 20.0) -> np.ndarray:
    """
    [13] Bilateral + Guided Filter (2-pass edge-preserving smoothing)
    Bilateral: w ∝ exp(-|Δpos|²/2σ_s²)·exp(-|ΔI|²/2σ_c²)
    Guided (r=4, ε=0.01): restores edges softened by bilateral.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bil = cv2.bilateralFilter(bgr, 7, sigma_c, sigma_s)
    rgb = cv2.cvtColor(bil, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        out[:,:,c] = _guided_filter(gray, rgb[:,:,c].astype(np.float32), r=4, eps=0.01)*255
    return np.clip(out, 0, 255).astype(np.uint8)


def f_laplacian(img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    [14] Laplacian Sharpening (2nd-order derivative)
    ∇²I ≈ [[0,-1,0],[-1,4,-1],[0,-1,0]] * I
    I_out = I + α·∇²I,  α=0.5 moderate edge boost.
    Detects all edge orientations simultaneously.
    """
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    return np.clip(img.astype(np.float32)+alpha*lap, 0, 255).astype(np.uint8)


def f_unsharp(img: np.ndarray, sigma: float = 0.8,
               strength: float = 1.2, threshold: int = 3) -> np.ndarray:
    """
    [15] Threshold Unsharp Masking
    M(x) = I(x) - GaussBlur(I,σ)     high-pass residual
    I_out = I + λ·M  if |M| > threshold
    threshold=3: flat regions skipped — no noise amplification.
    """
    blur  = cv2.GaussianBlur(img, (0,0), sigma)
    mask  = img.astype(np.float32)-blur.astype(np.float32)
    apply = (np.abs(mask) > threshold).astype(np.float32)
    return np.clip(img.astype(np.float32)+strength*mask*apply, 0, 255).astype(np.uint8)


def f_highboost(img: np.ndarray, A: float = 1.6) -> np.ndarray:
    """
    [16] High-Boost Filter
    HP(x) = I(x) - GaussBlur(I, σ=2)
    I_out = A·I - GaussBlur(I) = I + (A-1)·HP(x)
    A=1.6: strong high-frequency amplification for severely blurred images.
    """
    lp  = cv2.GaussianBlur(img, (0,0), 2.0)
    return np.clip(A*img.astype(np.float32)-lp.astype(np.float32), 0, 255).astype(np.uint8)


def f_sobel(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """
    [17] Sobel Edge Enhancement (directional gradients)
    Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * I  (horizontal)
    Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * I   (vertical)
    G  = √(Gx²+Gy²)  gradient magnitude
    I_out = I + strength·G   strength=0.25 subtle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    Gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(Gx**2+Gy**2)
    if mag.max() > 0: mag = mag/mag.max()*255
    return np.clip(img.astype(np.float32)+strength*mag[:,:,None], 0, 255).astype(np.uint8)


def f_gamma(img: np.ndarray, gamma: float = 0.82) -> np.ndarray:
    """
    [18] Power-Law Gamma Correction
    I_out = 255·(I/255)^γ   applied via LUT (O(1) per pixel).
    γ=0.82 brightens dark underwater scenes without overexposure.
    """
    lut = np.array([(i/255.0)**gamma*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def f_saturation(img: np.ndarray, factor: float = 1.25) -> np.ndarray:
    """
    [19] Saturation Boost in HSV space
    S_out = clip(S_in · factor, 0, 255)
    factor=1.25: restores colour vibrancy lost to water absorption.
    """
    hsv         = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1]  = np.clip(hsv[:,:,1]*factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def f_stretch(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """
    [20] Per-Channel Percentile Contrast Stretch
    I_out[c] = clip((I[c]-p_lo)/(p_hi-p_lo), 0,1)·255
    lo=0.5%, hi=99.5% — removes residual colour bias from any prior stage.
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        p_lo = np.percentile(img[:,:,c], lo)
        p_hi = np.percentile(img[:,:,c], hi)
        out[:,:,c] = np.clip((img[:,:,c].astype(np.float32)-p_lo)/(p_hi-p_lo+1e-6)*255, 0, 255)
    return out.astype(np.uint8)


def f_fusion(dl: np.ndarray, formula: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    """
    [→] Weighted LAB Fusion
    F = α·LAB(DL) + (1-α)·LAB(formula) → RGB
    LAB is perceptually uniform — blending avoids RGB colour fringing.
    Final bilateral(σ_c=15, σ_s=15) removes seam artefacts.
    """
    h, w   = dl.shape[:2]
    lab_d  = cv2.cvtColor(dl,  cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_f  = cv2.cvtColor(cv2.resize(formula,(w,h)), cv2.COLOR_RGB2LAB).astype(np.float32)
    fused  = np.clip(alpha*lab_d+(1-alpha)*lab_f, 0, 255).astype(np.uint8)
    rgb    = cv2.cvtColor(fused, cv2.COLOR_LAB2RGB)
    bgr    = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr    = cv2.bilateralFilter(bgr, 5, 15, 15)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE RUNNER
#  Blend weights chosen so each stage contributes without dominating
# ─────────────────────────────────────────────────────────────────────────────
STAGE_WEIGHTS = {
    "physics": 0.80, "wb": 0.80, "median": 0.70, "gaussian": 0.70,
    "clahe": 0.75,   "dcp": 0.72, "homo": 0.60,   "butter": 0.58,
    "wiener": 0.60,  "nlm": 0.75, "vign": 0.70,   "retinex": 0.62,
    "bilat": 0.75,   "lap": 0.70, "usm": 0.80,    "hb": 0.65,
    "sobel": 0.60,   "gamma": 1.0, "sat": 1.0,    "stretch": 0.90,
}

def run_pipeline(img: np.ndarray, model, cfg: dict, on_prog=None) -> dict:
    steps  = {}
    total  = 22
    tick_n = [0]

    def tick(key, raw_out, w):
        """Apply blend weight, store, report progress."""
        out = blend(img_cur[0], raw_out, w)
        img_cur[0] = out.copy()
        steps[key] = out.copy()
        tick_n[0] += 1
        if on_prog: on_prog(min(tick_n[0]/total, 1.0), key)
        return out

    img_cur = [img.copy()]   # mutable reference

    # DL branch — runs on raw input in parallel
    dl_ms  = 0.0
    if cfg["dl"] and model is not None:
        t0    = time.perf_counter()
        dl_out = dl_enhance(img, model)
        dl_ms  = (time.perf_counter()-t0)*1000
    else:
        dl_out = img.copy()
    steps["00·DL"] = dl_out.copy()
    tick_n[0] += 1

    # Formula branch — sequential with safe blending
    W = STAGE_WEIGHTS
    if cfg["physics"]: tick("01·Physics CC",    f_physics_cc(img_cur[0]),          W["physics"])
    if cfg["wb"]:      tick("02·White Balance",  f_white_balance(img_cur[0]),       W["wb"])
    if cfg["median"]:  tick("03·Median Filter",  f_median(img_cur[0], cfg["med_k"]),W["median"])
    if cfg["gaussian"]:tick("04·Gaussian Filter",f_gaussian(img_cur[0],cfg["g_s"]), W["gaussian"])
    if cfg["clahe"]:   tick("05·CLAHE",          f_clahe(img_cur[0],cfg["cl_c"],cfg["cl_t"]),W["clahe"])
    if cfg["dcp"]:     tick("06·Dark Channel",   f_dark_channel(img_cur[0],cfg["dcp_p"],cfg["dcp_o"]),W["dcp"])
    if cfg["homo"]:    tick("07·Homomorphic",    f_homomorphic(img_cur[0],cfg["h_gl"],cfg["h_gh"],cfg["h_d0"]),W["homo"])
    if cfg["butter"]:  tick("08·DFT Butterworth",f_butterworth(img_cur[0],cfg["b_hp"],cfg["b_lp"]),W["butter"])
    if cfg["wiener"]:  tick("09·Wiener",         f_wiener(img_cur[0],cfg["w_k"],cfg["w_s"]),W["wiener"])
    if cfg["nlm"]:     tick("10·NLM Denoise",    f_nlm(img_cur[0],cfg["nlm_h"]),    W["nlm"])
    if cfg["vign"]:    tick("11·Vignette",       f_vignette(img_cur[0],cfg["v_s"]), W["vign"])
    if cfg["retinex"]: tick("12·Retinex",        f_retinex(img_cur[0]),             W["retinex"])
    if cfg["bilat"]:   tick("13·Bilateral+Guided",f_bilateral(img_cur[0],cfg["bil_c"],cfg["bil_s"]),W["bilat"])
    if cfg["lap"]:     tick("14·Laplacian",      f_laplacian(img_cur[0],cfg["lap_a"]),W["lap"])
    if cfg["usm"]:     tick("15·Unsharp Mask",   f_unsharp(img_cur[0],cfg["u_s"],cfg["u_str"],cfg["u_thr"]),W["usm"])
    if cfg["hb"]:      tick("16·High-Boost",     f_highboost(img_cur[0],cfg["hb_A"]),W["hb"])
    if cfg["sobel"]:   tick("17·Sobel Sharpen",  f_sobel(img_cur[0],cfg["sb_str"]), W["sobel"])
    if cfg["gamma"]:   tick("18·Gamma",          f_gamma(img_cur[0],cfg["gam"]),    W["gamma"])
    if cfg["sat"]:     tick("19·Saturation",     f_saturation(img_cur[0],cfg["sat_f"]),W["sat"])
    if cfg["stretch"]: tick("20·Color Stretch",  f_stretch(img_cur[0],cfg["str_lo"],cfg["str_hi"]),W["stretch"])

    final = f_fusion(dl_out, img_cur[0], alpha=cfg["alpha"])
    steps["FINAL"] = np.clip(final, 0, 255).astype(np.uint8)
    steps["_dl_ms"] = dl_ms
    return steps


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(orig: np.ndarray, enh: np.ndarray) -> dict:
    o  = torch.from_numpy(orig.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
    e  = torch.from_numpy(enh.astype(np.float32) /255.).permute(2,0,1).unsqueeze(0)
    e  = F.interpolate(e,size=(o.shape[2],o.shape[3]),mode="bilinear",align_corners=False)
    psnr = 10.*np.log10(1./(F.mse_loss(o,e).item()+1e-8))
    ssim = compute_ssim(o,e,data_range=1.0,size_average=True).item()
    gray = cv2.cvtColor(enh,cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray,cv2.CV_64F).var()
    rg = enh[:,:,0].astype(float)-enh[:,:,1].astype(float)
    yb = 0.5*(enh[:,:,0].astype(float)+enh[:,:,1].astype(float))-enh[:,:,2].astype(float)
    colorfulness = np.sqrt(rg.std()**2+yb.std()**2)+0.3*np.sqrt(rg.mean()**2+yb.mean()**2)
    return {"psnr":psnr,"ssim":ssim,"sharpness":sharpness,"colorfulness":colorfulness}


def to_bytes(img: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO(); Image.fromarray(img).save(buf,format=fmt); return buf.getvalue()


def img_summary(img: np.ndarray) -> str:
    g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return (f"shape={img.shape[1]}×{img.shape[0]}, "
            f"brightness={g.mean():.1f}, std={g.std():.1f}, "
            f"R={img[:,:,0].mean():.1f} G={img[:,:,1].mean():.1f} B={img[:,:,2].mean():.1f}, "
            f"sharpness={cv2.Laplacian(g,cv2.CV_64F).var():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
#  GROQ AI
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM = """You are AquaEnhance AI, an expert in underwater image processing, computer vision, and the physics of light in water.
You help users understand:
- What degradation is present in their underwater image (haze, color cast, blur, noise, vignetting)
- Which pipeline stages improved the image and why
- Parameter tuning recommendations with specific values
- The physics behind each technique (Beer-Lambert, DFT, Wiener, Retinex, etc.)
- Metric interpretation (PSNR dB, SSIM 0-1, sharpness, colorfulness)
Keep answers focused and technical. No markdown headers or bullet lists — write in plain paragraphs."""


def groq_query(client, messages: list) -> str:
    try:
        r = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=600,
            temperature=0.35,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq error: {e}"


if "chat" not in st.session_state:
    st.session_state.chat = []
if "ctx" not in st.session_state:
    st.session_state.ctx = ""


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 AquaEnhance Pro")
    st.markdown("---")

    st.markdown("**Groq API Key**")
    groq_key = st.text_input("API key", type="password",
                              value=os.environ.get("GROQ_API_KEY",""),
                              placeholder="gsk_...", label_visibility="collapsed")
    if groq_key and GROQ_OK:
        st.markdown('<span class="ok">Groq ready</span>', unsafe_allow_html=True)
    elif not GROQ_OK:
        st.markdown('<span class="warn">pip install groq</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model Checkpoint**")
    ckpt_file = st.file_uploader("best_model.pth", type=["pth","pt"],
                                  label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Pipeline**")
    use_dl  = st.checkbox("DL UNet ResNet34",       value=True)
    physics = st.checkbox("01 Physics CC",           value=True)
    wb      = st.checkbox("02 White Balance",        value=True)
    median  = st.checkbox("03 Median Filter",        value=True)
    gaussian= st.checkbox("04 Gaussian Filter",      value=True)
    clahe   = st.checkbox("05 CLAHE",                value=True)
    dcp     = st.checkbox("06 Dark Channel",         value=True)
    homo    = st.checkbox("07 Homomorphic",          value=True)
    butter  = st.checkbox("08 DFT Butterworth",      value=True)
    wiener  = st.checkbox("09 Wiener",               value=True)
    nlm     = st.checkbox("10 NLM Denoise",          value=True)
    vign    = st.checkbox("11 Vignette",             value=True)
    retinex = st.checkbox("12 Retinex",              value=True)
    bilat   = st.checkbox("13 Bilateral+Guided",     value=True)
    lap     = st.checkbox("14 Laplacian",            value=True)
    usm     = st.checkbox("15 Unsharp Mask",         value=True)
    hb      = st.checkbox("16 High-Boost",           value=True)
    sobel   = st.checkbox("17 Sobel",                value=True)
    gamma   = st.checkbox("18 Gamma",                value=True)
    sat     = st.checkbox("19 Saturation",           value=True)
    stretch = st.checkbox("20 Color Stretch",        value=True)

    st.markdown("---")
    alpha = st.slider("Fusion α  (DL ← → Formula)", 0.0, 1.0, 0.50, 0.05)

    with st.expander("Parameters"):
        med_k  = st.select_slider("Median k",    [3,5,7], 3)
        g_s    = st.slider("Gaussian σ",          0.3,2.0,0.8,0.1)
        cl_c   = st.slider("CLAHE clip",          1.0,4.0,2.0,0.25)
        cl_t   = st.select_slider("CLAHE tile",   [4,8,16],8)
        dcp_p  = st.slider("DCP patch",           5,21,15,2)
        dcp_o  = st.slider("DCP omega",           0.7,0.95,0.92,0.01)
        h_gl   = st.slider("Homo γ_L",            0.25,0.9,0.5,0.05)
        h_gh   = st.slider("Homo γ_H",            1.1,2.0,1.5,0.05)
        h_d0   = st.slider("Homo D₀",             10.0,60.0,30.0,2.0)
        b_hp   = st.slider("DFT HP D₀",           10.0,50.0,25.0,1.0)
        b_lp   = st.slider("DFT LP D₀",           40.0,120.0,70.0,2.0)
        w_k    = st.slider("Wiener K",            0.001,0.05,0.005,0.001,format="%.3f")
        w_s    = st.slider("Wiener PSF σ",         0.5,3.0,1.2,0.1)
        nlm_h  = st.slider("NLM h",               2.0,10.0,5.0,0.5)
        v_s    = st.slider("Vignette σ frac",      0.3,0.9,0.60,0.05)
        bil_c  = st.slider("Bilateral σ_c",        5.0,40.0,20.0,1.0)
        bil_s  = st.slider("Bilateral σ_s",        5.0,40.0,20.0,1.0)
        lap_a  = st.slider("Laplacian α",          0.2,1.0,0.5,0.05)
        u_s    = st.slider("USM σ",                0.5,2.0,0.8,0.1)
        u_str  = st.slider("USM strength",         0.5,2.0,1.2,0.1)
        u_thr  = st.slider("USM threshold",        1,8,3,1)
        hb_A   = st.slider("High-Boost A",         1.2,2.2,1.6,0.05)
        sb_str = st.slider("Sobel strength",       0.1,0.5,0.25,0.05)
        gam    = st.slider("Gamma γ",              0.5,1.2,0.82,0.02)
        sat_f  = st.slider("Saturation ×",         1.0,1.6,1.25,0.05)
        str_lo = st.slider("Stretch lo %",         0.1,2.0,0.5,0.1)
        str_hi = st.slider("Stretch hi %",        97.0,99.9,99.5,0.1)

    st.markdown("---")
    export_fmt = st.selectbox("Export", ["PNG","JPEG"])
    dev = "CUDA" if torch.cuda.is_available() else "CPU"
    st.markdown(f'<span class="{"ok" if dev=="CUDA" else "warn"}">{dev}</span>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eye">Production · Computer Vision · Underwater Reconstruction</div>
  <div class="hdr-title">Aqua<span>Enhance</span> Pro</div>
  <div class="hdr-desc">
    20-stage formula pipeline · Physics + Denoising + Dehazing + Deblurring + Sharpening · DL UNet Fusion · Groq AI Assistant
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ribbon">
  <div class="rb on"><div class="rb-n">DL</div><div class="rb-l">UNet</div></div>
  <div class="rb on"><div class="rb-n">01</div><div class="rb-l">Physics</div></div>
  <div class="rb on"><div class="rb-n">02</div><div class="rb-l">WB</div></div>
  <div class="rb on"><div class="rb-n">03</div><div class="rb-l">Median</div></div>
  <div class="rb on"><div class="rb-n">04</div><div class="rb-l">Gauss</div></div>
  <div class="rb on"><div class="rb-n">05</div><div class="rb-l">CLAHE</div></div>
  <div class="rb on"><div class="rb-n">06</div><div class="rb-l">DCP</div></div>
  <div class="rb on"><div class="rb-n">07</div><div class="rb-l">Homo</div></div>
  <div class="rb on"><div class="rb-n">08</div><div class="rb-l">DFT</div></div>
  <div class="rb on"><div class="rb-n">09</div><div class="rb-l">Wiener</div></div>
  <div class="rb on"><div class="rb-n">10</div><div class="rb-l">NLM</div></div>
  <div class="rb on"><div class="rb-n">11</div><div class="rb-l">Vign.</div></div>
  <div class="rb on"><div class="rb-n">12</div><div class="rb-l">Retinex</div></div>
  <div class="rb on"><div class="rb-n">13</div><div class="rb-l">Bilat.</div></div>
  <div class="rb on"><div class="rb-n">14</div><div class="rb-l">Lap.</div></div>
  <div class="rb on"><div class="rb-n">15</div><div class="rb-l">USM</div></div>
  <div class="rb on"><div class="rb-n">16</div><div class="rb-l">HiBoost</div></div>
  <div class="rb on"><div class="rb-n">17</div><div class="rb-l">Sobel</div></div>
  <div class="rb on"><div class="rb-n">18</div><div class="rb-l">Gamma</div></div>
  <div class="rb on"><div class="rb-n">19</div><div class="rb-l">Sat.</div></div>
  <div class="rb on"><div class="rb-n">20</div><div class="rb-l">Stretch</div></div>
  <div class="rb final"><div class="rb-n">→</div><div class="rb-l">Fusion</div></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FORMULA REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📐 Complete Formula Reference", expanded=False):
    st.markdown("""
<div class="fwrap">
  <div class="fc key"><div class="ft">01 · Physics Beer-Lambert</div>
    <div class="fd">Wavelength-dependent water attenuation model.</div>
    <code class="fm">J(c)=I(c)/exp(-β_c·d),  β=[0.30,0.06,0.02]
d_est = 1-min_c(I(c)/255)</code></div>
  <div class="fc key"><div class="ft">02 · White Balance SoG+GW</div>
    <div class="fd">Minkowski p=6 + Gray World fusion, clip [0.5,3.0].</div>
    <code class="fm">scale=0.6·(μ/||I||_p)+0.4·(μ/μ_c)</code></div>
  <div class="fc"><div class="ft">03 · Median Filter</div>
    <div class="fd">Non-linear. Removes salt-and-pepper/impulse noise.</div>
    <code class="fm">I_out(x,y)=median{I(x+i,y+j)|(i,j)∈W_k}</code></div>
  <div class="fc"><div class="ft">04 · Gaussian Filter</div>
    <div class="fd">Convolution with Gaussian kernel. Smooths additive noise.</div>
    <code class="fm">I_out=I*G_σ, G=exp(-(x²+y²)/2σ²)/2πσ²</code></div>
  <div class="fc"><div class="ft">05 · CLAHE (LAB)</div>
    <div class="fd">Local HE per tile on L channel, clip=2.0.</div>
    <code class="fm">h_clip(i)=min(h(i),clip_limit), excess redistributed</code></div>
  <div class="fc key"><div class="ft">06 · Dark Channel Prior</div>
    <div class="fd">He 2009. Guided-filter refined transmission map.</div>
    <code class="fm">J_dark=min_c min_Ω(I),  t=1-ω·J_dark/A
J=(I-A)/max(t,t_min)+A</code></div>
  <div class="fc key"><div class="ft">07 · Homomorphic Filter</div>
    <div class="fd">Frequency-domain illumination/reflectance separation.</div>
    <code class="fm">H(D)=(γH-γL)·[1-exp(-c·D²/D₀²)]+γL
z=log(I+1) → IDFT(H·DFT(z)) → exp-1</code></div>
  <div class="fc"><div class="ft">08 · DFT Butterworth Bandpass</div>
    <div class="fd">HP removes haze, LP removes noise, per channel.</div>
    <code class="fm">H_HP=1-1/[1+(D/D0hp)^2n]
H_LP=1/[1+(D/D0lp)^2n],  H=H_HP·H_LP</code></div>
  <div class="fc"><div class="ft">09 · Wiener Deconvolution</div>
    <div class="fd">Optimal linear deblur, PSF=Gaussian(σ).</div>
    <code class="fm">W=H*/(|H|²+K),  Ĵ=IDFT(W·DFT(I))</code></div>
  <div class="fc key"><div class="ft">10 · NLM Denoising</div>
    <div class="fd">Patch-based optimal. Best texture preservation.</div>
    <code class="fm">NL[u](x)=Σ_y w(x,y)·u(y)/Σ w
w=exp(-||B_x-B_y||²/h²)</code></div>
  <div class="fc"><div class="ft">11 · Vignette Correction</div>
    <div class="fd">Gaussian radial model. Cap at 2.2× prevents noise.</div>
    <code class="fm">G=exp(-r²/2σ²), I_out=I/max(G,0.4), clip[1,2.2]</code></div>
  <div class="fc key"><div class="ft">12 · MSRCP Retinex</div>
    <div class="fd">Multi-scale illumination/reflectance. Preserves chromaticity.</div>
    <code class="fm">SSR_s=log(I+1)-log(G_σ*I+1)
MSR=Σ(1/S)·SSR_s,  CP=MSR·(I/I_gray)</code></div>
  <div class="fc"><div class="ft">13 · Bilateral + Guided</div>
    <div class="fd">2-pass edge-preserving. Bilateral then guided refine.</div>
    <code class="fm">w∝exp(-|Δp|²/2σ_s²)·exp(-|ΔI|²/2σ_c²)
guided: q=mean_a·I+mean_b</code></div>
  <div class="fc"><div class="ft">14 · Laplacian Sharpening</div>
    <div class="fd">2nd-order derivative. All-direction edge boost.</div>
    <code class="fm">∇²I≈[[0,-1,0],[-1,4,-1],[0,-1,0]]*I
I_out=I+α·∇²I</code></div>
  <div class="fc key"><div class="ft">15 · Threshold Unsharp Mask</div>
    <div class="fd">Only sharpens |mask|>threshold — no flat-area noise.</div>
    <code class="fm">M=I-G_σ(I),  I_out=I+λ·M  if |M|>thr</code></div>
  <div class="fc"><div class="ft">16 · High-Boost Filter</div>
    <div class="fd">Amplified high-frequency boost. Stronger than USM.</div>
    <code class="fm">I_out=A·I-LP(I)=I+(A-1)·HP(I),  A=1.6</code></div>
  <div class="fc"><div class="ft">17 · Sobel Edge Enhancement</div>
    <div class="fd">Directional gradient magnitude added to image.</div>
    <code class="fm">G=√(Gx²+Gy²),  I_out=I+0.25·G</code></div>
  <div class="fc"><div class="ft">18 · Gamma Correction</div>
    <div class="fd">Power-law brightness via LUT. γ=0.82 lifts darks.</div>
    <code class="fm">I_out=255·(I/255)^γ</code></div>
  <div class="fc"><div class="ft">19 · Saturation Boost</div>
    <div class="fd">HSV chroma ×1.25. Restores colour lost to absorption.</div>
    <code class="fm">S_out=clip(S_in·factor, 0, 255)  [HSV space]</code></div>
  <div class="fc"><div class="ft">20 · Color Balance Stretch</div>
    <div class="fd">Per-channel percentile. Removes residual bias.</div>
    <code class="fm">I_out[c]=(I[c]-p0.5%)/(p99.5%-p0.5%)·255</code></div>
  <div class="fc key"><div class="ft">→ · LAB Weighted Fusion</div>
    <div class="fd">Perceptual blend. Post bilateral removes seams.</div>
    <code class="fm">F=α·LAB(DL)+(1-α)·LAB(formula) → RGB
post: bilateral(σ_c=15,σ_s=15)</code></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
model = None; meta = {}
if ckpt_file:
    tmp = f"/tmp/{ckpt_file.name}"
    open(tmp,"wb").write(ckpt_file.read())
    with st.spinner("Loading model ..."):
        try:
            model, meta = load_model(tmp)
            st.markdown(f'<span class="ok">Model loaded · Epoch {meta.get("epoch","?")}</span>',
                        unsafe_allow_html=True)
            if meta.get("val_loss"):
                st.caption(f'val_loss={meta["val_loss"]:.4f}  SSIM={meta.get("val_ssim",0):.4f}  PSNR={meta.get("val_psnr",0):.2f} dB')
        except Exception as e:
            st.error(f"Model load failed: {e}")
else:
    st.info("Upload `best_model.pth` in the sidebar. Pipeline runs formula-only without it.")


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE UPLOAD & PROCESS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Input Image</div>', unsafe_allow_html=True)
up = st.file_uploader("Drop an underwater image", type=["jpg","jpeg","png","bmp"],
                       label_visibility="collapsed")

if up:
    img_np = np.array(Image.open(up).convert("RGB"))

    cfg = dict(
        dl=use_dl, physics=physics, wb=wb, median=median, gaussian=gaussian,
        clahe=clahe, dcp=dcp, homo=homo, butter=butter, wiener=wiener,
        nlm=nlm, vign=vign, retinex=retinex, bilat=bilat,
        lap=lap, usm=usm, hb=hb, sobel=sobel, gamma=gamma, sat=sat, stretch=stretch,
        med_k=med_k, g_s=g_s, cl_c=cl_c, cl_t=cl_t, dcp_p=dcp_p, dcp_o=dcp_o,
        h_gl=h_gl, h_gh=h_gh, h_d0=h_d0, b_hp=b_hp, b_lp=b_lp,
        w_k=w_k, w_s=w_s, nlm_h=nlm_h, v_s=v_s, bil_c=bil_c, bil_s=bil_s,
        lap_a=lap_a, u_s=u_s, u_str=u_str, u_thr=u_thr,
        hb_A=hb_A, sb_str=sb_str, gam=gam, sat_f=sat_f,
        str_lo=str_lo, str_hi=str_hi, alpha=alpha,
    )

    pb = st.progress(0); lbl = st.empty()
    def on_prog(frac, name):
        pb.progress(frac)
        lbl.markdown(f'<span style="font-family:\'DM Mono\',monospace;font-size:.58rem;color:#1e6a8a;">↳ {name}</span>',
                     unsafe_allow_html=True)

    with st.spinner("Running reconstruction pipeline ..."):
        t0    = time.perf_counter()
        steps = run_pipeline(img_np, model, cfg, on_prog=on_prog)
        elapsed = (time.perf_counter()-t0)*1000

    pb.empty(); lbl.empty()
    final = steps["FINAL"]
    m     = compute_metrics(img_np, final)

    # Update Groq context
    st.session_state.ctx = (
        f"Input:  {img_summary(img_np)}\n"
        f"Output: {img_summary(final)}\n"
        f"Metrics: PSNR={m['psnr']:.2f}dB SSIM={m['ssim']:.4f} "
        f"Sharpness={m['sharpness']:.1f} Colorfulness={m['colorfulness']:.1f}\n"
        f"Active stages: {[k for k,v in cfg.items() if isinstance(v,bool) and v]}"
    )

    # Metrics
    st.markdown(f"""
<div class="mrow">
  <div class="mc best"><div class="ml">PSNR</div>
    <div class="mv">{m['psnr']:.2f}<span class="mu">dB</span></div></div>
  <div class="mc best"><div class="ml">SSIM</div>
    <div class="mv">{m['ssim']:.4f}</div></div>
  <div class="mc hi"><div class="ml">Sharpness</div>
    <div class="mv">{m['sharpness']:.1f}</div></div>
  <div class="mc hi"><div class="ml">Colorfulness</div>
    <div class="mv">{m['colorfulness']:.1f}</div></div>
  <div class="mc"><div class="ml">Time</div>
    <div class="mv">{elapsed:.0f}<span class="mu">ms</span></div></div>
  <div class="mc"><div class="ml">DL</div>
    <div class="mv">{steps.get('_dl_ms',0):.0f}<span class="mu">ms</span></div></div>
</div>
""", unsafe_allow_html=True)

    # Main result
    st.markdown('<div class="sec">Reconstruction Result</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="pl">Original · Degraded</div>', unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)
    with c2:
        st.markdown('<div class="pl out">Reconstructed · Final Output</div>', unsafe_allow_html=True)
        st.image(final, use_container_width=True)

    st.download_button("⬇  Download Reconstructed Image",
                        data=to_bytes(final, export_fmt),
                        file_name=f"aquaenhance_{up.name.rsplit('.',1)[0]}.{export_fmt.lower()}",
                        mime=f"image/{'jpeg' if export_fmt=='JPEG' else 'png'}")

    # Stage viewer
    with st.expander("Stage-by-Stage Viewer", expanded=False):
        stage_keys = [k for k in steps if not k.startswith("_") and k != "FINAL"]
        for i in range(0, len(stage_keys), 4):
            row = stage_keys[i:i+4]; cols = st.columns(len(row))
            for col, key in zip(cols, row):
                with col:
                    st.markdown(f'<div class="pl">{key}</div>', unsafe_allow_html=True)
                    st.image(steps[key], use_container_width=True)

    # DFT Spectrum
    with st.expander("DFT Magnitude Spectrum", expanded=False):
        def spectrum(im):
            g = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)
            s = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
            return (s/s.max()*255).astype(np.uint8)
        fig, ax = plt.subplots(1,3,figsize=(14,3.5),facecolor="#07080a")
        for a,im,ttl,c in zip(ax,
            [img_np, steps.get("08·DFT Butterworth",img_np), final],
            ["Input","Post-DFT","Final"],["#7a9db5","#1aa8d8","#1dd4a0"]):
            a.imshow(spectrum(im),cmap="plasma"); a.set_title(ttl,color=c,fontsize=9); a.axis("off")
            a.set_facecolor("#0a0d10")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Histogram
    with st.expander("RGB Channel Histogram", expanded=False):
        fig, axes = plt.subplots(1,2,figsize=(12,3),facecolor="#07080a")
        for ax in axes:
            ax.set_facecolor("#0a0d10")
            for sp in ax.spines.values(): sp.set_color("#151e26")
        for i,(c,n) in enumerate(zip(["#e05252","#52e09a","#5282e0"],["R","G","B"])):
            axes[0].plot(cv2.calcHist([img_np],[i],None,[256],[0,256]).flatten(),color=c,lw=1,alpha=.85,label=n)
            axes[1].plot(cv2.calcHist([final], [i],None,[256],[0,256]).flatten(),color=c,lw=1,alpha=.85)
        axes[0].set_title("Input",         color="#7a9db5",fontsize=9)
        axes[1].set_title("Reconstructed", color="#1dd4a0",fontsize=9)
        axes[0].legend(facecolor="#0a0d10",labelcolor="#7a9db5",fontsize=8)
        for ax in axes: ax.set_xlim(0,255); ax.set_ylim(bottom=0); ax.tick_params(colors="#1e6a8a",labelsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Difference map
    with st.expander("Difference Map  (|Output − Input| × 5)", expanded=False):
        orig_r = cv2.resize(img_np,(final.shape[1],final.shape[0]))
        diff   = np.clip(np.abs(final.astype(np.int16)-orig_r.astype(np.int16))*5, 0,255).astype(np.uint8)
        st.image(diff, use_container_width=True)
        st.caption("Brighter pixels = stronger correction applied.")


# ─────────────────────────────────────────────────────────────────────────────
#  GROQ AI ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Groq AI Assistant</div>', unsafe_allow_html=True)

if not GROQ_OK:
    st.warning("Run `pip install groq` then restart.")
elif not groq_key:
    st.info("Enter your Groq API key in the sidebar to enable the AI assistant.")
else:
    groq_client = Groq(api_key=groq_key)
    ctx_note    = ("\n\nImage context:\n" + st.session_state.ctx) if st.session_state.ctx else ""

    # Quick action buttons
    if up and st.session_state.ctx:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Analyse degradation"):
                q = "Looking at the input image statistics, identify and explain all degradation present: color cast, haze, blur, noise, vignetting. Be specific with numbers."
                st.session_state.chat.append({"role":"user","content":q})
                msgs = [{"role":"system","content":SYSTEM+ctx_note}]+st.session_state.chat
                st.session_state.chat.append({"role":"assistant","content":groq_query(groq_client,msgs)})
                st.rerun()
        with b2:
            if st.button("Explain improvements"):
                q = "Compare input and output statistics. Explain what changed, which pipeline stages caused those changes, and why each formula works for underwater imagery."
                st.session_state.chat.append({"role":"user","content":q})
                msgs = [{"role":"system","content":SYSTEM+ctx_note}]+st.session_state.chat
                st.session_state.chat.append({"role":"assistant","content":groq_query(groq_client,msgs)})
                st.rerun()
        with b3:
            if st.button("Suggest parameters"):
                q = "Based on the current metrics, suggest specific parameter changes (with exact slider values) that would improve the reconstruction quality further."
                st.session_state.chat.append({"role":"user","content":q})
                msgs = [{"role":"system","content":SYSTEM+ctx_note}]+st.session_state.chat
                st.session_state.chat.append({"role":"assistant","content":groq_query(groq_client,msgs)})
                st.rerun()

    # Chat history
    if st.session_state.chat:
        html = "<div class='chat-wrap'>"
        for msg in st.session_state.chat:
            cls = "msg-u" if msg["role"]=="user" else "msg-a"
            txt = msg["content"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            html += f"<div class='{cls}'>{txt}</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        if st.button("Clear chat"):
            st.session_state.chat = []; st.rerun()

    # Text input
    with st.form("chat_form", clear_on_submit=True):
        user_in = st.text_area(
            "Ask about underwater image processing ...",
            height=75,
            placeholder="e.g. Why is my image bluish? How does Wiener filter work? What does SSIM measure?",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("Send")

    if sent and user_in.strip():
        st.session_state.chat.append({"role":"user","content":user_in.strip()})
        msgs = [{"role":"system","content":SYSTEM+ctx_note}] + st.session_state.chat[-12:]
        with st.spinner("Groq thinking ..."):
            ans = groq_query(groq_client, msgs)
        st.session_state.chat.append({"role":"assistant","content":ans})
        st.rerun()

    if not up:
        st.markdown("""
<div style="border:1px dashed #1e3a4e;background:#0a0d10;border-radius:6px;
padding:3rem 2rem;text-align:center;margin:1rem 0;">
  <div style="font-size:2rem;margin-bottom:.6rem">🌊</div>
  <div style="font-family:'DM Mono',monospace;font-size:.62rem;
  color:#1e6a8a;letter-spacing:3px;text-transform:uppercase;">Upload an underwater image to begin</div>
  <div style="font-size:.75rem;color:#1a3040;margin-top:.4rem;">
    20-stage pipeline · DL UNet · Groq AI Assistant
  </div>
</div>""", unsafe_allow_html=True)