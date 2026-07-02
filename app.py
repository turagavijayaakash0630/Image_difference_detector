"""
AI-Based Image Difference Detection — Main Streamlit Application.

Run with:
    streamlit run app.py
"""
from preprocessing import resize_to_match, to_grayscale
from detection.difference import (
    compute_ssim,
    compute_pixel_diff,
    find_contours,
)
from visualization.visualize import (
    create_heatmap,
    draw_bounding_boxes,
    side_by_side,
)
from statistics.stats import compute_statistics
from summarization.summary import generate_summary


import streamlit as st
from PIL import Image

from preprocessing import load_image, bgr_to_rgb

# ---------------------------------------------------------------------------
# Page configuration & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Image Difference Detector",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ─────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .main-header h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0;
        opacity: 0.90;
        font-size: 1.05rem;
    }

    /* ── Upload cards ───────────────────────────────────────────── */
    .upload-card {
        background: #f8f9fc;
        border: 2px dashed #d0d5e4;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: border-color 0.25s, box-shadow 0.25s;
    }
    .upload-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.12);
    }
    .upload-label {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
        color: #333;
    }

    /* ── Section titles ─────────────────────────────────────────── */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }

    /* ── Placeholder boxes ──────────────────────────────────────── */
    .placeholder-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf4 100%);
        border: 1px solid #d0d5e4;
        border-radius: 14px;
        padding: 3rem 1.5rem;
        text-align: center;
        color: #8a8fa8;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .placeholder-box .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    /* ── Compare button ─────────────────────────────────────────── */
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(102, 126, 234, 0.35);
    }

    /* ── Sidebar tweaks ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #f4f6fb;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #444;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>🔍 AI-Based Image Difference Detection</h1>
        <p>Upload two images, tune detection settings, and instantly see
        what changed — powered by computer vision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — Detection controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Detection Settings")
    st.markdown("---")

    detection_method: str = st.radio(
        "Detection Method",
        options=["SSIM", "Pixel Difference"],
        index=0,
        help=(
            "**SSIM** — Structural Similarity Index, great for perceptual "
            "changes.\n\n**Pixel Difference** — Raw absolute difference, "
            "best for exact pixel-level edits."
        ),
    )

    sensitivity: int = st.slider(
        "Sensitivity",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="Higher values detect smaller, subtler differences.",
    )

    st.markdown("---")
    st.caption("Built with Streamlit • OpenCV • scikit-image")

# ---------------------------------------------------------------------------
# Image upload area
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">📤 Upload Images</div>', unsafe_allow_html=True)

col_ref, col_cmp = st.columns(2, gap="large")

with col_ref:
    st.markdown(
        '<div class="upload-card"><div class="upload-label">Reference Image</div></div>',
        unsafe_allow_html=True,
    )
    uploaded_ref = st.file_uploader(
        "Upload reference image",
        type=["jpg", "jpeg", "png"],
        key="ref_upload",
        label_visibility="collapsed",
    )

with col_cmp:
    st.markdown(
        '<div class="upload-card"><div class="upload-label">Comparison Image</div></div>',
        unsafe_allow_html=True,
    )
    uploaded_cmp = st.file_uploader(
        "Upload comparison image",
        type=["jpg", "jpeg", "png"],
        key="cmp_upload",
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Side-by-side preview
# ---------------------------------------------------------------------------

if uploaded_ref and uploaded_cmp:
    # Decode via our preprocessing module so images are validated early.
    try:
        ref_bgr = load_image(uploaded_ref)
        uploaded_cmp.seek(0)  # reset pointer after possible previous read
        uploaded_ref.seek(0)
        ref_bgr = load_image(uploaded_ref)
        cmp_bgr = load_image(uploaded_cmp)

        ref_rgb = bgr_to_rgb(ref_bgr)
        cmp_rgb = bgr_to_rgb(cmp_bgr)

        st.markdown(
            '<div class="section-title">🖼️ Uploaded Images</div>',
            unsafe_allow_html=True,
        )

        prev_col1, prev_col2 = st.columns(2, gap="large")
        with prev_col1:
            st.image(ref_rgb, caption="Reference Image", use_container_width=True)
        with prev_col2:
            st.image(cmp_rgb, caption="Comparison Image", use_container_width=True)

    except ValueError as exc:
        st.error(f"⚠️ {exc}")

elif uploaded_ref or uploaded_cmp:
    st.warning("Please upload **both** a reference and a comparison image.")

# ---------------------------------------------------------------------------
# Compare button
# ---------------------------------------------------------------------------

if uploaded_ref and uploaded_cmp:
    st.markdown("")  # spacer
    compare_clicked = st.button("🚀  Compare Images", use_container_width=True)
else:
    compare_clicked = False

# ---------------------------------------------------------------------------
# Result placeholders (not yet functional)
# ---------------------------------------------------------------------------

if compare_clicked:
    try:
        # Resize images to same size
        cmp_bgr = resize_to_match(ref_bgr, cmp_bgr)

        # Convert to grayscale
        gray1 = to_grayscale(ref_bgr)
        gray2 = to_grayscale(cmp_bgr)

        # Detection
        if detection_method == "SSIM":
            result = compute_ssim(
                gray1,
                gray2,
                sensitivity=sensitivity,
            )

            diff_map = result.diff_map
            mask = result.thresh_mask
            ssim_score = result.score

        else:
            diff_map, mask = compute_pixel_diff(
                gray1,
                gray2,
                sensitivity=sensitivity,
            )

            ssim_score = 0.0

        # Find contours
        contour_result = find_contours(mask)

        # Statistics
        stats = compute_statistics(
            ssim_score,
            mask,
            contour_result.contours,
        )

        # Visualizations
        heatmap = create_heatmap(diff_map, ref_bgr)

        boxed = draw_bounding_boxes(
            ref_bgr,
            contour_result.bounding_boxes,
        )

        comparison = side_by_side(
            ref_bgr,
            cmp_bgr,
        )

        # Summary
        summary = generate_summary(
            stats.ssim_score,
            stats.num_regions,
            stats.percentage_changed,
            stats.largest_region_area,
            contour_result.bounding_boxes,
            gray1.shape,
        )

        st.markdown(
            '<div class="section-title">📊 Results</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                comparison,
                caption="Side-by-Side Comparison",
                use_container_width=True,
            )

            st.image(
                heatmap,
                caption="Difference Heatmap",
                use_container_width=True,
            )

        with col2:
            st.image(
                boxed,
                caption="Detected Changes",
                use_container_width=True,
            )

        st.markdown("### 📈 Statistics")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "SSIM Score",
            f"{stats.ssim_score:.3f}",
        )

        m2.metric(
            "Changed %",
            f"{stats.percentage_changed:.2f}%",
        )

        m3.metric(
            "Regions",
            stats.num_regions,
        )

        m4, m5 = st.columns(2)

        m4.metric(
            "Largest Region",
            stats.largest_region_area,
        )

        m5.metric(
            "Changed Pixels",
            stats.changed_pixels,
        )

        st.markdown("### 🤖 AI Summary")

        st.success(summary)

    except Exception as e:
        st.error(f"Error: {e}")