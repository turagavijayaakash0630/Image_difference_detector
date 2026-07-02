# 🔍 AI-Based Image Difference Detection

A modular **Streamlit** application that detects, visualizes, and summarizes differences between two images using computer-vision techniques.

---

## 📁 Project Structure

```
Image_difference_detector/
│
├── app.py                        # Streamlit entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── preprocessing/                # Image loading & preprocessing
│   ├── __init__.py
│   └── image_loader.py
│
├── detection/                    # Difference detection algorithms
│   ├── __init__.py
│   └── difference.py
│
├── visualization/                # Heatmaps, overlays & comparisons
│   ├── __init__.py
│   └── visualize.py
│
├── summarization/                # Natural-language change summaries
│   ├── __init__.py
│   └── summary.py
│
├── statistics/                   # Quantitative diff metrics
│   ├── __init__.py
│   └── stats.py
│
└── utils/                        # Shared helpers & converters
    ├── __init__.py
    └── helpers.py
```

---

## ⚙️ Prerequisites

- **Python 3.12+**
- **pip** (comes with Python)

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone <repo-url>
cd Image_difference_detector

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 🧩 Module Overview

| Module            | Purpose                                              |
| ----------------- | ---------------------------------------------------- |
| `preprocessing`   | Load, resize, and convert images                     |
| `detection`       | SSIM, pixel-diff, and contour-based change detection |
| `visualization`   | Heatmaps, bounding boxes, side-by-side views         |
| `summarization`   | Rule-based natural-language summaries                 |
| `statistics`      | Quantitative metrics (diff %, region stats)           |
| `utils`           | Format converters and shared helpers                  |

---

## 📝 License

This project is provided for educational purposes.
