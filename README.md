# TopoExplorer

A Streamlit-based visualization app for exploring topological deep learning neighborhoods, built on top of [TopoBench](https://github.com/geometric-intelligence/topobench).

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/geometric-intelligence/topoexplorer.git
cd topoexplorer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` (CPU build)
- `topobench` (from the [geometric-intelligence/topobench](https://github.com/geometric-intelligence/topobench) GitHub repository)
- `streamlit`

### 4. Run the app

```bash
streamlit run topoexplorer/neighborhood_explorer_app.py
```

The app will open in your browser at `http://localhost:8501`.
