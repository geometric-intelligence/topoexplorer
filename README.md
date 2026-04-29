<h2 align="center">
  <img src="resources/logo.jpg" width="800">
</h2>

<h3 align="center">
    A Comprehensive Benchmark Suite for Topological Deep Learning
</h3>

<p align="center">
Assess how your model compares against state-of-the-art topological neural networks.
</p>

<div align="center">

[![Lint](https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml/badge.svg)](https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml)
[![Test](https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml)
[![Codecov](https://codecov.io/gh/geometric-intelligence/TopoBench/branch/main/graph/badge.svg)](https://app.codecov.io/gh/geometric-intelligence/TopoBench)
[![Docs](https://img.shields.io/badge/docs-website-brightgreen)](https://geometric-intelligence.github.io/topobench/index.html)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![license](https://badgen.net/github/license/geometric-intelligence/TopoBench?color=green)](https://github.com/geometric-intelligence/TopoBench/blob/main/LICENSE)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/geometric-intelligenceworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw)


</div>

<p align="center">
  <a href="#jigsaw-get-started">Get Started</a>
</p>


---

## :jigsaw: Get Started

### Topobench setup

#### 🚀 Quick Install (Recommended)

TopoBench now uses [**uv**](https://docs.astral.sh/uv/), an extremely fast Python package manager and resolver. This allows for nearly instantaneous environment setup and reproducible builds.

1.  [**Install uv**](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

2.  **Clone and Navigate**:
    ```bash
    git clone git@github.com:geometric-intelligence/topobench.git
    cd TopoBench
    ```

3.  **Initialize Environment**:
    Use our centralized setup script to handle Python 3.11 virtualization and specialized hardware (CUDA) mapping.
    ```bash
    # Usage: source uv_env_setup.sh [cpu|cu118|cu121]
    source uv_env_setup.sh cpu
    ```
    *This script performs the following:*
    * Creates a `.venv` using Python 3.11.
    * Dynamically configures `pyproject.toml` to point to the correct **PyTorch** and **PyG** (PyTorch Geometric) wheels for your platform.
    * Generates a precise `uv.lock` file and syncs all dependencies.

---

#### 🛠️ Manual Environment Setup

If you prefer to manage the environment manually or are integrating into an existing workflow:

```bash
# Create a virtual environment with strict versioning
uv venv --python 3.11
source .venv/bin/activate

# Sync dependencies including all extras (dev, test, and doc)
uv sync --all-extras
```

🚄 Run Training Pipeline
Once the environment is active, you can launch the TopoBench pipeline:
```bash
# Using the activated virtual environment
python -m topobench 

# Or execute directly via uv without manual activation
uv run python -m topobench
```

✅ Verify Installation
You can verify that the correct versions of Torch and CUDA are detected by running:
```bash
python -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda}')"
```

---

### Visualization app setup

The interactive neighborhood explorer is a **Streamlit** app under `analysis/`. Use the **same virtual environment** you created for TopoBench (so `topobench` imports and dataset configs resolve correctly).

1. **Install Streamlit** (not included in the default dependency set):

   ```bash
   uv pip install streamlit
   ```

   Or, with the venv activated: `pip install streamlit`.

2. **Run from the repository root** (the directory that contains `configs/`, `datasets/`, `topobench/`, and `analysis/`):

   ```bash
   streamlit run analysis/neighborhood_explorer_app.py
   ```

   Open the local URL Streamlit prints (typically `http://localhost:8501`).

The app loads dataset YAMLs from `configs/dataset/...` and uses TopoBench loaders and transforms; keep your working directory aligned with the cloned repo layout.
