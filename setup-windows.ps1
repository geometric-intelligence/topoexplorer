# TopoExplorer setup for Windows
# Requires Python 3.11 (topobench pins >=3.11, <3.12)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Py = "py -3.11"

Write-Host "Creating virtual environment with Python 3.11..."
& $Py -m venv $Venv

$Python = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

& $Python -m pip install --upgrade pip

Write-Host "Installing torch and streamlit..."
& $Pip install torch==2.3.0+cpu streamlit `
    --extra-index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing topobench (without deps; PyTDC/tiledbsoma fails on Windows)..."
& $Pip install "topobench @ git+https://github.com/geometric-intelligence/topobench.git" --no-deps `
    --extra-index-url https://download.pytorch.org/whl/cpu `
    --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html

Write-Host "Installing topobench runtime dependencies..."
& $Pip install "setuptools>=69,<82" tqdm scipy scikit-learn matplotlib decorator `
    "hypernetx<2.0.0" trimesh spharapy "hydra-core==1.3.2" "hydra-colorlog==1.2.0" `
    "hydra-optuna-sweeper==1.2.0" "yacs==0.1.8" wandb tensorboard "einops==0.7.0" `
    tabulate ipykernel notebook jupyterlab rich ogb rootutils "graph-universe==0.1.2" `
    "lightning==2.4.0" torch-scatter torch-sparse torch-cluster torch-geometric `
    omegaconf pyyaml `
    "topomodelx @ git+https://github.com/pyt-team/TopoModelX.git" `
    "toponetx @ git+https://github.com/pyt-team/TopoNetX.git@c378925" `
    --extra-index-url https://download.pytorch.org/whl/cpu `
    --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html

Write-Host ""
Write-Host "Setup complete. Activate and run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  streamlit run topoexplorer\neighborhood_explorer_app.py"
