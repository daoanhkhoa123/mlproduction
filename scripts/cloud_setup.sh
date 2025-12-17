set -e

# Install uv and sync basic dependencies

pip install --upgrade uv
uv sync

# Install optional dependencies

ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
echo "ENV_FILE=$ENV_FILE"

if [ -f "$ENV_FILE" ]; then
    CLOUD_DEPS=$(grep '^CLOUD_DEPENDENCIES' "$ENV_FILE" | cut -d= -f2- | xargs)
else
    echo ".env file not found!"
    exit 1
fi

IFS=',' read -ra DEP_ARRAY <<< "$CLOUD_DEPS"

DEPS=""
for dep in "${DEP_ARRAY[@]}"; do
    dep=$(echo "$dep" | xargs)

    echo "Runing: uv add $dep"
    uv add $dep
done

# Set up custom ipynb kernel

PROJECT_NAME="mlproduction"
VENV_DIR=".venv"
KERNEL_NAME="mlproduction"
DISPLAY_NAME="Python (mlproduction)"

uv run python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${DISPLAY_NAME}"

echo "Kernel registered successfully"
echo "Available kernels:"
jupyter kernelspec list

# Activate kernel
KERNEL_NAME="mlproduction"
DISPLAY_NAME="Python (mlproduction)"

uv run python - <<'EOF'
import json
import glob
from pathlib import Path

KERNEL_NAME = "mlproduction"
DISPLAY_NAME = "Python (mlproduction)"

for nb in glob.glob("**/*.ipynb", recursive=True):
    if ".ipynb_checkpoints" in nb:
        continue

    path = Path(nb)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue

    metadata = data.setdefault("metadata", {})

    metadata["kernelspec"] = {
        "name": KERNEL_NAME,
        "display_name": DISPLAY_NAME,
        "language": "python",
    }

    metadata.setdefault("language_info", {"name": "python"})

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
EOF
