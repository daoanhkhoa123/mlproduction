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

# Path environment fix

KERNEL_DIR="$(jupyter kernelspec list --json | python -c "
import json,sys
data=json.load(sys.stdin)
print(data['kernelspecs']['mlproduction']['resource_dir'])
")"

cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "$(pwd)/.venv/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (mlproduction)",
  "language": "python",
  "env": {
    "PATH": "$(pwd)/.venv/bin:\${PATH}"
  }
}
EOF
