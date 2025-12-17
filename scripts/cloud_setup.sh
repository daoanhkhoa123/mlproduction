set -e

# Install uv and basic dependencies

pip install --upgrade uv
uv pip compile pyproject.toml -o requirements.txt \
  --extra-index-url https://pypi.org/simple \
  --emit-index-url

uv pip install --system -r requirements.txt

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

    echo "Runing: uv pip install --system $dep"
    uv pip install --system $dep
done