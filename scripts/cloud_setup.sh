set -e

pip install --upgrade uv
uv sync

ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
echo "$ENV_FILE"

if [ -f "$ENV_FILE" ]; then
    # Extract dependencies string after CLOUD_DEPENDENCIES=
    CLOUD_DEPS=$(grep '^CLOUD_DEPENDENCIES' "$ENV_FILE" | cut -d= -f2- | xargs)
else
    echo ".env file not found!"
    exit 1
fi

# Split by comma, trim spaces, and join back
IFS=',' read -ra DEP_ARRAY <<< "$CLOUD_DEPS"

# Build a single string of dependencies
DEPS=""
for dep in "${DEP_ARRAY[@]}"; do
    dep=$(echo "$dep" | xargs)   # trim spaces
    DEPS="$DEPS $dep"
done

echo "Adding dependencies:$DEPS"
uv add $DEPS
