set -e

pip install --upgrade uv
uv sync

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
    DEPS="$DEPS $dep"
done

CMD="uv add $DEPS"
echo "Running: $CMD"
$CMD
