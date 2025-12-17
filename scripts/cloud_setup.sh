set -e

pip install -q -r requirements.txt

# reading file
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
echo "Reading ENV_FILE=$ENV_FILE"

if [ -f "$ENV_FILE" ]; then
    CLOUD_DEPS=$(grep '^CLOUD_DEPENDENCIES' "$ENV_FILE" | cut -d= -f2- |xargs)
else
    echo ".env file not found!"
    exit 1
fi

# install
IFS=',' read -ra DEP_ARRAY <<< "$CLOUD_DEPS"

DEPS=""
for dep in "${DEP_ARRAY[@]}"; do
    dep = $(echo "$dep"| xargs)

    echo "Running: pip install -q $dep"
    pip install $dep

done
