set -e

ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    CLOUD_DEPS=$(grep '^CLOUD_DEPENDENCIES' "$ENV_FILE$" | cut -d= -f2 | tr -d ' ')

else
    echo ".env file not found!"
    exit 1
fi 

IFS=',' read -ra DEP_ARRAY <<< "$CLOUD_DEPS"

for dep in "${DEP_ARRAY[@]}"; do 
    echo "Adding dependency: $dep"
    uv add "$dep"
done 