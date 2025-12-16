set -e

read -p "Enter google drive link:" LINK

if [-z "$LINK"]; then   
    echo "ERROR: No link provided!"
    exit 1
fi

echo "Downloading from: $LINK"
gdown "$LINK"