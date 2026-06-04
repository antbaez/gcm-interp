#!/bin/bash

SRC="/root/gcm-interp/results"
DEST="/root/gcm-interp/results_copy"

echo "Copying txt, yml, and json files from $SRC to $DEST..."

find "$SRC" -type f \( -name "*.txt" -o -name "*.yml" -o -name "*.json" \) | while read -r file; do
    rel="${file#$SRC/}"
    dest_file="$DEST/$rel"
    mkdir -p "$(dirname "$dest_file")"
    cp "$file" "$dest_file"
done

echo "Done. Directory size:"
du -sh "$DEST"
