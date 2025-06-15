#!/bin/bash
# Post-render script to move build files to _build directory
# Following Adam's coding standards: automated, DRY, clear error handling

set -euo pipefail  # Exit on error, undefined vars, pipe failures

BUILD_DIR="_build"
TEX_DIR="$BUILD_DIR/tex"
PDF_DIR="$BUILD_DIR"

# Create build directories
mkdir -p "$TEX_DIR"
mkdir -p "$PDF_DIR"

echo "Moving build artifacts to $BUILD_DIR..."

# Move LaTeX source and auxiliary files
for ext in tex aux log bbl blg fls fdb_latexmk; do
    if [ -f "paper.$ext" ]; then
        mv "paper.$ext" "$TEX_DIR/" && echo "  Moved paper.$ext → $TEX_DIR/"
    fi
done

# Move copied BST files (format-resources artifacts)
for bst_file in *.bst; do
    if [ -f "$bst_file" ] && [ "$bst_file" != "template/bst/*" ]; then
        mv "$bst_file" "$TEX_DIR/" && echo "  Moved $bst_file → $TEX_DIR/"
    fi
done

# Move PDF from ./ to _build/ (if it exists there)
if [ -f "paper.pdf" ]; then
    mv "paper.pdf" "$PDF_DIR/" && echo "  Moved paper.pdf → $PDF_DIR/"
fi


echo "Build files organized in $BUILD_DIR/"
echo "  PDF: $PDF_DIR/paper.pdf"
echo "  LaTeX sources: $TEX_DIR/"
