#!/bin/bash
# This script robustly downloads and extracts the RecSys 2025 challenge data.

set -e # Exit immediately if a command fails.

# --- Configuration ---
DEST_DIR="ubc_data"
URL="https://data.recsys.synerise.com/dataset/ubc_data/ubc_data.tar.gz"
TEMP_ARCHIVE="/tmp/recsys_data.tar.gz"
TEMP_EXTRACT_DIR="/tmp/recsys_extract_$$" # $$ creates a unique temp directory

echo "Destination directory: ${DEST_DIR}"

# --- Step 1: Check if data already exists ---
# We check for a key file to be sure.
if [ -f "${DEST_DIR}/product_buy.parquet" ]; then
    echo "✅ Data already exists in '${DEST_DIR}'. Nothing to do."
    exit 0
fi

# --- Step 2: Download the data ---
echo "Downloading data from ${URL}..."
# Use -q (quiet) and --show-progress for a cleaner output
wget -q --show-progress -O "${TEMP_ARCHIVE}" "${URL}"

# --- Step 3: Extract and move ---
# This approach handles archives with or without a root 'ubc_data' folder.
echo "Extracting archive to a temporary location..."
mkdir -p "${TEMP_EXTRACT_DIR}"
tar -xvzf "${TEMP_ARCHIVE}" -C "${TEMP_EXTRACT_DIR}"

echo "Organizing extracted files into '${DEST_DIR}'..."
mkdir -p "${DEST_DIR}"

# Check if a 'ubc_data' folder was created inside our temp directory
if [ -d "${TEMP_EXTRACT_DIR}/ubc_data" ]; then
    # If yes, move the *contents* of that folder
    # The 'shopt' commands ensure that hidden files (like .DS_Store) are also moved if they exist.
    shopt -s dotglob
    mv ${TEMP_EXTRACT_DIR}/ubc_data/* "${DEST_DIR}/"
    shopt -u dotglob
else
    # If no, it means files were extracted directly. Move them.
    shopt -s dotglob
    mv ${TEMP_EXTRACT_DIR}/* "${DEST_DIR}/"
    shopt -u dotglob
fi

# --- Step 4: Cleanup ---
echo "Cleaning up temporary files..."
rm "${TEMP_ARCHIVE}"
rm -rf "${TEMP_EXTRACT_DIR}"

echo "✅ Data download and extraction complete. Files are in '${DEST_DIR}'."