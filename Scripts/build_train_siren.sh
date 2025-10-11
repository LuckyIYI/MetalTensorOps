#!/bin/bash
set -e

# Build the train_siren CLI executable
# This script compiles the training script along with all necessary dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building train_siren executable..."
echo "Project root: $PROJECT_ROOT"

# Output executable
OUTPUT_DIR="$PROJECT_ROOT/build"
mkdir -p "$OUTPUT_DIR"
OUTPUT_BIN="$OUTPUT_DIR/train_siren"

# Source files
SOURCES=(
    "$SCRIPT_DIR/train_siren.swift"
    "$PROJECT_ROOT/MetalTensorOp/Model/MLP.swift"
    "$PROJECT_ROOT/MetalTensorOp/Model/Model.swift"
    "$PROJECT_ROOT/MetalTensorOp/Encoders/SirenEncoder.swift"
    "$PROJECT_ROOT/MetalTensorOp/Training/SirenTrainer.swift"
    "$PROJECT_ROOT/MetalTensorOp/Render/RenderSupport.swift"
    "$PROJECT_ROOT/MetalTensorOp/RenderMode.swift"
)

# Compile with swiftc
swiftc \
    -D TRAINING_CLI \
    -O \
    -target arm64-apple-macosx26.0 \
    -sdk "$(xcrun --show-sdk-path --sdk macosx)" \
    -framework Metal \
    -framework MetalKit \
    -framework MetalPerformanceShaders \
    -framework QuartzCore \
    -framework CoreGraphics \
    -framework ImageIO \
    -framework UniformTypeIdentifiers \
    "${SOURCES[@]}" \
    -o "$OUTPUT_BIN"

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "Executable: $OUTPUT_BIN"
    echo ""
    echo "Run with: $OUTPUT_BIN --input <image.png>"
else
    echo "❌ Build failed"
    exit 1
fi
