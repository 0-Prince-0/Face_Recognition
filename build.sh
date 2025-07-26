#!/bin/bash

# Render Build Script for Face Recognition App

echo "🚀 Starting Render build process..."

# Update pip
pip install --upgrade pip

# Install system dependencies (if needed)
echo "📦 Installing Python dependencies..."
pip install -r requirements_deploy.txt

# Create necessary directories
mkdir -p training_data
mkdir -p static
mkdir -p templates

echo "✅ Build completed successfully!"
