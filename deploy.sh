#!/bin/bash

# Face Recognition App - Quick Deploy Script

echo "🚀 Face Recognition App - Deployment Helper"
echo "==========================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git initialized!"
fi

# Add all files
echo "📝 Adding files to Git..."
git add .

# Commit
echo "💾 Committing changes..."
git commit -m "Deploy face recognition app - $(date)"

echo ""
echo "🎯 Deployment Options:"
echo "======================"
echo ""
echo "1️⃣  HEROKU DEPLOYMENT"
echo "   Run these commands:"
echo "   heroku create your-face-recognition-app"
echo "   git push heroku main"
echo "   heroku open"
echo ""
echo "2️⃣  RAILWAY DEPLOYMENT"
echo "   1. Go to railway.app"
echo "   2. Connect your GitHub repo"
echo "   3. Deploy automatically"
echo ""
echo "3️⃣  RENDER DEPLOYMENT"
echo "   1. Go to render.com"
echo "   2. Create new Web Service"
echo "   3. Connect GitHub repo"
echo "   4. Use: pip install -r requirements_deploy.txt"
echo "   5. Start command: python app.py"
echo ""
echo "4️⃣  LOCAL NETWORK ACCESS"
echo "   python app.py"
echo "   Access from other devices: http://YOUR_IP:8090"
echo ""
echo "📚 For detailed instructions, see DEPLOYMENT_GUIDE.md"
echo ""
echo "🎉 Ready to deploy! Good luck!"
