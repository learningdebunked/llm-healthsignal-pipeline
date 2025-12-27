#!/bin/bash

echo "🔧 Healthcare AI - GPT-4 Setup"
echo "================================"
echo ""
echo "This application now uses GPT-4 instead of GPT-2 for better medical interpretations."
echo ""
echo "To use GPT-4, you need an OpenAI API key:"
echo "1. Go to: https://platform.openai.com/api-keys"
echo "2. Sign in or create an account"
echo "3. Create a new API key"
echo "4. Copy the key"
echo ""
read -p "Enter your OpenAI API key: " api_key
echo ""

if [ -z "$api_key" ]; then
    echo "❌ No API key provided. Exiting."
    exit 1
fi

echo "✓ API key received"
echo ""
echo "Setting up environment..."
export OPENAI_API_KEY="$api_key"

# Add to shell profile for persistence
if [ -f ~/.zshrc ]; then
    if ! grep -q "OPENAI_API_KEY" ~/.zshrc; then
        echo "" >> ~/.zshrc
        echo "# OpenAI API Key for Healthcare AI" >> ~/.zshrc
        echo "export OPENAI_API_KEY='$api_key'" >> ~/.zshrc
        echo "✓ Added to ~/.zshrc for future sessions"
    fi
elif [ -f ~/.bash_profile ]; then
    if ! grep -q "OPENAI_API_KEY" ~/.bash_profile; then
        echo "" >> ~/.bash_profile
        echo "# OpenAI API Key for Healthcare AI" >> ~/.bash_profile
        echo "export OPENAI_API_KEY='$api_key'" >> ~/.bash_profile
        echo "✓ Added to ~/.bash_profile for future sessions"
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application, run:"
echo "  python3 api.py"
echo ""
echo "Or run it now? (y/n)"
read -p "> " run_now

if [ "$run_now" = "y" ] || [ "$run_now" = "Y" ]; then
    echo ""
    echo "🚀 Starting Healthcare AI API Server..."
    python3 api.py
fi
