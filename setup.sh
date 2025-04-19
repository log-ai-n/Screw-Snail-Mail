#!/bin/bash
# Setup script for Desk-Mail-Bot
# This script creates a virtual environment and installs all required dependencies

echo "📬 Setting up Desk-Mail-Bot environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if Homebrew is installed (for macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo "⚠️ Homebrew is not installed. Some dependencies may need manual installation."
        read -p "Do you want to continue without Homebrew? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Visit https://brew.sh to install Homebrew, then run this script again."
            exit 1
        fi
    else
        # Install Tesseract OCR with Homebrew
        echo "📥 Installing Tesseract OCR engine..."
        brew install tesseract
    fi
else
    echo "⚠️ This is not a macOS system. Please ensure Tesseract OCR is installed manually."
    echo "   On Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "   On Fedora: sudo dnf install tesseract"
    read -p "Press Enter to continue once Tesseract is installed, or Ctrl+C to abort..."
fi

# Create and activate virtual environment
echo "🔧 Creating Python virtual environment..."
python3 -m venv .venv

# Determine the activation script based on shell
if [[ "$SHELL" == *"zsh"* ]]; then
    source .venv/bin/activate
    echo "source \$(pwd)/.venv/bin/activate" > activate.sh
elif [[ "$SHELL" == *"bash"* ]]; then
    source .venv/bin/activate
    echo "source \$(pwd)/.venv/bin/activate" > activate.sh
else
    echo "source \$(pwd)/.venv/bin/activate" > activate.sh
    source .venv/bin/activate
fi

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create a sample .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating sample .env file..."
    echo 'GEMINI_API_KEY="paste-your-key-here"' > .env
    echo "⚠️ Please edit .env file and add your actual Gemini API key."
fi

echo "✅ Setup complete!"
echo
echo "To use the application:"
echo "1. Edit .env file and add your Gemini API key (if you haven't already)"
echo "2. Run 'source activate.sh' to activate the environment"
echo "3. Run 'python camera_test.py' to test your camera"
echo "4. Run 'python desk_mail.py' or 'python desk_mail_enhanced.py' to start the application"
echo
echo "You can also use the run.sh script to activate and run the application in one step:"
echo "  ./run.sh"
echo

# Make run.sh executable
chmod +x run.sh

# Exit with success
exit 0
