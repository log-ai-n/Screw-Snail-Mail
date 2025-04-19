#!/bin/bash
# Run script for Desk-Mail-Bot
# This script activates the virtual environment and runs the application

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found."
    echo "Please run './setup.sh' first to set up the environment."
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Function to check if the Gemini API key is set
check_api_key() {
    if [ ! -f ".env" ]; then
        echo "❌ .env file not found."
        echo "Please run './setup.sh' to create it, then add your Gemini API key."
        return 1
    fi
    
    # Check if the API key is still the placeholder
    if grep -q "paste-your-key-here" .env; then
        echo "⚠️ It appears you haven't set your Gemini API key yet."
        echo "Please edit the .env file and replace 'paste-your-key-here' with your actual key."
        echo "You can get a key from: https://makersuite.google.com/app/apikey"
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    return 0
}

# Main menu function
show_menu() {
    clear
    echo "┌───────────────────────────────────┐"
    echo "│       Desk-Mail-Bot Runner        │"
    echo "└───────────────────────────────────┘"
    echo
    echo " 1) Test Camera"
    echo " 2) Run Basic Version"
    echo " 3) Run Enhanced Version"
    echo " 4) Run iPhone Continuity Camera Version"
    echo " 5) Run Gemini Vision Version (Recommended)"
    echo " 6) Exit"
    echo
    read -p "Select an option [1-6]: " choice
    
    case $choice in
        1)
            echo "📷 Running camera test..."
            python camera_test.py
            read -p "Press Enter to continue..."
            show_menu
            ;;
        2)
            if check_api_key; then
                echo "📬 Running basic Desk-Mail-Bot..."
                python desk_mail.py
            fi
            read -p "Press Enter to continue..."
            show_menu
            ;;
        3)
            if check_api_key; then
                echo "📬 Running enhanced Desk-Mail-Bot..."
                python desk_mail_enhanced.py
            fi
            read -p "Press Enter to continue..."
            show_menu
            ;;
        4)
            if check_api_key; then
                echo "📱 Running iPhone Continuity Camera version..."
                python desk_mail_iphone.py
            fi
            read -p "Press Enter to continue..."
            show_menu
            ;;
        5)
            if check_api_key; then
                echo "🔍 Running Gemini Vision version (best accuracy)..."
                python desk_mail_vision.py
            fi
            read -p "Press Enter to continue..."
            show_menu
            ;;
        6)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "⚠️ Invalid option. Please try again."
            read -p "Press Enter to continue..."
            show_menu
            ;;
    esac
}

# Start the menu
show_menu

# Deactivate virtual environment on exit (this part will never execute due to the exit in the menu)
deactivate
