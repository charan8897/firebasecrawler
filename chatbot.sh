#!/bin/bash
###############################################################################
# Firebase Chain Chatbot - Bash Wrapper
# Usage: ./chatbot.sh "Your query here"
#        ./chatbot.sh (interactive mode)
###############################################################################

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Run the chatbot
if [ $# -eq 0 ]; then
    # Interactive mode
    python3 "$SCRIPT_DIR/firebase_chain_chatbot.py"
else
    # Query mode
    python3 "$SCRIPT_DIR/firebase_chain_chatbot.py" "$@"
fi
