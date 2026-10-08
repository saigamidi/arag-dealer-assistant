#!/usr/bin/env bash
# Starts the mock API in the background, then the chat app. Ctrl+C stops both.
set -e
uvicorn mock_api.main:app --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT INT TERM
sleep 2
kill -0 $API_PID 2>/dev/null || { echo "Mock API failed to start (port 8000 in use?)"; exit 1; }
streamlit run app.py
