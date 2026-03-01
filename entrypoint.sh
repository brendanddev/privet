#!/bin/bash

# Wait for Ollama to be ready before starting the app
echo "Waiting for Ollama to be ready..."
until curl -s http://ollama:11434/api/tags > /dev/null; do
    sleep 2
done
echo "Ollama is ready — starting app..."
streamlit run app.py --server.address=0.0.0.0 --server.port=8501