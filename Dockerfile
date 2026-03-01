
# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working dir inside container
WORKDIR /app

# Copy requirements first (Docker caches this layer so deps only reinstall when it changes)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create required directories if they dont exist
RUN mkdir -p docs logs

# Expose Streamlits default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]