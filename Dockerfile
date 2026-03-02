
# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working dir inside container
WORKDIR /app

# Install curl for the health check in entrypoint.sh
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer so deps only reinstall when it changes)
# and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create required directories if they dont exist
RUN mkdir -p docs logs

# Expose Streamlits default port
EXPOSE 8501

# Run the app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]