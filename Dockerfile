# Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and model code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variable for production
# the below ensures python outputs are sent straight to the terminal (stdout) without buffering
# that is, all print() and logging is real-time 
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]