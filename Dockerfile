# Use a Python image that works well with ROCm/AMD if needed, 
# but for the API, standard Python is usually fine.
FROM python:3.10-slim

# Install system dependencies for networking and pinging
RUN apt-get update && apt-get install -y curl iputils-ping && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# Copy the rest of your code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]