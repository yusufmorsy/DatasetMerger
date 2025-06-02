# Use the official slim Python 3.11 image as a lightweight base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker’s layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies without caching wheels
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code into the working directory
COPY . .

# Expose port 8000 for the FastAPI/Uvicorn server
EXPOSE 8000

# Command to run the FastAPI app via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]