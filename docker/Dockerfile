# Use an official slim Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install OS-level dependencies if needed (e.g., build tools)
# If you have libraries that need compilation (e.g., torch-geometric), you may need apt-get installs here.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code into container
COPY . .

# (Optional) Expose port for JupyterLab
EXPOSE 8888

# Default command: launch JupyterLab on container start
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
