FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .

# Expose Streamlit default port
EXPOSE 7860

# Health-check so HF Space knows the app is live
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Launch the Streamlit app on 0.0.0.0 so HF routing works
ENTRYPOINT ["streamlit", "run", "app.py",
            "--server.port=7860",
            "--server.address=0.0.0.0"]
