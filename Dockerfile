FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app
COPY . /app

# Install any additional Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
