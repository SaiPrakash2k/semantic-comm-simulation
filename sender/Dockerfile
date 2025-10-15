FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
# Copy the images into the container
COPY ./images ./images
CMD ["python", "-u", "app.py"]