# Gunakan image Python resmi
FROM python:3.10-slim

# Set lingkungan kerja di dalam container
WORKDIR /app

# Salin file project ke dalam container
COPY . .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Ekspos port untuk Flask
EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
