# Gunakan base image yang ringan dan support ML
FROM continuumio/miniconda3:latest

# Set workdir
WORKDIR /app

# Salin semua file dari project ke dalam container
COPY . .

# Install dependencies menggunakan conda.yaml
RUN conda env create -f conda.yaml

# Pastikan conda environment tersedia
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Menambahkan volume untuk menyimpan model hasil training
VOLUME /app/artifacts

# Jalankan model training saat container dijalankan
CMD ["conda", "run", "-n", "mlflow-env", "python", "modelling.py"]
