# Gunakan base image yang ringan dan support ML
FROM continuumio/miniconda3:latest

# Set workdir
WORKDIR /app

# Salin semua file dari project ke dalam container
COPY . .

# Install dependencies menggunakan conda.yaml
RUN conda env create -f conda.yaml

# Set environment untuk menjalankan perintah dalam conda environment
ENV PATH /opt/conda/envs/mlflow-env/bin:$PATH

# Menambahkan volume untuk menyimpan model hasil training
VOLUME /app/artifacts

# Jalankan model training saat container dijalankan
CMD ["python", "modelling.py"]
