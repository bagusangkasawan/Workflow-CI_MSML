# Gunakan image base yang sudah ada
FROM continuumio/miniconda3:latest

# Set working directory di dalam container
WORKDIR /app

# Salin file environment conda dan project ke dalam container
COPY MLProject/conda.yaml /app/conda.yaml
COPY MLProject/obesity_data_preprocessing.csv /app/obesity_data_preprocessing.csv
COPY MLProject/modelling.py /app/modelling.py

# Install dependencies menggunakan conda
RUN conda env create -f /app/conda.yaml

# Set environment variable untuk mengaktifkan conda environment
ENV PATH /opt/conda/envs/mlflow-env/bin:$PATH

# Install Docker dependencies jika diperlukan
RUN apt-get update && apt-get install -y git

# Install mlflow untuk logging
RUN pip install mlflow

# Install Docker CLI untuk build dan push image jika diperlukan
RUN apt-get install -y docker.io

# Perintah untuk menjalankan model
CMD ["python", "modelling.py"]
