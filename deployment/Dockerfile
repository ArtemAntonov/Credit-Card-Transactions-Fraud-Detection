FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi==0.116.1 uvicorn==0.35.0 numpy==1.26.4 dill==0.4.0 pandas==1.4.1 scikit-learn==1.2.2 prince==0.13.1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]