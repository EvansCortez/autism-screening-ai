FROM python:3.14.5-slim

WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY research/evaluation/pooled_metrics.csv research/evaluation/settings.json research/evaluation/audit.json research/evaluation/
COPY research/uncertainty/leave_country_out_metrics.csv research/uncertainty/
COPY research/report/figures/ research/report/figures/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
