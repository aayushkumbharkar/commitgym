FROM node:20-alpine AS frontend
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend COMMITGYM_DB=/app/data/commitgym.sqlite3
COPY backend/requirements.lock.txt ./requirements.lock.txt
RUN pip install --no-cache-dir torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.lock.txt
COPY backend/ ./backend/
COPY models/ppo_commitgym.zip models/training.json models/evaluation.json ./models/
COPY --from=frontend /web/dist ./frontend/dist/
RUN useradd --uid 10001 --create-home commitgym && mkdir -p data && chown -R commitgym:commitgym /app
USER commitgym
EXPOSE 8000
# Hosts such as Render inject PORT; local Docker defaults to 8000.
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/api/health',timeout=3)"
CMD ["sh","-c","exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
