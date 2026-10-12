FROM python:3.11-slim
WORKDIR /srv
COPY pyproject.toml requirements.txt ./
COPY app ./app
COPY scripts ./scripts
COPY dashboard ./dashboard
COPY data ./data
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
RUN pip install --no-cache-dir .[test]
EXPOSE 8000
CMD ["uvicorn","app.api.main:app","--host","0.0.0.0","--port","8000"]
