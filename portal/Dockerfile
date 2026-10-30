# The portal (portal/app.py) behind Caddy, and its worker (portal/worker.py), which compiles
# the submissions with anta_database: see docker-compose.yml and README.md
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt requirements-worker.txt portal/
RUN pip install --no-cache-dir -r portal/requirements.txt -r portal/requirements-worker.txt
# The package, copied next to the portal by deploy.sh
COPY package/anta_database anta_database/
COPY *.py var_attrs.json portal/
RUN useradd --system --uid 1000 portal && mkdir -p /data && chown portal /data
USER portal
ENV PORTAL_DATA=/data MPLCONFIGDIR=/tmp/matplotlib
# Only Caddy can reach port 8000 (not published): trust its X-Forwarded-For for the client IPs
CMD ["uvicorn", "portal.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
