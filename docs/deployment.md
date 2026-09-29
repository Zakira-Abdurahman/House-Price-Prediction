# Deployment Guide

Three ways to run the dashboard, from quickest to most production-ready. All three serve the same
code (`app/`, `src/`) and the same committed artifacts (`data/`, `models/`) - nothing is rebuilt or
retrained at deploy time.

## Option A - Streamlit Community Cloud (fastest, no Docker needed)

Streamlit Community Cloud runs `requirements.txt` + a chosen entrypoint script directly, so the
Dockerfile is not used here - it reads the same repository the Docker image is built from.

1. Push this repository to GitHub (public, or private on a plan that supports it).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at:
   - **Repository**: your fork/copy of this repo
   - **Branch**: `main`
   - **Main file path**: `app/Home.py`
3. Deploy. Build logs will show `pip install -r requirements.txt`, then the app starts.
4. Any push to `main` redeploys automatically.

Resource note: the platform's free tier has limited memory; the feature table
(`data/eth_food_prices_features_cereals.csv`, ~9 MB) and the model (~0.2 MB) are both small enough to
load comfortably within it.

## Option B - Docker on any host (VM, on-prem server, etc.)

This is what `Dockerfile` and `docker-compose.yml` are for. Any host with Docker installed works
identically, whether that's a laptop, a cloud VM, or a container platform (ECS, Cloud Run, etc. all
accept the same image).

```bash
# Build and run in the foreground (Ctrl+C to stop)
docker compose up --build

# Or run detached, and check health
docker compose up --build -d
docker compose ps
curl http://localhost:8501/_stcore/health
```

Open `http://localhost:8501` (or `http://<server-ip>:8501` on a remote host - open port 8501 in the
firewall/security group first).

### Running the built image directly (no compose)

```bash
docker build -t ethiopia-food-prices .
docker run -d --name ethiopia-food-prices -p 8501:8501 --restart unless-stopped ethiopia-food-prices
```

### Persistent deployment on a Linux VM with systemd

For a host where you want the container to survive reboots without a compose file running as a
service, create `/etc/systemd/system/ethiopia-food-prices.service`:

```ini
[Unit]
Description=Ethiopia Food Prices dashboard
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker rm -f ethiopia-food-prices
ExecStart=/usr/bin/docker run --name ethiopia-food-prices -p 8501:8501 ethiopia-food-prices:latest
ExecStop=/usr/bin/docker stop ethiopia-food-prices

[Install]
WantedBy=multi-user.target
```

Then: `systemctl daemon-reload && systemctl enable --now ethiopia-food-prices`.

Put a reverse proxy (nginx, Caddy, or a cloud load balancer) in front for HTTPS - Streamlit itself
serves plain HTTP on port 8501.

## Option C - Any container platform (Cloud Run, ECS, Azure Container Apps, Fly.io, ...)

The image is a standard container with no platform-specific assumptions: it listens on `$PORT`-agnostic
port 8501, has no required environment variables, and needs no external database or network access at
runtime (everything it reads is baked into the image). Point the platform at the `Dockerfile` (or a
pre-built image pushed to a registry) and set:

- **Container port**: 8501
- **Health check path**: `/_stcore/health`
- **Memory**: 512 MB is comfortable headroom; the app itself uses well under that
- **Concurrency**: Streamlit serves multiple browser sessions from one process; 1 instance handles a
  small team comfortably. Scale to more instances/replicas for larger audiences.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR: installs dependencies, lints, runs the full test
suite with coverage, then builds the Docker image and smoke-tests the running container (waits for the
health check, then requests the home page). A green run on `main` is a reasonable signal that the image
is safe to deploy; wiring the docker job's final step to `docker push` to a registry is the natural next
step for a fully automated pipeline, left out here since it needs registry credentials specific to
wherever this is actually hosted.

## Updating the model or data

The running app never retrains anything - it only reads files under `data/` and `models/`. To ship a
new model version:

1. Re-run the relevant notebooks (`08` through `13`) to produce new `data/eth_food_prices_features_cereals.csv`,
   a new `models/13_final_spike_classifier.joblib`, and a new `models/13_model_card.json`.
2. Re-run `python scripts/build_artifacts.py` to regenerate `data/results/model_summary.json` (the file
   the *Models & Method* page reads).
3. Run `pytest -q` locally - several tests (`tests/test_build_artifacts.py`) assert the project's
   headline findings (persistence beats ML, Hist Gradient Boosting wins classification); if the new
   model changes those conclusions, the dashboard's explanatory text needs updating too, not just the
   files.
4. Commit, push, and either let Streamlit Community Cloud redeploy automatically, or rebuild and
   redeploy the Docker image (`docker compose up --build -d`).

## Rollback

Because the image is self-contained (no external database), rolling back is just running the previous
image tag: `docker run ... ethiopia-food-prices:<previous-tag>`. Tag images with a version or commit SHA
when pushing to a registry to make this straightforward.
