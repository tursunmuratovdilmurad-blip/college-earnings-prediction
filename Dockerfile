# Runs the Streamlit app in a container, with exactly the package versions in uv.lock.
#
#   docker build -t college-earnings .
#   docker run -p 8501:8501 college-earnings
#
# Then open http://localhost:8501

FROM python:3.12-slim

# uv, the same version that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv

WORKDIR /app

# Install the packages first, so this slow step is reused when only the code changes
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache

# Then the app itself: code, processed data, trained models and result tables
COPY . .

# Run as a normal user, not as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501

# Lets Docker see whether the app is up
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
