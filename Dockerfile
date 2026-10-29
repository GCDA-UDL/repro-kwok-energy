# Analysis environment of the reproduction package: regenerates the paper's numbers
# and tables from the dataset (no RAPL, GPU or Kubernetes needed).
#   docker build -t repro-kwok-energy .
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repro" repro-kwok-energy
# Run the tests instead:
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repro" --entrypoint python repro-kwok-energy \
#       -m pytest -q -p no:cacheprovider tests/unit tests/integration
# Capturing new traces needs the bench hosts themselves: see REPRODUCE.md.
FROM python:3.11.10-slim-bookworm@sha256:840e180ebcc6e5c8efab209c43f5e40fd2af98cb49db5c7103c90539c56bb30e

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/repro/src:/repro/tools/e2e PYTHONDONTWRITEBYTECODE=1
WORKDIR /repro
ENTRYPOINT ["bash", "reproduce.sh"]
