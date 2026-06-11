FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy>=2,<3" "scipy>=1.14,<2" "matplotlib>=3.9,<4"
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl
COPY runner.py /runner.py
