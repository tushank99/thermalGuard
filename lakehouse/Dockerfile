# Local Spark + Delta Lake runtime (free; avoids Hadoop winutils on Windows).
# Build:  docker build -t thermalguard-spark -f lakehouse/Dockerfile .
# Run:    docker run --rm -v "${PWD}:/app" thermalguard-spark python -m thermalguard.lakehouse.pipeline
FROM python:3.12-slim-bookworm

# Use HTTPS mirrors (some networks block plain HTTP on port 80).
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

COPY lakehouse/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-fetch the Delta jars into the image so runs work offline and start fast.
RUN python -c "from delta import configure_spark_with_delta_pip as c; from pyspark.sql import SparkSession as S; \
s=c(S.builder.master('local[1]').config('spark.sql.extensions','io.delta.sql.DeltaSparkSessionExtension')).getOrCreate(); s.stop()"

WORKDIR /app
