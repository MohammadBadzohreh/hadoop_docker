FROM python:3.8-slim

# Install Java for Spark
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install PySpark
RUN pip install --no-cache-dir pyspark

# Hadoop config
RUN mkdir -p /etc/hadoop/conf
COPY core-site.xml /etc/hadoop/conf/core-site.xml
ENV HADOOP_CONF_DIR=/etc/hadoop/conf

# Java env
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:${PATH}"

# Copy script
WORKDIR /app
COPY people_you_might_know.py /app/people_you_might_know.py
RUN chmod +x /app/people_you_might_know.py

# No ENTRYPOINT: allow direct invocation of python
