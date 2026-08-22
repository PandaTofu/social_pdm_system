# Spark on Kubernetes migration

Only use this path after the Compose prototype has passed the development experiment. A one-node k3s cluster validates packaging and scheduling, not horizontal scalability.

1. Build and push an image containing the Spark job and its Python dependencies.
2. Create a `pdm` namespace and Secrets for Kafka, MongoDB and object storage credentials.
3. Install the Spark Operator or submit directly with `spark-submit --master k8s://...`.
4. Apply `streaming-sparkapplication.yaml`, replacing image, service account, S3 endpoint and secret names.
5. Confirm shuffle tracking or an external shuffle service before enabling dynamic allocation.

For published multi-node claims, use at least two worker nodes and preserve the Spark event logs plus Kubernetes pod metrics for each run.
