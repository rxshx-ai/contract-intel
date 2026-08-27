#!/usr/bin/env bash
# Create a small RDS Postgres for the service, and print the DATABASE_URL.
#
# Run this BEFORE deploy/aws-apprunner.sh, then export the URL it prints so the
# App Runner service is created with it. Without it the service falls back to
# SQLite on an ephemeral container disk and loses uploads on every recycle.
set -euo pipefail

REGION="${REGION:-us-east-1}"
ID="${ID:-contract-intel-db}"
CLASS="${CLASS:-db.t4g.micro}"      # ~$13/month, burstable, arm
STORAGE="${STORAGE:-20}"
DBNAME=contract_intel
USER=postgres

command -v openssl >/dev/null || { echo "openssl required"; exit 1; }
aws sts get-caller-identity >/dev/null || { echo "Run: aws configure"; exit 1; }

PASSWORD="$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-24)"

echo "==> Creating $ID ($CLASS, ${STORAGE}GB) in $REGION"
aws rds create-db-instance \
  --region "$REGION" \
  --db-instance-identifier "$ID" \
  --db-instance-class "$CLASS" \
  --engine postgres \
  --allocated-storage "$STORAGE" \
  --master-username "$USER" \
  --master-user-password "$PASSWORD" \
  --db-name "$DBNAME" \
  --backup-retention-period 7 \
  --storage-encrypted \
  --no-publicly-accessible \
  --no-multi-az >/dev/null

echo "==> Waiting for it to become available (usually 5-10 minutes)"
aws rds wait db-instance-available --region "$REGION" --db-instance-identifier "$ID"

HOST="$(aws rds describe-db-instances --region "$REGION" \
  --db-instance-identifier "$ID" \
  --query 'DBInstances[0].Endpoint.Address' --output text)"

cat <<MSG

Created. Store this password now -- it is not recoverable from AWS:

  export DATABASE_URL="postgresql://$USER:$PASSWORD@$HOST:5432/$DBNAME"

The instance is NOT publicly accessible, which is correct. App Runner reaches
it through a VPC connector:

  aws apprunner create-vpc-connector --region $REGION \\
    --vpc-connector-name contract-intel-vpc \\
    --subnets <private-subnet-ids> --security-groups <sg-id>

then pass that connector when creating the service. For a quick demo you can
instead use --publicly-accessible and lock the security group to your IP, but
do not leave a database open to the internet.
MSG
