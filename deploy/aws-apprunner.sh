#!/usr/bin/env bash
# Deploy the contract intelligence service to AWS App Runner, from the CLI.
#
# App Runner is chosen over Lambda or ECS because this app is single-instance
# by construction (in-memory state, local SQLite, local extraction cache), and
# App Runner gives HTTPS, a managed certificate and a public URL without a VPC,
# load balancer or task definition to maintain.
#
# Prerequisites you must do yourself:
#   1. aws configure           (or `aws login`) -- credentials
#   2. Docker Desktop running
#
# Usage:
#   ./deploy/aws-apprunner.sh            # create or update
#   REGION=eu-west-1 ./deploy/aws-apprunner.sh
set -euo pipefail

REGION="${REGION:-us-east-1}"
APP="${APP:-contract-intel}"
REPO="$APP"
SERVICE="$APP"
ROLE_NAME="AppRunnerECRAccessRole"
CPU="${CPU:-1024}"        # 1 vCPU
MEMORY="${MEMORY:-2048}"  # 2 GB

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

say "Checking prerequisites"
aws sts get-caller-identity >/dev/null || {
  echo "No AWS credentials. Run: aws configure"; exit 1; }
docker info >/dev/null 2>&1 || {
  echo "Docker daemon is not running. Start Docker Desktop."; exit 1; }
[ -n "${GROQ_API_KEY:-}" ] || {
  echo "GROQ_API_KEY is not set in your shell."
  echo "The demo works without it; uploads and the agent do not."
  read -rp "Continue without it? [y/N] " ok; [ "$ok" = y ] || exit 1; }

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$ECR/$REPO:latest"
echo "  account $ACCOUNT  region $REGION"

say "Ensuring ECR repository"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null
echo "  $ECR/$REPO"

say "Building image"
# App Runner runs x86_64. On an Apple Silicon Mac the default build is arm64
# and the service would fail to start with an exec-format error, so the
# platform is pinned explicitly.
docker build --platform linux/amd64 -t "$REPO:latest" .
docker tag "$REPO:latest" "$IMAGE"

say "Pushing to ECR"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR"
docker push "$IMAGE"

say "Ensuring the ECR access role App Runner needs"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow",
        "Principal":{"Service":"build.apprunner.amazonaws.com"},
        "Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  echo "  created $ROLE_NAME (waiting for IAM to propagate)"
  sleep 12
fi
ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"

ENV_JSON='{}'
[ -n "${GROQ_API_KEY:-}" ] && ENV_JSON="{\"GROQ_API_KEY\":\"$GROQ_API_KEY\"}"

EXISTING="$(aws apprunner list-services --region "$REGION" \
  --query "ServiceSummaryList[?ServiceName=='$SERVICE'].ServiceArn" --output text)"

if [ -z "$EXISTING" ]; then
  say "Creating App Runner service"
  cat > /tmp/apprunner-src.json <<JSON
{
  "ImageRepository": {
    "ImageIdentifier": "$IMAGE",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": {
      "Port": "8080",
      "RuntimeEnvironmentVariables": $ENV_JSON
    }
  },
  "AutoDeploymentsEnabled": false,
  "AuthenticationConfiguration": { "AccessRoleArn": "$ROLE_ARN" }
}
JSON
  ARN="$(aws apprunner create-service --region "$REGION" \
    --service-name "$SERVICE" \
    --source-configuration file:///tmp/apprunner-src.json \
    --instance-configuration "Cpu=$CPU,Memory=$MEMORY" \
    --health-check-configuration 'Protocol=HTTP,Path=/portfolio/stats,Interval=20,Timeout=5,HealthyThreshold=1,UnhealthyThreshold=5' \
    --query Service.ServiceArn --output text)"
else
  say "Updating existing service"
  ARN="$EXISTING"
  aws apprunner start-deployment --region "$REGION" --service-arn "$ARN" >/dev/null
fi

say "Waiting for the service to come up (a few minutes)"
for _ in $(seq 1 60); do
  STATUS="$(aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
    --query Service.Status --output text)"
  printf '  status: %s\r' "$STATUS"
  case "$STATUS" in
    RUNNING) break ;;
    CREATE_FAILED|DELETE_FAILED) echo; echo "Service failed. Check:"; \
      echo "  aws apprunner list-operations --region $REGION --service-arn $ARN"; exit 1 ;;
  esac
  sleep 15
done

URL="https://$(aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
  --query Service.ServiceUrl --output text)"
say "Live"
echo "  $URL"
echo
echo "  smoke test:  curl -s $URL/portfolio/stats"
echo "  pause (stop compute billing):"
echo "    aws apprunner pause-service --region $REGION --service-arn $ARN"
echo "  delete:"
echo "    aws apprunner delete-service --region $REGION --service-arn $ARN"
