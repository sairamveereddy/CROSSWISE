#!/usr/bin/env bash
# Deploy CrossWise to the team namespace: ConfigMap (code + data) -> Secret (VSS creds) -> Deployment/Service/Ingress.
# Requires env: USERNAME (team-N), INGRESS_URL, PASSWORD, KUBECONFIG. Never echoes credentials.
set -euo pipefail
cd "$(dirname "$0")/.."

NS="${USERNAME:?USERNAME not set}"
APP_HOST="video-lab-team-${NS#team-}.cosmos.vastdata.com"
export KUBECONFIG="${KUBECONFIG:-/config/${NS}-k8s.yaml}"

kubectl -n "$NS" create configmap crosswise-code \
  --from-file=main.py=app/main.py --from-file=index.html=app/index.html --from-file=label.html=app/label.html --from-file=logo.jpg=app/logo.jpg \
  --from-file=agent.py=agent.py --from-file=decisions.json=data/decisions.json \
  --from-file=cosmos_captions.json=data/cosmos_captions.json --from-file=label_set.json=data/label_set.json --from-file=perception.json=data/perception.json \
  --dry-run=client -o yaml | kubectl apply --server-side --force-conflicts -f -

# The public INGRESS_URL host does not resolve inside the cluster; use the backend Service instead.
VSS_INTERNAL_URL="${VSS_INTERNAL_URL:-http://video-backend-service.${NS}.svc.cluster.local:8000}"
kubectl -n "$NS" create secret generic crosswise-vss-creds \
  --from-literal=VSS_URL="$VSS_INTERNAL_URL" --from-literal=VSS_USERNAME="$NS" \
  --from-literal=VSS_PASSWORD="${PASSWORD:?}" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null && echo "secret/crosswise-vss-creds configured"

sed "s/APP_HOST/${APP_HOST}/" deploy/k8s.yaml | kubectl -n "$NS" apply -f -
kubectl -n "$NS" rollout restart deploy/crosswise
kubectl -n "$NS" rollout status deploy/crosswise --timeout=180s
curl -sS -o /dev/null -w "GET http://${APP_HOST}/app -> %{http_code}\n" "http://${APP_HOST}/app/"
curl -sS "http://${APP_HOST}/app/health"; echo
echo "Open https://workshop.thecosmoslabs.com and click the App button."
