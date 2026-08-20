#!/usr/bin/env bash
set -euo pipefail
umask 077

cd "$(dirname "$0")/.."

candidate_size="2cpu-4gi"
while (($#)); do
	case "$1" in
	--promote)
		printf '%s\n' "Azure promotion is blocked pending the replacement paired latency gate" >&2
		exit 2
		;;
	--candidate-size)
		shift
		[[ $# -gt 0 ]] || {
			printf '%s\n' "--candidate-size requires 2cpu-4gi or 4cpu-8gi" >&2
			exit 2
		}
		candidate_size="$1"
		;;
	*)
		printf 'usage: %s [--candidate-size 2cpu-4gi|4cpu-8gi]\n' "$0" >&2
		exit 2
		;;
	esac
	shift
done
if [[ $candidate_size != "2cpu-4gi" && $candidate_size != "4cpu-8gi" ]]; then
	printf '%s\n' "--candidate-size requires 2cpu-4gi or 4cpu-8gi" >&2
	exit 2
fi
readonly candidate_size

readonly SUBSCRIPTION_ID="25d0cf2e-a75c-46f5-b26c-f57a48f96967"
readonly OWNER="waleed@vulsight.com"
readonly LOCATION="eastus2"
readonly RESOURCE_GROUP="morgott-preview-rg"
readonly REGISTRY="morgottvulsightacr"
readonly VAULT="morgott-vulsight-kv"
readonly APP="morgott-api"
readonly MODEL_KEY="mmbert-lora-full-ctx1024-u17000-s42"
readonly STORAGE_ACCOUNT="vulsightdata"
readonly STORAGE_CONTAINER="morgott"

log() { printf '%s\n' "$*"; }

deploy_temp=$(mktemp -d)
candidate_retained=false
app_update_started=false
app_preexisting=false
previous_external=false
previous_mode="single"
previous_revision=""
new_revision=""

cleanup() {
	if $app_update_started && ! $candidate_retained; then
		if [[ -z $new_revision ]]; then
			new_revision=$(az containerapp show \
				--name "$APP" \
				--resource-group "$RESOURCE_GROUP" \
				--query properties.latestRevisionName \
				--output tsv 2>/dev/null || true)
		fi
		if $app_preexisting; then
			log "Deployment failed; keeping traffic on the previous revision" >&2
			az containerapp revision set-mode \
				--name "$APP" \
				--resource-group "$RESOURCE_GROUP" \
				--mode multiple \
				--output none 2>/dev/null || true
			az containerapp revision activate \
				--name "$APP" \
				--resource-group "$RESOURCE_GROUP" \
				--revision "$previous_revision" \
				--output none 2>/dev/null || true
			az containerapp ingress traffic set \
				--name "$APP" \
				--resource-group "$RESOURCE_GROUP" \
				--revision-weight "$previous_revision=100" \
				--output none 2>/dev/null || true
			if $previous_external; then
				az containerapp ingress enable \
					--name "$APP" \
					--resource-group "$RESOURCE_GROUP" \
					--type external \
					--allow-insecure false \
					--target-port 8000 \
					--transport auto \
					--output none 2>/dev/null || true
			else
				az containerapp ingress disable \
					--name "$APP" \
					--resource-group "$RESOURCE_GROUP" \
					--output none 2>/dev/null || true
			fi
		else
			az containerapp ingress disable \
				--name "$APP" \
				--resource-group "$RESOURCE_GROUP" \
				--output none 2>/dev/null || true
		fi
		if [[ -n $new_revision && $new_revision != "$previous_revision" ]]; then
			az containerapp revision deactivate \
				--name "$APP" \
				--resource-group "$RESOURCE_GROUP" \
				--revision "$new_revision" \
				--output none 2>/dev/null || true
		fi
		az containerapp revision set-mode \
			--name "$APP" \
			--resource-group "$RESOURCE_GROUP" \
			--mode "$previous_mode" \
			--output none 2>/dev/null || true
	fi
	rm -rf -- "$deploy_temp"
}
trap cleanup EXIT

require() {
	command -v "$1" >/dev/null || {
		log "missing required command: $1" >&2
		exit 1
	}
}

dotenv_value() {
	local name="$1" value
	[[ -f .env ]] || return 0
	value=$(awk -v key="$name" '
    substr($0, 1, length(key) + 1) == key "=" {
      value = substr($0, length(key) + 2)
    }
    END { printf "%s", value }
  ' .env)
	value=${value%$'\r'}
	if [[ ${#value} -ge 2 ]]; then
		if [[ ${value:0:1} == '"' && ${value: -1} == '"' ]] ||
			[[ ${value:0:1} == "'" && ${value: -1} == "'" ]]; then
			value=${value:1:${#value}-2}
		fi
	fi
	printf '%s' "$value"
}

set_secret() {
	local name="$1" value="$2" secret_file
	[[ -n "$value" ]] || return 0
	secret_file="$deploy_temp/secret-$name"
	printf '%s' "$value" >"$secret_file"
	for _ in {1..18}; do
		if az keyvault secret set \
			--vault-name "$VAULT" \
			--name "$name" \
			--file "$secret_file" \
			--encoding utf-8 \
			--output none 2>/dev/null; then
			rm -f "$secret_file"
			return 0
		fi
		sleep 10
	done
	rm -f "$secret_file"
	log "could not write Key Vault secret: $name" >&2
	return 1
}

ensure_secret_matches() {
	local name="$1" value="$2" existing error_file
	[[ -n $value ]] || return 0
	error_file="$deploy_temp/secret-$name.error"
	if existing=$(az keyvault secret show \
		--vault-name "$VAULT" \
		--name "$name" \
		--query value \
		--output tsv 2>"$error_file"); then
		if [[ $existing != "$value" ]]; then
			log "Key Vault secret $name differs from .env; rotate it separately" >&2
			return 1
		fi
		return 0
	fi
	if rg -q 'SecretNotFound' "$error_file"; then
		set_secret "$name" "$value"
		return
	fi
	log "Could not verify Key Vault secret: $name" >&2
	sed 's/^/  /' "$error_file" >&2
	return 1
}

revision_smoke() {
	local revision="$1" raw="$deploy_temp/smoke-$1.raw"
	script --quiet --return --command \
		"az containerapp exec --name $APP --resource-group $RESOURCE_GROUP --revision $revision --command 'python -m morgott.azure_app'" \
		"$raw" >/dev/null
	uv run --locked python - "$raw" <<'PY'
import json
import re
import sys

raw = open(sys.argv[1], encoding="utf-8", errors="ignore").read().replace("\r", "")
for match in re.finditer(r"\{", raw):
    try:
        value = json.JSONDecoder().raw_decode(raw[match.start() :])[0]
    except json.JSONDecodeError:
        continue
    if (
        isinstance(value, dict)
        and isinstance(value.get("status"), dict)
        and isinstance(value.get("routed_probe"), dict)
    ):
        print(json.dumps(value, sort_keys=True))
        break
else:
    raise SystemExit("revision smoke result was not found")
PY
}

for command in az awk git jq openssl rg script sha256sum uv; do
	require "$command"
done

if [[ -n $(git status --porcelain=v1 --untracked-files=all) ]]; then
	log "A clean Git worktree is required for an attributable deployment" >&2
	exit 1
fi
account_user=$(az account show --query user.name -o tsv)
account_subscription=$(az account show --query id -o tsv)
if [[ ${account_user,,} != "$OWNER" || $account_subscription != "$SUBSCRIPTION_ID" ]]; then
	log "Azure must be signed in as $OWNER on $SUBSCRIPTION_ID" >&2
	exit 1
fi

log "Verifying the registered 1,024-token serving artifact"
git lfs pull --include="artifacts/models/$MODEL_KEY/serving/**" --exclude=""
CASCADE_POLICY_PATH="artifacts/models/$MODEL_KEY/serving/promotion-retrieval.json"
EVIDENCE_PATH="reports/retrieval-lineage-hybrid-parity-relaxed-20260820.json"
RETRIEVAL_MANIFEST_PATH="artifacts/models/$MODEL_KEY/serving/retrieval/lineage-hybrid-v3/manifest.json"
if ! RETRIEVAL_MANIFEST_SHA256=$(jq -er \
	--arg key "$MODEL_KEY" \
	--arg path "$RETRIEVAL_MANIFEST_PATH" '
    .models[$key].serving.retrieval
    | select(.format == "morgott-lineage-hybrid-v1" and .manifest.path == $path)
    | .manifest.sha256
    | select(type == "string" and test("^[0-9a-f]{64}$"))
  ' model-artifacts.json); then
	log "The registered retrieval manifest identity is invalid" >&2
	exit 1
fi
readonly CASCADE_POLICY_PATH EVIDENCE_PATH
readonly RETRIEVAL_MANIFEST_PATH RETRIEVAL_MANIFEST_SHA256

log "Verifying the frozen routed-canary probe"
probe_identity=$(
	OMP_NUM_THREADS=1 \
		OPENBLAS_NUM_THREADS=1 \
		MKL_NUM_THREADS=1 \
		NUMEXPR_NUM_THREADS=1 \
		uv run --locked --extra cascade python - <<'PY'
import hashlib
import json
from pathlib import Path

from morgott.azure_app import (
    ROUTED_PROBE_PACKET_SHA256,
    ROUTED_PROBE_SCORE_RANGE,
    ROUTED_PROBE_SHA256,
    ROUTED_PROBE_TEXT,
)
from morgott.models.downstream import route
from morgott.models.mmbert.serving import MmbertRuntime

if hashlib.sha256(ROUTED_PROBE_TEXT.encode()).hexdigest() != ROUTED_PROBE_SHA256:
    raise SystemExit("routed-canary probe text identity changed")
runtime = MmbertRuntime.from_artifacts(
    Path("model-artifacts.json"), inference_precision="auto"
)
scores = runtime.score(runtime.prepare(ROUTED_PROBE_TEXT).windows)
if len(scores) != 1:
    raise SystemExit("routed-canary probe no longer fits one model window")
score = scores[0]
if not ROUTED_PROBE_SCORE_RANGE[0] <= score <= ROUTED_PROBE_SCORE_RANGE[1]:
    raise SystemExit("routed-canary probe score left its frozen range")
result = route(score, input_channel="untrusted_content")
if (result.route, result.reason) != ("review", "deepseek_required"):
    raise SystemExit("routed-canary probe no longer requires review")
print(
    json.dumps(
        {
            "expected_packet_sha256": ROUTED_PROBE_PACKET_SHA256,
            "score_max": ROUTED_PROBE_SCORE_RANGE[1],
            "score_min": ROUTED_PROBE_SCORE_RANGE[0],
            "sha256": ROUTED_PROBE_SHA256,
        },
        sort_keys=True,
    )
)
PY
)
ROUTED_PROBE_PACKET_SHA256=$(jq -er '.expected_packet_sha256' <<<"$probe_identity")
ROUTED_PROBE_SCORE_MAX=$(jq -er '.score_max' <<<"$probe_identity")
ROUTED_PROBE_SCORE_MIN=$(jq -er '.score_min' <<<"$probe_identity")
ROUTED_PROBE_SHA256=$(jq -er '.sha256' <<<"$probe_identity")
readonly ROUTED_PROBE_PACKET_SHA256 ROUTED_PROBE_SCORE_MAX
readonly ROUTED_PROBE_SCORE_MIN ROUTED_PROBE_SHA256

log "Staging the registered private retrieval bundle"
build_context="$deploy_temp/build-context"
mkdir -p "$build_context"
cp Dockerfile .dockerignore pyproject.toml uv.lock model-artifacts.json "$build_context/"
cp -a src "$build_context/"
serving_source="artifacts/models/$MODEL_KEY/serving"
serving_target="$build_context/$serving_source"
mkdir -p "$serving_target"
for name in model.onnx tokenizer.json export.json verification.json benchmark.json; do
	cp "$serving_source/$name" "$serving_target/$name"
done
policy_target="$build_context/$CASCADE_POLICY_PATH"
mkdir -p "$(dirname "$policy_target")"
cp "$CASCADE_POLICY_PATH" "$policy_target"
evidence_target="$build_context/$EVIDENCE_PATH"
mkdir -p "$(dirname "$evidence_target")"
cp "$EVIDENCE_PATH" "$evidence_target"

retrieval_manifest_target="$build_context/$RETRIEVAL_MANIFEST_PATH"
mkdir -p "$(dirname "$retrieval_manifest_target")"
az storage blob download \
	--account-name "$STORAGE_ACCOUNT" \
	--container-name "$STORAGE_CONTAINER" \
	--name "$RETRIEVAL_MANIFEST_PATH" \
	--file "$retrieval_manifest_target" \
	--auth-mode login \
	--overwrite true \
	--output none
if [[ $(sha256sum "$retrieval_manifest_target" | cut -d' ' -f1) != "$RETRIEVAL_MANIFEST_SHA256" ]]; then
	log "Downloaded retrieval manifest failed SHA-256 verification" >&2
	exit 1
fi
manifest_sha256=$(sha256sum data/manifest.json | cut -d' ' -f1)
retrieval_data_manifest_sha256=$(jq -er '.source.data_manifest_sha256' "$retrieval_manifest_target")
if [[ $retrieval_data_manifest_sha256 != "$manifest_sha256" ]]; then
	log "retrieval bank was built from a different data manifest" >&2
	exit 1
fi
readonly manifest_sha256 retrieval_data_manifest_sha256

retrieval_files="$deploy_temp/retrieval-files.tsv"
uv run --locked python - "$retrieval_manifest_target" "$MODEL_KEY" >"$retrieval_files" <<'PY'
import json
import sys
from collections import Counter
from pathlib import PurePosixPath

manifest_path, model_key = sys.argv[1:]
manifest = json.loads(open(manifest_path, encoding="utf-8").read())
files = manifest.get("files")
if (
    manifest.get("schema_version") != 1
    or manifest.get("variant") != "lineage_hybrid_v1"
    or not isinstance(files, list)
    or len(files) != 10
):
    raise SystemExit("retrieval bundle manifest contract failed")
bundle_root = PurePosixPath(
    f"artifacts/models/{model_key}/serving/retrieval/lineage-hybrid-v3"
)
seen = set()
roles = Counter()
for spec in files:
    if not isinstance(spec, dict) or set(spec) != {"role", "path", "sha256", "bytes"}:
        raise SystemExit("retrieval file spec contract failed")
    role = spec["role"]
    path = spec["path"]
    digest = spec["sha256"]
    size = spec["bytes"]
    pure_path = PurePosixPath(path) if isinstance(path, str) else None
    if (
        role not in {"bank", "sparse", "index", "row_map"}
        or pure_path is None
        or pure_path.is_absolute()
        or len(pure_path.parts) != 1
        or ".." in pure_path.parts
        or path in seen
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise SystemExit("retrieval file identity is invalid")
    seen.add(path)
    roles[role] += 1
    print(bundle_root / pure_path, digest, size, sep="\t")
if roles != {"bank": 1, "sparse": 1, "index": 4, "row_map": 4}:
    raise SystemExit("retrieval bundle roles are incomplete")
PY

download_retrieval_file() {
	local blob_path="$1" expected_sha256="$2" expected_bytes="$3" target
	target="$build_context/$blob_path"
	mkdir -p "$(dirname "$target")"
	if [[ -f $blob_path ]] &&
		[[ $(stat -c '%s' "$blob_path") == "$expected_bytes" ]] &&
		[[ $(sha256sum "$blob_path" | cut -d' ' -f1) == "$expected_sha256" ]]; then
		cp --reflink=auto "$blob_path" "$target"
	else
		az storage blob download \
			--account-name "$STORAGE_ACCOUNT" \
			--container-name "$STORAGE_CONTAINER" \
			--name "$blob_path" \
			--file "$target" \
			--auth-mode login \
			--overwrite true \
			--output none
	fi
	if [[ $(stat -c '%s' "$target") != "$expected_bytes" ]] ||
		[[ $(sha256sum "$target" | cut -d' ' -f1) != "$expected_sha256" ]]; then
		log "Downloaded retrieval artifact failed verification: $blob_path" >&2
		return 1
	fi
}

while IFS=$'\t' read -r blob_path expected_sha256 expected_bytes; do
	if ! download_retrieval_file "$blob_path" "$expected_sha256" "$expected_bytes"; then
		log "Private retrieval bundle download failed" >&2
		exit 1
	fi
done <"$retrieval_files"

policy_identity=$(
	uv run --locked --extra cascade python - \
		"$build_context/model-artifacts.json" \
		"$retrieval_manifest_target" <<'PY'
import json
import sys
from pathlib import Path

from morgott.models.cascade import _verify_registered_policy
from morgott.models.downstream import PIPELINE_PROFILE, THRESHOLD_SHA256

manifest = Path(sys.argv[1])
expected_retrieval_manifest = Path(sys.argv[2]).resolve()
policy_sha256, retrieval_manifest, retrieval_sha256 = _verify_registered_policy(
    manifest
)
if retrieval_manifest != expected_retrieval_manifest:
    raise SystemExit("registered retrieval manifest path changed during staging")
registry = json.loads(manifest.read_text(encoding="utf-8"))
serving = registry["models"]["mmbert-lora-full-ctx1024-u17000-s42"]["serving"]
promotion_path = manifest.parent / serving["cascade_policy"]["path"]
runtime_contract = json.loads(promotion_path.read_text(encoding="utf-8"))[
    "runtime_contract"
]
reviewer = runtime_contract["reviewer"]
retrieval = runtime_contract["retrieval"]
print(
    json.dumps(
        {
            "onnx_sha256": serving["onnx"]["sha256"],
            "policy_sha256": policy_sha256,
            "profile": PIPELINE_PROFILE,
            "embedding_request_sha256": retrieval["embedding_request_sha256"],
            "retrieval_manifest_sha256": retrieval_sha256,
            "reviewer_prompt_sha256": reviewer["prompt_sha256"],
            "reviewer_provider": reviewer["requested_provider"],
            "reviewer_request_sha256": reviewer["request_sha256"],
            "threshold_sha256": THRESHOLD_SHA256,
        },
        sort_keys=True,
    )
)
PY
)
ONNX_SHA256=$(jq -er '.onnx_sha256 | select(test("^[0-9a-f]{64}$"))' <<<"$policy_identity")
PIPELINE_PROFILE=$(jq -er '.profile' <<<"$policy_identity")
POLICY_SHA256=$(jq -er '.policy_sha256' <<<"$policy_identity")
EMBEDDING_REQUEST_SHA256=$(jq -er '.embedding_request_sha256' <<<"$policy_identity")
verified_retrieval_sha256=$(jq -er '.retrieval_manifest_sha256' <<<"$policy_identity")
REVIEWER_PROMPT_SHA256=$(jq -er '.reviewer_prompt_sha256' <<<"$policy_identity")
REVIEWER_PROVIDER=$(jq -er '.reviewer_provider' <<<"$policy_identity")
REVIEWER_REQUEST_SHA256=$(jq -er '.reviewer_request_sha256' <<<"$policy_identity")
THRESHOLD_SHA256=$(jq -er '.threshold_sha256' <<<"$policy_identity")
if [[ $verified_retrieval_sha256 != "$RETRIEVAL_MANIFEST_SHA256" ]]; then
	log "Registered retrieval manifest identity changed during staging" >&2
	exit 1
fi
readonly EMBEDDING_REQUEST_SHA256 ONNX_SHA256 PIPELINE_PROFILE POLICY_SHA256
readonly REVIEWER_PROMPT_SHA256 REVIEWER_PROVIDER REVIEWER_REQUEST_SHA256
readonly THRESHOLD_SHA256

log "Registering Azure providers and validating infrastructure"
for namespace in \
	Microsoft.App \
	Microsoft.ContainerRegistry \
	Microsoft.KeyVault \
	Microsoft.OperationalInsights \
	Microsoft.ManagedIdentity \
	Microsoft.Consumption \
	Microsoft.Insights; do
	az provider register --namespace "$namespace" --wait --output none
done
if ! az extension show --name containerapp --output none 2>/dev/null; then
	az extension add --name containerapp --yes --output none
fi
az bicep build --file infra/main.bicep --stdout >/dev/null

az group create \
	--name "$RESOURCE_GROUP" \
	--location "$LOCATION" \
	--output none
az group update \
	--name "$RESOURCE_GROUP" \
	--set tags.project=morgott tags.environment=preview tags.owner=waleed \
	--output none

deployer_principal_id=$(az ad signed-in-user show --query id -o tsv)
budget_start=$(date -u +%Y-%m-01T00:00:00Z)
parameters=(
	"deployerPrincipalId=$deployer_principal_id"
	"budgetStartDate=$budget_start"
)

az deployment group create \
	--name morgott-preview-foundation \
	--resource-group "$RESOURCE_GROUP" \
	--template-file infra/main.bicep \
	--parameters "${parameters[@]}" deployApplication=false \
	--output none

log "Verifying the OpenRouter key against Key Vault"
openrouter_key=$(dotenv_value OPENROUTER_API_KEY)
if [[ -z $openrouter_key ]]; then
	log "OPENROUTER_API_KEY is required in .env" >&2
	exit 1
fi
ensure_secret_matches openrouter-api-key "$openrouter_key"

api_key_error="$deploy_temp/morgott-api-key.error"
if api_key=$(az keyvault secret show \
	--vault-name "$VAULT" \
	--name morgott-api-key \
	--query value \
	--output tsv 2>"$api_key_error"); then
	if [[ ${#api_key} -lt 32 ]]; then
		log "Existing morgott-api-key is shorter than 32 characters; refusing to rotate it" >&2
		exit 1
	fi
elif rg -q 'SecretNotFound' "$api_key_error"; then
	api_key=$(openssl rand -hex 32)
	set_secret morgott-api-key "$api_key"
else
	log "Could not read the existing morgott-api-key; refusing to rotate it" >&2
	sed 's/^/  /' "$api_key_error" >&2
	exit 1
fi

log "Building an immutable ACR image"
image_fingerprint=$(
	(
		cd "$build_context"
		{
			find src -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
			sha256sum Dockerfile .dockerignore pyproject.toml uv.lock model-artifacts.json
			sha256sum "artifacts/models/$MODEL_KEY/serving/model.onnx"
			printf '%s  retrieval-manifest\n' "$RETRIEVAL_MANIFEST_SHA256"
		} | sha256sum | cut -d' ' -f1
	)
)
image_tag="lineage-hybrid-${image_fingerprint:0:16}"
az acr build \
	--registry "$REGISTRY" \
	--image "morgott-api:$image_tag" \
	--file Dockerfile \
	"$build_context"
image_digest=""
for _ in {1..6}; do
	if image_digest=$(az acr repository show \
		--name "$REGISTRY" \
		--image "morgott-api:$image_tag" \
		--query digest \
		--output tsv 2>/dev/null) &&
		[[ $image_digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
		break
	fi
	image_digest=""
	sleep 5
done
if [[ ! $image_digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
	log "ACR returned an invalid image digest" >&2
	exit 1
fi
image="$REGISTRY.azurecr.io/morgott-api@$image_digest"

log "Deploying a zero-traffic revision for validation"
app_state_error="$deploy_temp/container-app.error"
if previous_state=$(az containerapp show \
	--name "$APP" \
	--resource-group "$RESOURCE_GROUP" \
	--query '{external: properties.configuration.ingress.external, latestReadyRevisionName: properties.latestReadyRevisionName, mode: properties.configuration.activeRevisionsMode, traffic: properties.configuration.ingress.traffic}' \
	--output json 2>"$app_state_error"); then
	app_preexisting=true
	previous_external=$(jq -r '.external // false' <<<"$previous_state")
	previous_mode=$(jq -r '(.mode // "single") | ascii_downcase' <<<"$previous_state")
	previous_revision=$(jq -r '
    .latestReadyRevisionName as $latest
    | [.traffic[]? | select(.weight == 100)] as $routes
    | if ($routes | length) != 1 then ""
      elif $routes[0].revisionName then $routes[0].revisionName
      elif $routes[0].latestRevision == true then $latest
      else ""
      end
  ' <<<"$previous_state")
	if [[ -z $previous_revision ]]; then
		log "Expected exactly one existing revision with 100 percent traffic" >&2
		exit 1
	fi
elif ! rg -q 'ResourceNotFound' "$app_state_error"; then
	log "Could not inspect the existing Container App; refusing to replace it" >&2
	sed 's/^/  /' "$app_state_error" >&2
	exit 1
fi

app_update_started=true
az deployment group create \
	--name morgott-preview-internal \
	--resource-group "$RESOURCE_GROUP" \
	--template-file infra/main.bicep \
	--parameters "${parameters[@]}" \
	deployApplication=true \
	"image=$image" \
	"stableRevisionName=$previous_revision" \
	"externalIngress=$previous_external" \
	"candidateSize=$candidate_size" \
	--output none

revision_state=""
for _ in {1..36}; do
	if ! revision_name=$(az containerapp show \
		--name "$APP" \
		--resource-group "$RESOURCE_GROUP" \
		--query properties.latestRevisionName \
		--output tsv); then
		sleep 10
		continue
	fi
	if $app_preexisting && [[ $revision_name == "$previous_revision" ]]; then
		sleep 10
		continue
	fi
	new_revision=$revision_name
	if ! revision_details=$(az containerapp revision show \
		--name "$APP" \
		--revision "$revision_name" \
		--resource-group "$RESOURCE_GROUP" \
		--query '{active:properties.active,state:properties.runningState}' \
		--output json); then
		sleep 10
		continue
	fi
	if [[ $(jq -r .active <<<"$revision_details") != true ]]; then
		if ! az containerapp revision activate \
			--name "$APP" \
			--resource-group "$RESOURCE_GROUP" \
			--revision "$revision_name" \
			--output none; then
			sleep 10
			continue
		fi
		sleep 10
		continue
	fi
	revision_state=$(jq -r .state <<<"$revision_details")
	[[ $revision_state == Running || $revision_state == RunningAtMaxScale ]] && break
	sleep 10
done
if [[ $revision_state != Running && $revision_state != RunningAtMaxScale ]]; then
	log "Container App revision did not reach Running" >&2
	exit 1
fi

if $app_preexisting; then
	post_deploy_traffic=$(az containerapp show \
		--name "$APP" \
		--resource-group "$RESOURCE_GROUP" \
		--query properties.configuration.ingress.traffic \
		--output json)
	if ! jq -e --arg stable "$previous_revision" --arg candidate "$revision_name" '
      ([.[]? | select(.revisionName == $stable) | .weight] | add // 0) == 100
      and ([.[]? | select(.revisionName == $candidate and .weight > 0)] | length) == 0
      and ([.[]? | select((.latestRevision // false) == true and .weight > 0)] | length) == 0
      and ([.[]?.weight] | add // 0) == 100
    ' <<<"$post_deploy_traffic" >/dev/null; then
		log "Candidate is not a distinct zero-traffic revision" >&2
		exit 1
	fi
fi

candidate_resources=$(az containerapp revision show \
	--name "$APP" \
	--resource-group "$RESOURCE_GROUP" \
	--revision "$revision_name" \
	--query properties.template.containers[0].resources \
	--output json)
candidate_memory_limit_bytes=$(jq -er \
	'.memory | capture("^(?<gib>[1-9][0-9]*)Gi$").gib | tonumber * 1073741824' \
	<<<"$candidate_resources")

log "Running the candidate status, routed probe, and local-pass memory smoke"
candidate_smoke=$(revision_smoke "$revision_name")
if ! jq -e \
	--arg embedding_request "$EMBEDDING_REQUEST_SHA256" \
	--arg key "$MODEL_KEY" \
	--arg onnx "$ONNX_SHA256" \
	--arg packet "$ROUTED_PROBE_PACKET_SHA256" \
	--arg policy "$POLICY_SHA256" \
	--arg probe "$ROUTED_PROBE_SHA256" \
	--arg profile "$PIPELINE_PROFILE" \
	--arg prompt "$REVIEWER_PROMPT_SHA256" \
	--arg provider "$REVIEWER_PROVIDER" \
	--arg provider_request "$REVIEWER_REQUEST_SHA256" \
	--arg retrieval "$RETRIEVAL_MANIFEST_SHA256" \
	--arg threshold "$THRESHOLD_SHA256" \
	--argjson memory_limit "$candidate_memory_limit_bytes" \
	--argjson score_max "$ROUTED_PROBE_SCORE_MAX" \
	--argjson score_min "$ROUTED_PROBE_SCORE_MIN" '
	    .status.model_key == $key
	    and .status.onnx_sha256 == $onnx
	    and .status.pipeline_profile == $profile
	    and .status.policy_sha256 == $policy
	    and .status.retrieval_enabled == true
	    and .status.retrieval_manifest_sha256 == $retrieval
	    and .status.threshold_sha256 == $threshold
	    and .status.context_length == 1024
	    and .status.window_overlap == 128
	    and .status.requested_precision == "auto"
	    and (.status.precision == "bf16" or .status.precision == "fp32")
    and .routed_probe.decision == "allow"
    and .routed_probe.advisory_route == "pass"
    and .routed_probe.reason == "deepseek_clear"
    and .routed_probe.complete == true
    and .routed_probe.artifact_sha256 == $probe
    and .routed_probe.middle_windows == 1
    and .routed_probe.low_windows == 0
    and .routed_probe.high_windows == 0
    and .routed_probe.max_mmbert_score >= $score_min
    and .routed_probe.max_mmbert_score <= $score_max
    and .routed_probe.deepseek_calls >= 1
    and .routed_probe.deepseek_failures == 0
    and .routed_probe.retrieval_status == "ok"
    and .routed_probe.selected_example_count == 4
    and .routed_probe.retrieval_packet_sha256 == $packet
    and .routed_probe.embedding_request_sha256 == $embedding_request
    and .routed_probe.prompt_sha256 == $prompt
    and .routed_probe.provider == $provider
    and .routed_probe.provider_request_sha256 == $provider_request
    and (
      (.cgroup_memory_peak_bytes // ((.peak_rss_kib // 0) * 1024)) as $peak
      | (.cgroup_memory_limit_bytes // $memory_limit) as $limit
      | $peak > 0 and $limit > 0 and ($limit - $peak) >= 536870912
    )
  ' <<<"$candidate_smoke" >/dev/null; then
	log "Candidate identity, routed probe, or 512 MiB memory headroom check failed" >&2
	log "Use --candidate-size 4cpu-8gi if the candidate exceeded the 4 GiB shape" >&2
	exit 1
fi

candidate_retained=true
log "Validated candidate retained at zero traffic: $revision_name"
