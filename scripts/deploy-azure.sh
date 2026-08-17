#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

readonly SUBSCRIPTION_ID="25d0cf2e-a75c-46f5-b26c-f57a48f96967"
readonly OWNER="waleed@vulsight.com"
readonly LOCATION="eastus2"
readonly RESOURCE_GROUP="morgott-preview-rg"
readonly REGISTRY="morgottvulsightacr"
readonly VAULT="morgott-vulsight-kv"
readonly APP="morgott-api"
readonly JOB="morgott-daily-canary"
readonly MODEL_KEY="mmbert-lora-full-ctx1024-u17000-s42"

log() { printf '%s\n' "$*"; }

deploy_temp=$(mktemp -d)
published=false
app_update_started=false
app_preexisting=false
job_preexisting=false
previous_external=false
previous_mode="single"
previous_revision=""
previous_job_image=""
previous_job_command_id=""
new_revision=""

cleanup() {
	if $app_update_started && ! $published; then
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
		if $job_preexisting; then
			az containerapp job update \
				--name "$JOB" \
				--resource-group "$RESOURCE_GROUP" \
				--image "$previous_job_image" \
				--set-env-vars "MORGOTT_CANARY_COMMAND_ID=$previous_job_command_id" \
				--output none 2>/dev/null || true
		else
			az containerapp job delete \
				--name "$JOB" \
				--resource-group "$RESOURCE_GROUP" \
				--yes \
				--output none 2>/dev/null || true
		fi
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

for command in az awk curl git jq openssl rg script sha256sum uv; do
	require "$command"
done

account_user=$(az account show --query user.name -o tsv)
account_subscription=$(az account show --query id -o tsv)
if [[ ${account_user,,} != "$OWNER" || $account_subscription != "$SUBSCRIPTION_ID" ]]; then
	log "Azure must be signed in as $OWNER on $SUBSCRIPTION_ID" >&2
	exit 1
fi

log "Verifying the registered 1,024-token serving artifact"
git lfs pull --include="artifacts/models/$MODEL_KEY/serving/**" --exclude=""
policy_identity=$(
	uv run --locked python - <<'PY'
import json
from pathlib import Path

from morgott.models.cascade import _verify_registered_policy
from morgott.models.downstream import PIPELINE_PROFILE, THRESHOLD_SHA256
from morgott.models.mmbert.core import file_sha256

key = "mmbert-lora-full-ctx1024-u17000-s42"
manifest = Path("model-artifacts.json")
entry = json.loads(manifest.read_text(encoding="utf-8"))["models"][key]
serving = entry.get("serving")
if (
    not isinstance(serving, dict)
    or serving.get("max_tokens") != 1024
    or serving.get("window_overlap") != 128
    or serving.get("inference_precision") != "bf16"
):
    raise SystemExit("the verified 1,024 serving artifact is not registered")
for name in ("onnx", "tokenizer", "export", "verification"):
    spec = serving[name]
    path = manifest.parent / spec["path"]
    if not path.is_file() or file_sha256(path) != spec["sha256"]:
        raise SystemExit(f"registered serving artifact failed: {name}")
policy_sha256 = _verify_registered_policy(manifest)
print(json.dumps({
    "policy_sha256": policy_sha256,
    "profile": PIPELINE_PROFILE,
    "threshold_sha256": THRESHOLD_SHA256,
}, sort_keys=True))
PY
)
PIPELINE_PROFILE=$(jq -er '.profile' <<<"$policy_identity")
POLICY_SHA256=$(jq -er '.policy_sha256' <<<"$policy_identity")
THRESHOLD_SHA256=$(jq -er '.threshold_sha256' <<<"$policy_identity")
readonly PIPELINE_PROFILE POLICY_SHA256 THRESHOLD_SHA256

log "Registering Azure providers and validating infrastructure"
for namespace in \
	Microsoft.App \
	Microsoft.ContainerRegistry \
	Microsoft.ServiceBus \
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
manifest_sha256=$(sha256sum data/manifest.json | cut -d' ' -f1)
budget_start=$(date -u +%Y-%m-01T00:00:00Z)
canary_command_id=$(openssl rand -hex 16)
parameters=(
	"deployerPrincipalId=$deployer_principal_id"
	"dataManifestSha256=$manifest_sha256"
	"budgetStartDate=$budget_start"
	"canaryCommandId=$canary_command_id"
)

az deployment group create \
	--name morgott-preview-foundation \
	--resource-group "$RESOURCE_GROUP" \
	--template-file infra/main.bicep \
	--parameters "${parameters[@]}" deployApplication=false \
	--output none

log "Copying only the approved .env keys into Key Vault"
openrouter_key=$(dotenv_value OPENROUTER_API_KEY)
hf_token=$(dotenv_value HF_TOKEN)
hugging_face_token=$(dotenv_value HUGGING_FACE_HUB_TOKEN)
morgott_sas_url=$(dotenv_value MORGOTT_SAS_URL)
openai_key=$(dotenv_value OPENAI_API_KEY)
if [[ -z $openrouter_key ]]; then
	log "OPENROUTER_API_KEY is required in .env" >&2
	exit 1
fi
if [[ -n $hf_token && -n $hugging_face_token && $hf_token != "$hugging_face_token" ]]; then
	log "HF_TOKEN and HUGGING_FACE_HUB_TOKEN differ; refusing to choose" >&2
	exit 1
fi
[[ -n $hf_token ]] || hf_token=$hugging_face_token

set_secret openrouter-api-key "$openrouter_key"
set_secret hf-token "$hf_token"
set_secret morgott-sas-url "$morgott_sas_url"
set_secret openai-api-key "$openai_key"

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
	{
		find src -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
		sha256sum Dockerfile pyproject.toml uv.lock model-artifacts.json
		sha256sum "artifacts/models/$MODEL_KEY/serving/model.onnx"
	} | sha256sum | cut -d' ' -f1
)
image_tag="ctx1024-${image_fingerprint:0:16}"
tag_lookup_error="$deploy_temp/acr-tag-lookup.error"
if tag_count=$(az acr repository show-tags \
	--name "$REGISTRY" \
	--repository morgott-api \
	--query "[?@ == '$image_tag'] | length(@)" \
	--output tsv 2>"$tag_lookup_error"); then
	:
elif rg -q 'RepositoryNotFound' "$tag_lookup_error"; then
	tag_count=0
else
	log "Could not inspect the ACR image repository" >&2
	sed 's/^/  /' "$tag_lookup_error" >&2
	exit 1
fi
if [[ $tag_count == 0 ]]; then
	az acr build \
		--registry "$REGISTRY" \
		--image "morgott-api:$image_tag" \
		--file Dockerfile \
		.
elif [[ $tag_count != 1 ]]; then
	log "Unexpected ACR tag count: $tag_count" >&2
	exit 1
fi
image_digest=$(az acr repository show \
	--name "$REGISTRY" \
	--image "morgott-api:$image_tag" \
	--query digest \
	--output tsv)
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

job_state_error="$deploy_temp/container-job.error"
if previous_job_state=$(az containerapp job show \
	--name "$JOB" \
	--resource-group "$RESOURCE_GROUP" \
	--output json 2>"$job_state_error"); then
	job_preexisting=true
	previous_job_image=$(jq -r '.properties.template.containers[0].image // empty' <<<"$previous_job_state")
	previous_job_command_id=$(jq -r '
    [.properties.template.containers[0].env[]? | select(.name == "MORGOTT_CANARY_COMMAND_ID") | .value][0] // empty
  ' <<<"$previous_job_state")
	if [[ -z $previous_job_image || -z $previous_job_command_id ]]; then
		log "Could not capture the existing Container App Job rollback state" >&2
		exit 1
	fi
elif ! rg -q 'ResourceNotFound' "$job_state_error"; then
	log "Could not inspect the existing Container App Job" >&2
	sed 's/^/  /' "$job_state_error" >&2
	exit 1
fi
deployment_to_running_started=$(date +%s)
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
	--output none

for _ in {1..36}; do
	revision_name=$(az containerapp show \
		--name "$APP" \
		--resource-group "$RESOURCE_GROUP" \
		--query properties.latestRevisionName \
		--output tsv)
	new_revision=$revision_name
	revision_state=$(az containerapp revision show \
		--name "$APP" \
		--revision "$revision_name" \
		--resource-group "$RESOURCE_GROUP" \
		--query properties.runningState \
		--output tsv)
	[[ $revision_state == Running || $revision_state == RunningAtMaxScale ]] && break
	sleep 10
done
if [[ $revision_state != Running && $revision_state != RunningAtMaxScale ]]; then
	log "Container App revision did not reach Running" >&2
	exit 1
fi
deployment_to_running_seconds=$(($(date +%s) - deployment_to_running_started))

log "Running precision, auth, bounds, advisory, latency, and memory smoke measurements"
script --quiet --return --command \
	"az containerapp exec --name $APP --resource-group $RESOURCE_GROUP --revision $revision_name --command 'python -m morgott.azure_app smoke-local'" \
	/dev/null

log "Publishing 100 percent of external traffic to the verified revision"
az containerapp ingress enable \
	--name "$APP" \
	--resource-group "$RESOURCE_GROUP" \
	--type external \
	--allow-insecure false \
	--target-port 8000 \
	--transport auto \
	--output none
az containerapp ingress traffic set \
	--name "$APP" \
	--resource-group "$RESOURCE_GROUP" \
	--revision-weight "$revision_name=100" \
	--output none

fqdn=$(az containerapp show \
	--name "$APP" \
	--resource-group "$RESOURCE_GROUP" \
	--query properties.configuration.ingress.fqdn \
	--output tsv)
curl --fail --silent --show-error "https://$fqdn/healthz" | jq -e '.status == "ready"' >/dev/null
curl_config="$deploy_temp/curl.conf"
printf 'header = "Authorization: Bearer %s"\n' "$api_key" >"$curl_config"
curl --fail --silent --show-error \
	--config "$curl_config" \
	"https://$fqdn/v1/status" |
	jq -e \
		--arg key "$MODEL_KEY" \
		--arg profile "$PIPELINE_PROFILE" \
		--arg policy "$POLICY_SHA256" \
		--arg threshold "$THRESHOLD_SHA256" \
		'.ready == true and .model_key == $key and .pipeline_profile == $profile and .policy_sha256 == $policy and .threshold_sha256 == $threshold and .context_length == 1024 and .requested_precision == "auto" and (.precision == "bf16" or .precision == "fp32")' \
		>/dev/null
rm -f "$curl_config"

if [[ -n $previous_revision && $previous_revision != "$revision_name" ]]; then
	az containerapp revision deactivate \
		--name "$APP" \
		--resource-group "$RESOURCE_GROUP" \
		--revision "$previous_revision" \
		--output none || log "Warning: previous revision remains active" >&2
fi
mode_set=false
for _ in {1..3}; do
	if az containerapp revision set-mode \
		--name "$APP" \
		--resource-group "$RESOURCE_GROUP" \
		--mode single \
		--output none; then
		mode_set=true
		break
	fi
	sleep 5
done
if ! $mode_set; then
	log "Could not return the Container App to single-revision mode" >&2
	exit 1
fi

log "Running the managed-identity Service Bus and Blob canary round trip"
az containerapp job start \
	--name "$JOB" \
	--resource-group "$RESOURCE_GROUP" \
	--output none
canary_passed=false
for _ in {1..24}; do
	logs=$(az containerapp logs show \
		--name "$APP" \
		--resource-group "$RESOURCE_GROUP" \
		--type console \
		--tail 200 2>/dev/null || true)
	canary_log=$(printf '%s' "$logs" |
		rg 'daily_canary_complete' |
		rg -F "$canary_command_id" || true)
	if printf '%s' "$canary_log" | rg -q 'provider_calls[^0-9]*[1-9]'; then
		canary_passed=true
		break
	fi
	sleep 10
done
if ! $canary_passed; then
	log "Service Bus, Blob, or OpenRouter canary did not pass" >&2
	exit 1
fi

published=true
log "Azure preview deployed: https://$fqdn"
log "Image: $image"
log "Policy: $PIPELINE_PROFILE ($POLICY_SHA256)"
log "Observed deployment-to-Running: ${deployment_to_running_seconds}s"
