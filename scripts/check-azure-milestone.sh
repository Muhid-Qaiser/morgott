#!/usr/bin/env bash
set -euo pipefail

readonly SUBSCRIPTION_ID="25d0cf2e-a75c-46f5-b26c-f57a48f96967"
readonly OWNER="waleed@vulsight.com"
readonly RESOURCE_GROUP="morgott-preview-rg"
readonly DAY_ZERO_TAG="mfs25kDayZero"
readonly SERVICES_JSON='["Virtual Machines","Storage","Container Registry","Azure Container Apps","Service Bus"]'

user=$(az account show --query user.name -o tsv)
subscription=$(az account show --query id -o tsv)
if [[ ${user,,} != "$OWNER" || $subscription != "$SUBSCRIPTION_ID" ]]; then
	printf 'Azure must be signed in as %s on %s\n' "$OWNER" "$SUBSCRIPTION_ID" >&2
	exit 1
fi

from=$(date -u -d '61 days ago' +%Y-%m-%d)
today=$(date -u +%Y-%m-%d)
to=$(date -u -d 'tomorrow' +%Y-%m-%d)
query=$(jq -cn \
	--arg from "$from" \
	--arg to "$to" \
	--argjson services "$SERVICES_JSON" \
	'{
    type: "ActualCost",
    timeframe: "Custom",
    timePeriod: {from: ($from + "T00:00:00Z"), to: ($to + "T00:00:00Z")},
    dataset: {
      granularity: "Daily",
      aggregation: {cost: {name: "PreTaxCost", function: "Sum"}},
      grouping: [{type: "Dimension", name: "ServiceName"}],
      filter: {dimensions: {name: "ServiceName", operator: "In", values: $services}}
    }
  }')

cost_error=$(mktemp)
trap 'rm -f "$cost_error"' EXIT
cost_response=""
for attempt in {1..6}; do
	if cost_response=$(az rest \
		--method post \
		--url "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/providers/Microsoft.CostManagement/query?api-version=2025-03-01" \
		--body "$query" 2>"$cost_error"); then
		break
	fi
	if ! rg -q '(^|[^0-9])(429|503)([^0-9]|$)|Too Many Requests|ServiceUnavailable' "$cost_error"; then
		printf '%s\n' 'Azure Cost Management query failed:' >&2
		sed 's/^/  /' "$cost_error" >&2
		exit 1
	fi
	if [[ $attempt == 6 ]]; then
		printf '%s\n' 'Azure Cost Management remained throttled after six attempts.' >&2
		exit 1
	fi
	sleep 10
done

summary=$(jq -c \
	--argjson services "$SERVICES_JSON" \
	'
  .properties as $properties
  | ($properties.columns | map(.name)) as $columns
  | [
      $properties.rows[]?
      | [range(0; $columns | length) as $index | {key: $columns[$index], value: .[$index]}]
      | from_entries
    ] as $rows
  | $services
  | map(
      . as $service
      | ([
          $rows[]
          | select(.ServiceName == $service)
          | {date: (.UsageDate | tostring), cost: (.PreTaxCost | tonumber), currency: .Currency}
        ] | sort_by(.date)) as $daily
	      | (reduce $daily[] as $row (
	          {cost: 0, crossing: null, currency: "USD"};
	          .cost += $row.cost
	          | .currency = $row.currency
	          | if .cost < 1 then .crossing = null
	            elif .crossing == null then .crossing = $row.date
	            else .
	            end
        )) as $total
      | {
          service: $service,
          cost: $total.cost,
          currency: $total.currency,
          crossing: (
            if $total.crossing == null then null
            else ($total.crossing[0:4] + "-" + $total.crossing[4:6] + "-" + $total.crossing[6:8])
            end
          )
        }
    )
  ' <<<"$cost_response")

printf '%-24s %12s %8s %12s\n' 'Workload' '61-day cost' 'Currency' 'Crossed USD 1'
jq -r '.[] | [.service, (.cost | tostring), .currency, (.crossing // "pending")] | @tsv' <<<"$summary" |
	awk -F '\t' '{printf "%-24s %12.4f %8s %12s\n", $1, $2, $3, $4}'

qualified_count=$(jq '[.[] | select(.cost >= 1)] | length' <<<"$summary")
day_zero=$(az group show \
	--name "$RESOURCE_GROUP" \
	--query "tags.$DAY_ZERO_TAG" \
	--output tsv)

if [[ $qualified_count != 5 ]]; then
	printf '\nQualified intended workloads: %s/5. Day zero has not started.\n' "$qualified_count"
	if [[ -n $day_zero ]]; then
		az group update \
			--name "$RESOURCE_GROUP" \
			--remove "tags.$DAY_ZERO_TAG" \
			--output none
		printf 'Continuity broke, so the saved day zero (%s) was cleared.\n' "$day_zero"
	fi
	exit 0
fi

if [[ -z $day_zero ]]; then
	candidate_day_zero=$(jq -r '[.[].crossing] | max' <<<"$summary")
	if [[ ${1:-} != --portal-confirmed ]]; then
		printf '\nCost data suggests day zero %s. Confirm five workloads in the Startup portal, then rerun with --portal-confirmed.\n' "$candidate_day_zero"
		exit 0
	fi
	day_zero=$candidate_day_zero
	az group update \
		--name "$RESOURCE_GROUP" \
		--set "tags.$DAY_ZERO_TAG=$day_zero" \
		--output none
	printf '\nAll five intended workloads crossed USD 1. Saved day zero: %s.\n' "$day_zero"
else
	printf '\nAll five intended workloads remain above USD 1. Saved day zero: %s.\n' "$day_zero"
fi

elapsed_days=$(((\
	$(date -u -d "$today" +%s) - $(date -u -d "$day_zero" +%s)) / 86400))
if ((elapsed_days >= 60)); then
	printf 'Local continuity tracker: %s days. Confirm eligibility in the Startup portal.\n' "$elapsed_days"
else
	printf 'Local continuity tracker: %s/60 days.\n' "$elapsed_days"
fi
