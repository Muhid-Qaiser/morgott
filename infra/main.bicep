targetScope = 'resourceGroup'

@description('Deploy the Container App and scheduled job after the image exists.')
param deployApplication bool = false
param image string = ''
param deployerPrincipalId string
param dataManifestSha256 string
param budgetStartDate string
param canaryCommandId string
@description('Existing revision that keeps receiving traffic while the new revision is validated.')
param stableRevisionName string = ''
@description('Preserve external ingress for an existing working deployment during validation.')
param externalIngress bool = false

var location = resourceGroup().location
var appLocation = 'eastus'
var alertEmail = 'waleed@vulsight.com'
var storageResourceGroup = 'Data'
var storageAccountName = 'vulsightdata'
var acrPullRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var serviceBusReceiverRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4f6d3b9b-027b-4f4c-9142-0e5a2a2247e0')
var serviceBusSenderRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '69a216fc-b8fb-44d8-bc22-1f3c2cd27a39')
var keyVaultSecretsOfficerRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7')
var keyVaultSecretsUserRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')

resource storage 'Microsoft.Storage/storageAccounts@2025-06-01' existing = {
  name: storageAccountName
  scope: resourceGroup(subscription().subscriptionId, storageResourceGroup)
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' = {
  name: 'morgottvulsightacr'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    anonymousPullEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2024-11-30' = {
  name: 'morgott-preview-mi'
  location: location
}

resource vault 'Microsoft.KeyVault/vaults@2026-02-01' = {
  name: 'morgott-vulsight-kv'
  location: location
  properties: {
    tenantId: tenant().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 30
    enablePurgeProtection: true
    publicNetworkAccess: 'Enabled'
    sku: {
      family: 'A'
      name: 'standard'
    }
  }
}

resource serviceBus 'Microsoft.ServiceBus/namespaces@2026-01-01' = {
  name: 'morgott-vulsight-bus'
  location: location
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
  properties: {
    disableLocalAuth: true
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    zoneRedundant: false
  }
}

resource queue 'Microsoft.ServiceBus/namespaces/queues@2026-01-01' = {
  parent: serviceBus
  name: 'daily-canary'
  properties: {
    deadLetteringOnMessageExpiration: true
    defaultMessageTimeToLive: 'P14D'
    enableBatchedOperations: true
    lockDuration: 'PT5M'
    maxDeliveryCount: 5
  }
}

resource workspace 'Microsoft.OperationalInsights/workspaces@2025-07-01' = {
  name: 'morgott-preview-logs'
  location: location
  properties: {
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    retentionInDays: 30
    sku: {
      name: 'PerGB2018'
    }
  }
}

resource managedEnvironment 'Microsoft.App/managedEnvironments@2026-01-01' = {
  name: 'morgott-preview-env-eastus'
  location: appLocation
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspace.properties.customerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
    publicNetworkAccess: 'Enabled'
    zoneRedundant: false
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

resource registryPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, acrPullRole)
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: acrPullRole
  }
}

module blobRole 'blob-role.bicep' = {
  name: 'morgott-preview-blob-role'
  scope: resourceGroup(subscription().subscriptionId, storageResourceGroup)
  params: {
    principalId: identity.properties.principalId
    storageAccountName: storageAccountName
  }
}

resource serviceBusSend 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(serviceBus.id, identity.id, serviceBusSenderRole)
  scope: serviceBus
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: serviceBusSenderRole
  }
}

resource serviceBusReceive 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(serviceBus.id, identity.id, serviceBusReceiverRole)
  scope: serviceBus
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: serviceBusReceiverRole
  }
}

resource vaultRead 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, identity.id, keyVaultSecretsUserRole)
  scope: vault
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsUserRole
  }
}

resource vaultWrite 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, deployerPrincipalId, keyVaultSecretsOfficerRole)
  scope: vault
  properties: {
    principalId: deployerPrincipalId
    principalType: 'User'
    roleDefinitionId: keyVaultSecretsOfficerRole
  }
}

resource actionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: 'morgott-preview-alerts'
  location: 'global'
  properties: {
    enabled: true
    groupShortName: 'morgott'
    emailReceivers: [
      {
        name: 'Waleed'
        emailAddress: alertEmail
        useCommonAlertSchema: true
      }
    ]
  }
}

resource app 'Microsoft.App/containerApps@2026-01-01' = if (deployApplication) {
  name: 'morgott-api'
  location: appLocation
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: managedEnvironment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Multiple'
      ingress: {
        allowInsecure: false
        external: externalIngress
        targetPort: 8000
        traffic: empty(stableRevisionName)
          ? [
              {
                latestRevision: true
                weight: 100
              }
            ]
          : [
              {
                revisionName: stableRevisionName
                weight: 100
              }
            ]
        transport: 'auto'
      }
      registries: [
        {
          identity: identity.id
          server: registry.properties.loginServer
        }
      ]
      secrets: [
        {
          identity: identity.id
          keyVaultUrl: '${vault.properties.vaultUri}secrets/openrouter-api-key'
          name: 'openrouter-api-key'
        }
        {
          identity: identity.id
          keyVaultUrl: '${vault.properties.vaultUri}secrets/morgott-api-key'
          name: 'morgott-api-key'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'morgott-api'
          image: image
          env: [
            {
              name: 'AZURE_CLIENT_ID'
              value: identity.properties.clientId
            }
            {
              name: 'AZURE_SERVICEBUS_FQDN'
              value: '${serviceBus.name}.servicebus.windows.net'
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT_URL'
              value: 'https://${storage.name}.blob.${environment().suffixes.storage}'
            }
            {
              name: 'MORGOTT_DATA_MANIFEST_SHA256'
              value: dataManifestSha256
            }
            {
              name: 'MORGOTT_API_KEY'
              secretRef: 'morgott-api-key'
            }
            {
              name: 'OPENROUTER_API_KEY'
              secretRef: 'openrouter-api-key'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              tcpSocket: {
                port: 8000
              }
              initialDelaySeconds: 60
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/healthz'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 30
              periodSeconds: 10
            }
          ]
          resources: {
            cpu: 2
            memory: '4Gi'
          }
        }
      ]
      scale: {
        maxReplicas: 1
        minReplicas: 1
        rules: [
          {
            name: 'http'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
          {
            name: 'daily-canary'
            custom: {
              type: 'azure-servicebus'
              metadata: {
                messageCount: '1'
                namespace: serviceBus.name
                queueName: queue.name
              }
              identity: identity.id
            }
          }
        ]
      }
    }
  }
  dependsOn: [
    registryPull
    serviceBusReceive
    vaultRead
  ]
}

resource job 'Microsoft.App/jobs@2026-01-01' = if (deployApplication) {
  name: 'morgott-daily-canary'
  location: appLocation
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    environmentId: managedEnvironment.id
    workloadProfileName: 'Consumption'
    configuration: {
      replicaRetryLimit: 2
      replicaTimeout: 300
      scheduleTriggerConfig: {
        cronExpression: '0 2 * * *'
        parallelism: 1
        replicaCompletionCount: 1
      }
      triggerType: 'Schedule'
      registries: [
        {
          identity: identity.id
          server: registry.properties.loginServer
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'enqueue-canary'
          image: image
          command: [
            'python'
            '-m'
            'morgott.azure_app'
            'enqueue-canary'
          ]
          env: [
            {
              name: 'AZURE_CLIENT_ID'
              value: identity.properties.clientId
            }
            {
              name: 'AZURE_SERVICEBUS_FQDN'
              value: '${serviceBus.name}.servicebus.windows.net'
            }
            {
              name: 'MORGOTT_CANARY_COMMAND_ID'
              value: canaryCommandId
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
    }
  }
  dependsOn: [
    registryPull
    serviceBusSend
  ]
}

resource canaryAlert 'Microsoft.Insights/scheduledQueryRules@2023-12-01' = if (deployApplication) {
  name: 'morgott-daily-canary-failures'
  location: location
  kind: 'LogAlert'
  properties: {
    actions: {
      actionGroups: [
        actionGroup.id
      ]
    }
    autoMitigate: true
    criteria: {
      allOf: [
        {
          dimensions: []
          failingPeriods: {
            minFailingPeriodsToAlert: 1
            numberOfEvaluationPeriods: 1
          }
          operator: 'GreaterThan'
          query: '''
            let logs = union isfuzzy=true (datatable(ContainerAppName_s:string, Log_s:string)[]), ContainerAppConsoleLogs_CL
              | where ContainerAppName_s == "morgott-api";
            let failures = toscalar(logs
              | where TimeGenerated > ago(1h)
              | where Log_s has "daily_canary_failed"
              | summarize count());
            let completions = toscalar(logs
              | where TimeGenerated > ago(26h)
              | where Log_s has "daily_canary_complete"
              | summarize count());
            print alertCount = failures + iff(completions == 0, 1, 0)
              | where alertCount > 0
          '''
          threshold: 0
          timeAggregation: 'Count'
        }
      ]
    }
    description: 'Daily Morgott canary failed or produced no result without recording prompt text.'
    displayName: 'Morgott daily canary failures or missing results'
    enabled: true
    evaluationFrequency: 'PT1H'
    scopes: [
      workspace.id
    ]
    severity: 2
    windowSize: 'P2D'
  }
}

module budget 'budget.bicep' = {
  name: 'morgott-preview-budget'
  scope: subscription()
  params: {
    alertEmail: alertEmail
    resourceGroupName: resourceGroup().name
    startDate: budgetStartDate
  }
}
