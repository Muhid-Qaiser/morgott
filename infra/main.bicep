targetScope = 'resourceGroup'

@description('Deploy the Container App after the image exists.')
param deployApplication bool = false
param image string = ''
param deployerPrincipalId string
param budgetStartDate string
@description('Existing revision that keeps receiving traffic while the new revision is validated.')
param stableRevisionName string = ''
@description('Preserve external ingress for an existing working deployment during validation.')
param externalIngress bool = false
@allowed([
  '2cpu-4gi'
  '4cpu-8gi'
])
@description('Resource shape for the candidate revision.')
param candidateSize string = '2cpu-4gi'

var location = resourceGroup().location
var appLocation = 'eastus'
var alertEmail = 'waleed@vulsight.com'
var acrPullRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var keyVaultSecretsOfficerRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7')
var keyVaultSecretsUserRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
var candidateCpu = candidateSize == '4cpu-8gi' ? 4 : 2
var candidateMemory = candidateSize == '4cpu-8gi' ? '8Gi' : '4Gi'

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
              name: 'MORGOTT_API_KEY'
              secretRef: 'morgott-api-key'
            }
            {
              name: 'OPENROUTER_API_KEY'
              secretRef: 'openrouter-api-key'
            }
            {
              name: 'OMP_NUM_THREADS'
              value: '1'
            }
            {
              name: 'OPENBLAS_NUM_THREADS'
              value: '1'
            }
            {
              name: 'MKL_NUM_THREADS'
              value: '1'
            }
            {
              name: 'NUMEXPR_NUM_THREADS'
              value: '1'
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
            cpu: candidateCpu
            memory: candidateMemory
          }
        }
      ]
      scale: {
        maxReplicas: 1
        minReplicas: 1
      }
    }
  }
  dependsOn: [
    registryPull
    vaultRead
  ]
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
