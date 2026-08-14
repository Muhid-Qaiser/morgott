targetScope = 'subscription'

param alertEmail string
param startDate string
param resourceGroupName string

resource budget 'Microsoft.Consumption/budgets@2024-08-01' = {
  name: 'morgott-preview-monthly'
  properties: {
    amount: 100
    category: 'Cost'
    filter: {
      dimensions: {
        name: 'ResourceGroupName'
        operator: 'In'
        values: [resourceGroupName]
      }
    }
    timeGrain: 'Monthly'
    timePeriod: {
      endDate: '2036-01-01T00:00:00Z'
      startDate: startDate
    }
    notifications: toObject([50, 80, 100], threshold => 'Actual_GreaterThan_${threshold}_Percent', threshold => {
        contactEmails: [alertEmail]
        enabled: true
        operator: 'GreaterThan'
        threshold: threshold
        thresholdType: 'Actual'
    })
  }
}
