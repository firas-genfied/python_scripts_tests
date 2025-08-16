#!/bin/bash

NAMESPACE="genfied"
OUTPUT_FILE="router-performance-$(date +%Y%m%d-%H%M%S).csv"

# CSV Header
echo "timestamp,pod_name,cpu_cores,memory_mb,restarts,status,node" > $OUTPUT_FILE

echo "📊 Starting detailed performance monitoring..."
echo "💾 Data saved to: $OUTPUT_FILE"

monitor_performance() {
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get detailed pod metrics
    kubectl get pods -n $NAMESPACE -l app=milvus-router -o json | jq -r '
        .items[] | 
        "\(.metadata.name),\(.status.phase),\(.spec.nodeName),\(.status.containerStatuses[0].restartCount)"
    ' | while IFS=',' read -r pod_name status node_name restarts; do
        
        # Get resource usage
        resource_data=$(kubectl top pod $pod_name -n $NAMESPACE --no-headers 2>/dev/null || echo "$pod_name N/A N/A")
        cpu=$(echo $resource_data | awk '{print $2}' | sed 's/m//')
        memory=$(echo $resource_data | awk '{print $3}' | sed 's/Mi//')
        
        # Log to CSV
        echo "$timestamp,$pod_name,$cpu,$memory,$restarts,$status,$node_name" >> $OUTPUT_FILE
        
        # Display real-time
        printf "%-30s CPU: %6s Memory: %8s Status: %s\n" "$pod_name" "${cpu}m" "${memory}Mi" "$status"
    done
}

# Monitor router endpoints performance
test_router_performance() {
    echo "🔥 Testing router response times..."
    
    for pod in $(kubectl get pods -n $NAMESPACE -l app=milvus-router --no-headers -o custom-columns=":metadata.name"); do
        echo "Testing $pod:"
        
        # Test health endpoint response time
        start_time=$(date +%s.%3N)
        kubectl exec $pod -n $NAMESPACE -- curl -s -w "%{time_total}" http://localhost:8000/health -o /dev/null 2>/dev/null
        end_time=$(date +%s.%3N)
        response_time=$(echo "$end_time - $start_time" | bc)
        
        echo "  Health endpoint: ${response_time}s"
        
        # Test topology endpoint
        start_time=$(date +%s.%3N)
        kubectl exec $pod -n $NAMESPACE -- curl -s -w "%{time_total}" http://localhost:8000/topology -o /dev/null 2>/dev/null
        end_time=$(date +%s.%3N)
        response_time=$(echo "$end_time - $start_time" | bc)
        
        echo "  Topology endpoint: ${response_time}s"
    done
}

# Main monitoring loop
while true; do
    echo "======================== $(date) ========================"
    monitor_performance
    test_router_performance
    echo ""
    sleep 15
done
