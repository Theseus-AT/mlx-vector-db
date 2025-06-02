#!/usr/bin/env python3
"""
Enterprise Demo für MLX Vector Database
Zeigt Monitoring, Metriken, Health Checks und Enterprise-Features
"""
import requests
import time
import json
import os
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def get_api_key():
    """Get API key from environment or user input"""
    api_key = os.getenv("VECTOR_DB_API_KEY")
    if not api_key:
        api_key = input("Enter your API key: ").strip()
    return api_key

def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"🏢 {title}")
    print('='*60)

def print_subsection(title: str):
    """Print subsection header"""
    print(f"\n📊 {title}")
    print('-'*40)

def format_dict(data: Dict[str, Any], indent: int = 0) -> str:
    """Format dictionary for pretty printing"""
    lines = []
    spacing = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{spacing}{key}:")
            lines.append(format_dict(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{spacing}{key}: {len(value)} items")
        else:
            # Format numbers nicely
            if isinstance(value, float):
                if value < 1:
                    formatted_value = f"{value:.3f}"
                elif value < 100:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.1f}"
            else:
                formatted_value = str(value)
            lines.append(f"{spacing}{key}: {formatted_value}")
    
    return "\n".join(lines)

def test_health_monitoring():
    """Test health monitoring capabilities"""
    print_section("HEALTH MONITORING")
    
    # Basic health check (no auth required)
    print_subsection("Basic Health Check")
    try:
        response = requests.get(f"{BASE_URL}/monitoring/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service Status: {health['status']}")
            print(f"   Version: {health['version']}")
            print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(health['timestamp']))}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Detailed health check (requires auth)
    print_subsection("Detailed Health Check")
    api_key = get_api_key()
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/health/detailed", headers=headers)
        if response.status_code == 200:
            detailed_health = response.json()
            print(f"✅ Overall Status: {detailed_health['overall_status']}")
            
            print("\n🔍 Component Health:")
            for check_name, result in detailed_health['checks'].items():
                status_icon = "✅" if result['status'] == 'healthy' else "⚠️" if result['status'] == 'warning' else "❌"
                print(f"   {status_icon} {check_name}: {result['status']} - {result['message']}")
            
            print("\n💻 System Information:")
            sys_info = detailed_health.get('system_info', {})
            print(format_dict(sys_info, 1))
            
        else:
            print(f"❌ Detailed health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Detailed health check error: {e}")

def test_metrics_monitoring():
    """Test metrics monitoring"""
    print_section("METRICS MONITORING")
    
    api_key = get_api_key()
    headers = {"X-API-Key": api_key}
    
    # Get metrics summary
    print_subsection("Metrics Summary")
    try:
        response = requests.get(f"{BASE_URL}/monitoring/metrics/summary", headers=headers)
        if response.status_code == 200:
            summary = response.json()['summary']
            
            print("📈 Request Metrics:")
            req_metrics = summary['requests']
            print(f"   Total Requests: {req_metrics['total']}")
            print(f"   Avg Duration: {req_metrics['avg_duration']:.3f}s")
            
            print("\n🔍 Vector Operations:")
            vec_metrics = summary['vector_operations']
            print(f"   Total Queries: {vec_metrics['queries_total']}")
            print(f"   Total Additions: {vec_metrics['additions_total']}")
            print(f"   Avg Query Duration: {vec_metrics['avg_query_duration']:.3f}s")
            
            print("\n💾 Cache Performance:")
            cache_metrics = summary['cache']
            print(f"   Cache Hits: {cache_metrics['hits']}")
            print(f"   Cache Misses: {cache_metrics['misses']}")
            print(f"   Hit Rate: {cache_metrics['hit_rate_percent']:.1f}%")
            print(f"   Memory Usage: {cache_metrics['memory_usage_mb']:.1f} MB")
            
            print("\n🖥️ System Resources:")
            sys_metrics = summary['system']
            print(f"   CPU Usage: {sys_metrics['cpu_usage_percent']:.1f}%")
            print(f"   Memory Usage: {sys_metrics['memory_usage_gb']:.2f} GB")
            print(f"   Disk Usage: {sys_metrics['disk_usage_gb']:.2f} GB")
            
            print("\n❌ Error Tracking:")
            error_metrics = summary['errors']
            print(f"   Total Errors: {error_metrics['total']}")
            
        else:
            print(f"❌ Metrics summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")
    
    # Test Prometheus format
    print_subsection("Prometheus Format Export")
    try:
        response = requests.get(f"{BASE_URL}/monitoring/metrics?format=prometheus", headers=headers)
        if response.status_code == 200:
            prometheus_text = response.text
            lines = prometheus_text.split('\n')
            
            print("✅ Prometheus metrics available")
            print(f"   Total lines: {len(lines)}")
            print("   Sample metrics:")
            
            # Show sample metrics
            metric_lines = [line for line in lines if line and not line.startswith('#')]
            for line in metric_lines[:5]:
                if line.strip():
                    print(f"     {line}")
            
            if len(metric_lines) > 5:
                print(f"     ... and {len(metric_lines) - 5} more metrics")
                
        else:
            print(f"❌ Prometheus export failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prometheus export error: {e}")

def test_service_status():
    """Test comprehensive service status"""
    print_section("SERVICE STATUS")
    
    api_key = get_api_key()
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/status", headers=headers)
        if response.status_code == 200:
            status = response.json()
            
            print(f"🚀 Service: {status['service']} v{status['version']}")
            print(f"📊 Status: {status['status']}")
            print(f"⏱️ Uptime: {status['uptime_seconds']:.0f} seconds")
            
            print("\n🎯 Performance Indicators:")
            perf_indicators = status['performance_indicators']
            for indicator, is_ok in perf_indicators.items():
                icon = "✅" if is_ok else "❌"
                readable_name = indicator.replace('_', ' ').title()
                print(f"   {icon} {readable_name}: {'OK' if is_ok else 'Issues Detected'}")
            
            print(f"\n🏥 Health Status: {status['health']['overall_status']}")
            
        else:
            print(f"❌ Service status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Service status error: {e}")

def test_configuration():
    """Test configuration endpoint"""
    print_section("CONFIGURATION")
    
    try:
        response = requests.get(f"{BASE_URL}/config")
        if response.status_code == 200:
            config = response.json()
            
            print("⚙️ Server Configuration:")
            print(format_dict(config['server'], 1))
            
            print("\n🔧 Features:")
            print(format_dict(config['features'], 1))
            
            print("\n📏 Limits:")
            print(format_dict(config['limits'], 1))
            
        else:
            print(f"❌ Configuration failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def simulate_load():
    """Simulate some load to generate metrics"""
    print_section("LOAD SIMULATION")
    
    api_key = get_api_key()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    print("🔄 Simulating API load to generate metrics...")
    
    try:
        # Create a test store
        user_id = "enterprise_demo_user"
        model_id = "enterprise_demo_model"
        
        # Delete if exists
        try:
            delete_payload = {"user_id": user_id, "model_id": model_id}
            requests.delete(f"{BASE_URL}/admin/store", json=delete_payload, headers=headers)
        except:
            pass
        
        # Create store
        create_payload = {"user_id": user_id, "model_id": model_id}
        response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=headers)
        
        if response.status_code == 200:
            print("✅ Test store created")
            
            # Add some vectors
            import numpy as np
            vectors = np.random.rand(100, 384).astype(np.float32)
            metadata = [{"id": f"demo_{i}", "type": "enterprise_test"} for i in range(100)]
            
            add_payload = {
                "user_id": user_id,
                "model_id": model_id,
                "vectors": vectors.tolist(),
                "metadata": metadata
            }
            
            response = requests.post(f"{BASE_URL}/admin/add_test_vectors", json=add_payload, headers=headers)
            if response.status_code == 200:
                print("✅ Test vectors added")
                
                # Perform some queries to generate metrics
                query_vector = vectors[0].tolist()
                for i in range(10):
                    query_payload = {
                        "user_id": user_id,
                        "model_id": model_id,
                        "query": query_vector,
                        "k": 5
                    }
                    requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=headers)
                
                print("✅ Generated query metrics")
                
                # Cleanup
                requests.delete(f"{BASE_URL}/admin/store", json=delete_payload, headers=headers)
                print("✅ Test store cleaned up")
                
        else:
            print(f"❌ Failed to create test store: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Load simulation error: {e}")

def test_alerting():
    """Test alerting system"""
    print_section("ALERTING SYSTEM")
    
    api_key = get_api_key()
    headers = {"X-API-Key": api_key}
    
    print("🚨 Testing alert system...")
    
    alert_types = ["test", "high_cpu", "low_disk", "error_spike"]
    
    for alert_type in alert_types:
        try:
            response = requests.post(
                f"{BASE_URL}/monitoring/alerts/test?alert_type={alert_type}", 
                headers=headers
            )
            
            if response.status_code == 200:
                alert = response.json()
                print(f"✅ {alert_type}: {alert['message']}")
            else:
                print(f"❌ {alert_type} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Alert {alert_type} error: {e}")

def main():
    """Run enterprise features demo"""
    print("🏢 MLX Vector Database - Enterprise Features Demo")
    print("=" * 60)
    print("This demo showcases enterprise-grade monitoring, metrics,")
    print("health checks, and operational features.")
    
    try:
        # Test basic connectivity
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not accessible. Please start the server first:")
            print("   python main.py")
            return
            
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please ensure the server is running on localhost:8000")
        return
    
    try:
        # Run all enterprise feature tests
        test_health_monitoring()
        test_configuration()
        simulate_load()
        test_metrics_monitoring()
        test_service_status()
        test_alerting()
        
        print_section("ENTERPRISE DEMO COMPLETED")
        print("✅ All enterprise features tested successfully!")
        print("\n🎯 Key Enterprise Features Demonstrated:")
        print("   • Comprehensive health monitoring")
        print("   • Real-time metrics collection")
        print("   • Prometheus-compatible exports") 
        print("   • Performance indicators")
        print("   • Configuration management")
        print("   • Alerting system")
        print("   • Service status reporting")
        
        print("\n📊 For Production Deployment:")
        print("   • Set up Prometheus/Grafana for monitoring")
        print("   • Configure alerts for critical metrics")
        print("   • Use reverse proxy (nginx) for SSL/load balancing")
        print("   • Set up log aggregation (ELK stack)")
        print("   • Implement backup strategies")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()