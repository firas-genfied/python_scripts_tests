#!/bin/bash
# Setup script for GPU stress testing

set -e

echo "=== GPU Stress Test Setup ==="

# Check if running in the correct directory
if [[ ! -f "rtsp_stream_processor_multiple_cameras.py" ]]; then
    echo "Error: This script should be run from the directory containing your processing code"
    exit 1
fi

# Create test images directory
echo "Creating test images directory..."
mkdir -p test_images

if [[ ! "$(ls -A test_images)" ]]; then
    echo "Warning: test_images directory is empty"
    echo "Please add your 20 test images to the test_images/ directory"
    echo "Supported formats: .jpg, .jpeg, .png, .bmp"
fi

# Create results directory
echo "Creating results directory..."
mkdir -p stress_test_results

# Install additional dependencies if needed
echo "Checking dependencies..."

# Check for pynvml for GPU monitoring
python3 -c "import pynvml" 2>/dev/null || {
    echo "Installing pynvml for GPU monitoring..."
    pip install pynvml
}

# Check for pandas and matplotlib for analysis
python3 -c "import pandas, matplotlib" 2>/dev/null || {
    echo "Installing pandas and matplotlib for analysis..."
    pip install pandas matplotlib
}

# Create sample configuration files
echo "Creating sample configuration files..."
python3 -c "
import json

configs = {
    'baseline_test.json': {
        'images_dir': './test_images',
        'camera_count': 1,
        'fps_per_camera': 1.0,
        'batch_size': 4,
        'batch_interval': 0.3
    },
    'medium_load_test.json': {
        'images_dir': './test_images',
        'camera_count': 4,
        'fps_per_camera': 5.0,
        'batch_size': 8,
        'batch_interval': 0.3
    },
    'high_load_test.json': {
        'images_dir': './test_images',
        'camera_count': 8,
        'fps_per_camera': 5.0,
        'batch_size': 16,
        'batch_interval': 0.2
    }
}

import os
for filename, config in configs.items():
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f'Created {filename}')
"

# Create a quick test script
echo "Creating quick test script..."
cat > quick_test.sh << 'EOF'
#!/bin/bash
# Quick validation test

echo "Running quick validation test..."
python3 gpu_stress_test.py \
    --images-dir ./test_images \
    --camera-count 2 \
    --fps-per-camera 3.0 \
    --batch-size 8 \
    --duration 30 \
    --output quick_test_results.json

echo "Quick test completed. Check quick_test_results.json for results."
EOF

chmod +x quick_test.sh

# Create comprehensive test script
echo "Creating comprehensive test script..."
cat > run_comprehensive_tests.sh << 'EOF'
#!/bin/bash
# Run comprehensive stress tests

set -e

echo "=== Running Comprehensive GPU Stress Tests ==="

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./stress_test_results/comprehensive_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "Results will be saved to: ${RESULTS_DIR}"

# Test duration (adjust as needed)
DURATION=120  # 2 minutes per test

echo "1. Running baseline test..."
python3 run_stress_tests.py --single --config baseline_test.json --duration ${DURATION} --results-dir "${RESULTS_DIR}"

echo "2. Running batch size optimization..."
python3 run_stress_tests.py --batch-optimization --duration ${DURATION} --results-dir "${RESULTS_DIR}"

echo "3. Running FPS scaling tests..."
python3 run_stress_tests.py --fps-scaling --duration ${DURATION} --results-dir "${RESULTS_DIR}"

echo "4. Running camera scaling tests..."
python3 run_stress_tests.py --camera-scaling --duration ${DURATION} --results-dir "${RESULTS_DIR}"

echo "5. Generating analysis report..."
python3 run_stress_tests.py --analyze --results-dir "${RESULTS_DIR}"

echo "=== Tests completed! ==="
echo "Check ${RESULTS_DIR} for detailed results and report."
EOF

chmod +x run_comprehensive_tests.sh

# Create monitoring script
echo "Creating GPU monitoring script..."
cat > monitor_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
Simple GPU monitoring script to watch GPU usage during tests
"""

import time
import sys
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

def monitor_gpu(interval=1):
    if not NVML_AVAILABLE:
        print("pynvml not available. Install with: pip install pynvml")
        return
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print("GPU Monitoring Started (Ctrl+C to stop)")
        print("Time\t\tGPU%\tMemory(MB)\tTemp(C)")
        print("-" * 50)
        
        while True:
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = mem_info.used / (1024 * 1024)
            mem_total = mem_info.total / (1024 * 1024)
            
            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp}\t{util.gpu}%\t{mem_used:.0f}/{mem_total:.0f}\t{temp}°C")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    interval = 1
    if len(sys.argv) > 1:
        try:
            interval = float(sys.argv[1])
        except ValueError:
            print("Usage: python3 monitor_gpu.py [interval_seconds]")
            sys.exit(1)
    
    monitor_gpu(interval)
EOF

chmod +x monitor_gpu.py

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Add your test images to the test_images/ directory"
echo "2. Run a quick test: ./quick_test.sh"
echo "3. Monitor GPU during tests: python3 monitor_gpu.py"
echo "4. Run comprehensive tests: ./run_comprehensive_tests.sh"
echo ""
echo "Individual test commands:"
echo "- Baseline test: python3 run_stress_tests.py --single --config baseline_test.json --duration 60"
echo "- Batch optimization: python3 run_stress_tests.py --batch-optimization --duration 60"
echo "- FPS scaling: python3 run_stress_tests.py --fps-scaling --duration 60"
echo "- Camera scaling: python3 run_stress_tests.py --camera-scaling --duration 60"
echo ""
echo "Files created:"
echo "- test_images/ (directory for your test images)"
echo "- stress_test_results/ (directory for results)"
echo "- *.json (sample configuration files)"
echo "- quick_test.sh (quick validation test)"
echo "- run_comprehensive_tests.sh (full test suite)"
echo "- monitor_gpu.py (GPU monitoring utility)"
echo ""
