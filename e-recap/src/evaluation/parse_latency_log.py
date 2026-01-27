"""
Parse latency log files and convert to JSON format
Supports parsing from inference output logs
"""
import re
import json
import argparse
import os


def parse_log_file(log_path, baseline_out, erecap_out):
    """
    Parse log file with format:
    [Length 4096] baseline=0.7065s  erecap=0.2527s  speedup=2.80x
    
    Args:
        log_path: Path to log file
        baseline_out: Output path for baseline JSON
        erecap_out: Output path for E-RECAP JSON
    """
    # Pattern to match: [Length 4096] baseline=0.7065s  erecap=0.2527s  speedup=2.80x
    pattern = re.compile(
        r"\[Length\s+(\d+)\].*baseline=([\d.]+)s.*erecap=([\d.]+)s"
    )
    
    baseline = {}
    erecap = {}
    
    if not os.path.exists(log_path):
        print(f"[Error] Log file not found: {log_path}")
        return
    
    print(f"[Parsing] Reading log file: {log_path}")
    
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                length = int(match.group(1))
                baseline_latency = float(match.group(2))
                erecap_latency = float(match.group(3))
                
                baseline[length] = baseline_latency
                erecap[length] = erecap_latency
                
                print(f"  Found: Length={length}, baseline={baseline_latency:.4f}s, "
                      f"erecap={erecap_latency:.4f}s")
    
    if not baseline:
        print("[Warning] No matching patterns found in log file")
        print("[Info] Expected format: [Length 4096] baseline=0.7065s  erecap=0.2527s  speedup=2.80x")
        return
    
    # Save as JSON with string keys (will be converted to int when loading)
    os.makedirs(os.path.dirname(baseline_out), exist_ok=True)
    with open(baseline_out, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    os.makedirs(os.path.dirname(erecap_out), exist_ok=True)
    with open(erecap_out, 'w') as f:
        json.dump(erecap, f, indent=2)
    
    print(f"[OK] Parsed {len(baseline)} data points")
    print(f"[OK] Baseline data saved to: {baseline_out}")
    print(f"[OK] E-RECAP data saved to: {erecap_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse latency log files and convert to JSON"
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to log file to parse"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/latency_baseline.json",
        help="Output path for baseline JSON"
    )
    parser.add_argument(
        "--erecap",
        type=str,
        default="results/latency_erecap.json",
        help="Output path for E-RECAP JSON"
    )
    
    args = parser.parse_args()
    
    parse_log_file(args.log, args.baseline, args.erecap)


if __name__ == "__main__":
    main()

