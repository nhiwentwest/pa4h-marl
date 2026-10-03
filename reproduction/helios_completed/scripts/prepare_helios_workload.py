import argparse
import json
from helios_workload import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--outcome-policy', choices=['completed', 'all_observed'], required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.outcome_policy), indent=2))
