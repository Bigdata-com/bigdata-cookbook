#!/usr/bin/env python3
"""
Execution script for smart batching.

This script provides a command-line interface to the SmartBatchingPlanner,
allowing users to generate optimized batching plans for semantic search queries.

Usage:
    python run_smart_batching.py --topic "earnings outperforming expectations"
    python run_smart_batching.py --topic "earnings outperforming expectations" --output report.json
    python run_smart_batching.py --topic "earnings outperforming expectations" --universe custom_universe.csv
"""

import argparse
import os
import sys

from smart_batching import SmartBatchingPlanner


def main():
    parser = argparse.ArgumentParser(
        description="Smart Batching for Semantic Search Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default topic
  python run_smart_batching.py

  # Custom topic
  python run_smart_batching.py --topic "AI adoption in healthcare"

  # Save JSON report
  python run_smart_batching.py --topic "earnings outperforming expectations" --output report.json

  # Custom universe file
  python run_smart_batching.py --topic "earnings" --universe my_companies.csv
        """
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="earnings outperforming expectations",
        help="Topic string for comention and semantic search queries",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON report (optional)",
    )
    parser.add_argument(
        "--entities-csv",
        type=str,
        default="output/entities_baskets.csv",
        help="Path for entities CSV file (default: output/entities_baskets.csv)",
    )
    parser.add_argument(
        "--baskets-csv",
        type=str,
        default="output/baskets_details.csv",
        help="Path for baskets CSV file (default: output/baskets_details.csv)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Path to universe CSV file (default: us_top3000.csv)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="BigData API key (or set BIGDATA_API_KEY environment variable)",
    )

    args = parser.parse_args()

    # Validate API key
    api_key = args.api_key or os.getenv("BIGDATA_API_KEY")
    if not api_key:
        print("Error: API key must be provided via --api-key or BIGDATA_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    # Initialize planner
    try:
        planner = SmartBatchingPlanner(api_key=api_key)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Override universe path if provided
    if args.universe:
        from smart_batching_config import UNIVERSE_CSV_PATH
        import smart_batching_config
        smart_batching_config.UNIVERSE_CSV_PATH = args.universe

    # Ensure output directory exists
    entities_dir = os.path.dirname(args.entities_csv) or "output"
    baskets_dir = os.path.dirname(args.baskets_csv) or "output"
    os.makedirs(entities_dir, exist_ok=True)
    os.makedirs(baskets_dir, exist_ok=True)

    # Generate plan
    print(f"Generating smart batching plan for topic: '{args.topic}'")
    print("This may take a while as we query the comention endpoint for all companies...")
    print()

    try:
        report = planner.plan_all_periods(topic=args.topic)
        
        # Generate and print report
        report_text = planner.generate_report(report, output_path=args.output)
        print(report_text)
        
        # Export to CSVs (always export by default)
        print(f"\n{'=' * 80}")
        print("Exporting to CSV files...")
        print(f"{'=' * 80}")
        try:
            entities_csv, baskets_csv = planner.export_to_csvs(
                report,
                entities_csv_path=args.entities_csv,
                baskets_csv_path=args.baskets_csv,
            )
            print(f"✓ Entities CSV saved to: {entities_csv}")
            print(f"✓ Baskets CSV saved to: {baskets_csv}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating plan: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
