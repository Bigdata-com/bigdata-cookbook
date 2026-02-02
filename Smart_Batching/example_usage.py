"""
Example usage of the smart batching search function.

This script demonstrates the two-step workflow:
1. Planning: Organize the search and get total expected chunks
2. Execution: Execute search with proportional sampling
"""

from search_function import plan_search, execute_search, save_plan, load_plan
import os

def main():
    # Check for API key
    if not os.getenv("BIGDATA_API_KEY"):
        print("ERROR: BIGDATA_API_KEY environment variable not set")
        print("Please set it before running this example:")
        print("  export BIGDATA_API_KEY='your_api_key_here'")
        return
    
    print("=" * 80)
    print("Smart Batching Search Example")
    print("=" * 80)
    
    # Step 1: Plan the search
    print("\nStep 1: Planning search...")
    print("-" * 80)
    
    plan = plan_search(
        text="earnings revenue profit",
        universe_csv_path="us_top3000.csv",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"✓ Planning complete!")
    print(f"  Total expected chunks: {plan['total_expected_chunks']:,}")
    print(f"  Number of baskets: {len(plan['baskets'])}")
    print(f"  Companies processed: {plan['planning_metadata'].get('total_companies', 'N/A')}")
    
    # Show example query structure
    if plan['baskets']:
        example_query = plan['baskets'][0]['query']
        print(f"\n  Example query text: '{example_query['text']}'")
        print(f"  Example max_chunks: {example_query['max_chunks']}")
    
    # Step 2: Execute search with different percentages
    print("\n" + "=" * 80)
    print("Step 2: Executing searches with different percentages")
    print("-" * 80)
    
    percentages = [0.1, 0.25, 0.5]
    
    for pct in percentages:
        print(f"\nExecuting with {pct*100:.0f}% of chunks...")
        results = execute_search(
            search_plan=plan,
            chunk_percentage=pct,
            requests_per_minute=100,
            sort_results=True
        )
        
        print(f"  ✓ Retrieved {len(results):,} chunks")
        if results:
            print(f"  Top result relevance: {results[0].get('relevance', 0):.3f}")
            print(f"  Sample text: {results[0].get('text', '')[:80]}...")
    
    # Optional: Save plan for later use
    print("\n" + "=" * 80)
    print("Optional: Saving plan for later use")
    print("-" * 80)
    
    save_plan(plan, "example_search_plan.json")
    print("✓ Plan saved to 'example_search_plan.json'")
    print("\nYou can load it later with:")
    print("  plan = load_plan('example_search_plan.json')")
    print("  results = execute_search(plan, chunk_percentage=0.1)")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
