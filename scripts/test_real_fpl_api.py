#!/usr/bin/env python3
"""
Test script to verify FPL API client works with real API.

This script makes real HTTP requests to the FPL API (free, no auth required).
Run this to verify the API wrapper is working correctly.

Usage:
    python scripts/test_real_fpl_api.py
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.common.fpl_api import FPLApiClient


def main():
    """Test FPL API client with real API."""
    print("=" * 60)
    print("Testing FPL API Client with Real API")
    print("=" * 60)
    print()

    with FPLApiClient() as client:
        # Test 1: Get bootstrap-static
        print("Test 1: Fetching bootstrap-static...")
        try:
            bootstrap = client.get_bootstrap_static()
            print(f"✓ Success! Found {len(bootstrap['events'])} gameweeks")
            print(f"✓ Found {len(bootstrap['teams'])} teams")
            print(f"✓ Found {len(bootstrap['elements'])} players")
            print()
        except Exception as e:
            print(f"✗ Failed: {e}")
            return 1

        # Test 2: Get current gameweek
        print("Test 2: Getting current gameweek...")
        try:
            current_gw = client.get_current_gameweek()
            if current_gw:
                print(f"✓ Current gameweek: {current_gw}")
            else:
                print("✓ Season not started or ended")
            print()
        except Exception as e:
            print(f"✗ Failed: {e}")
            return 1

        # Test 3: Get season string
        print("Test 3: Getting season string...")
        try:
            season = client.get_season_string()
            print(f"✓ Current season: {season}")
            print()
        except Exception as e:
            print(f"✗ Failed: {e}")
            return 1

        # Test 4: Check if last gameweek finished
        print("Test 4: Checking gameweek status...")
        try:
            if current_gw and current_gw > 1:
                finished = client.is_gameweek_finished(current_gw - 1)
                print(f"✓ Gameweek {current_gw - 1} finished: {finished}")
            print()
        except Exception as e:
            print(f"✗ Failed: {e}")
            return 1

        # Test 5: Get fixtures
        print("Test 5: Fetching fixtures...")
        try:
            if current_gw:
                fixtures = client.get_fixtures(gameweek=current_gw)
                print(f"✓ Found {len(fixtures)} fixtures for GW{current_gw}")
            else:
                fixtures = client.get_fixtures()
                print(f"✓ Found {len(fixtures)} total fixtures")
            print()
        except Exception as e:
            print(f"✗ Failed: {e}")
            return 1

        # Test 6: Get player summary (Mohamed Salah, usually ID 350 or similar)
        print("Test 6: Fetching player summary...")
        try:
            # Find Salah's ID from bootstrap
            salah = next(
                (p for p in bootstrap['elements']
                 if 'Salah' in p['web_name']),
                None
            )

            if salah:
                player_id = salah['id']
                print(f"✓ Found player: {salah['web_name']} (ID: {player_id})")

                summary = client.get_player_summary(player_id)
                print(f"✓ Got player history with {len(summary.get('history', []))} gameweeks")
                print()
            else:
                print("✓ Skipped (couldn't find Salah)")
                print()
        except Exception as e:
            print(f"✗ Failed: {e}")
            return 1

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
