
import os
import time
import json
import requests
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# --- Configuration ---
GITHUB_API_URL = "https://api.github.com/search/repositories"
SAVE_DIR = "./data/crawled_indicators"
TOPICS = [
    "trading-indicator",
    "technical-analysis",
    "pine-script",
    "trading-algorithms",
    "quantitative-finance"
]
LANGUAGES = ["Python", "Pine Script", "C++", "C#", "EasyLanguage"]

@dataclass
class IndicatorRepo:
    name: str
    url: str
    description: str
    language: str
    stars: int
    last_update: str
    topics: List[str]
    license: Optional[str] = None

class GitHubCrawler:
    def __init__(self, token: Optional[str] = None):
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self.headers["Authorization"] = f"token {token}"
            
    def search_repositories(self, query: str, min_stars: int = 10) -> List[IndicatorRepo]:
        """Search GitHub for repositories matching the query."""
        results = []
        page = 1
        
        # Build search query
        q = f"{query} stars:>{min_stars}"
        
        while True:
            print(f"Searching: {q} | Page: {page}")
            params = {
                "q": q,
                "sort": "stars",
                "order": "desc",
                "per_page": 30, # Max per page
                "page": page
            }
            
            try:
                resp = requests.get(GITHUB_API_URL, headers=self.headers, params=params)
                if resp.status_code == 403:
                    print("Rate limit exceeded. Waiting 60s...")
                    time.sleep(60)
                    continue
                elif resp.status_code != 200:
                    print(f"Error: {resp.status_code} - {resp.text}")
                    break
                    
                data = resp.json()
                items = data.get("items", [])
                
                if not items:
                    break
                    
                for item in items:
                    repo = IndicatorRepo(
                        name=item["full_name"],
                        url=item["html_url"],
                        description=item["description"] or "",
                        language=item["language"] or "Unknown",
                        stars=item["stargazers_count"],
                        last_update=item["updated_at"],
                        topics=item.get("topics", []),
                        license=item["license"]["name"] if item["license"] else "None"
                    )
                    results.append(repo)
                    
                page += 1
                if page > 3: # Limit to 3 pages per query for safety
                    break
                
                # Respect rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Exception: {e}")
                break
                
        return results

    def run_collection(self):
        """Run the collection process across multiple keywords."""
        all_repos = {}
        
        # Create Save Dir
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        for topic in TOPICS:
            print(f"\n--- Collecting for topic: {topic} ---")
            repos = self.search_repositories(f"topic:{topic}")
            
            for r in repos:
                if r.name not in all_repos:
                    all_repos[r.name] = r
            
            print(f"Found {len(repos)} repositories for {topic}")
            time.sleep(5)
            
        # Filter by language logic if needed, but we keep broad for now
        
        # Convert to list
        final_list = list(all_repos.values())
        print(f"\nTotal Unique Repositories Collected: {len(final_list)}")
        
        # Save to JSON
        output_path = f"{SAVE_DIR}/indicators_github.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in final_list], f, indent=4, ensure_ascii=False)
            
        print(f"Saved database to {output_path}")

if __name__ == "__main__":
    print("Starting GitHub Indicator Crawler...")
    # Recommendation: Provide a token if running heavily
    # token = os.getenv("GITHUB_TOKEN") 
    crawler = GitHubCrawler()
    crawler.run_collection()
