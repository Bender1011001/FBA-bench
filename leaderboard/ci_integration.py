import os
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime

class CIIntegration:
    """
    Utilities for integrating FBA-Bench leaderboard with CI/CD pipelines,
    specifically for GitHub Actions badge generation and regression detection.
    """
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.leaderboard_json_path = os.path.join(self.artifacts_dir, "leaderboard.json")

    def get_latest_leaderboard_data(self) -> Optional[Dict[str, Any]]:
        """Reads the latest generated leaderboard JSON artifact."""
        if os.path.exists(self.leaderboard_json_path):
            with open(self.leaderboard_json_path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {self.leaderboard_json_path}")
                    return None
        return None

    def generate_github_badge_markdown(self, leaderboard_data: Dict[str, Any], renderer_instance: Any) -> str:
        """
        Generates GitHub Markdown for the leaderboard badge.
        Requires an instance of LeaderboardRenderer to use its badge generation logic.
        """
        if not renderer_instance:
            raise ValueError("LeaderboardRenderer instance must be provided to generate badge markdown.")
        
        return renderer_instance.generate_badge_markdown(leaderboard_data)

    def detect_regression(self, new_leaderboard_data: Dict[str, Any], previous_leaderboard_data: Optional[Dict[str, Any]], threshold: float = 0.02) -> List[str]:
        """
        Detects significant score regressions for baseline bots.
        Compares current scores against previous runs.
        
        Args:
            new_leaderboard_data: The latest leaderboard data.
            previous_leaderboard_data: Data from a previous run (e.g., last successful main branch build).
            threshold: Percentage drop (e.g., 0.02 for 2%) to consider a regression.
            
        Returns:
            A list of strings describing detected regressions.
        """
        regressions = []
        if not previous_leaderboard_data:
            return ["No previous leaderboard data available for regression detection."]

        new_rankings = {entry['bot_name']: entry for entry in new_leaderboard_data.get("rankings", [])}
        prev_rankings = {entry['bot_name']: entry for entry in previous_leaderboard_data.get("rankings", [])}

        for bot_name, new_entry in new_rankings.items():
            if bot_name in prev_rankings:
                prev_entry = prev_rankings[bot_name]
                new_score = new_entry.get("score", 0.0)
                prev_score = prev_entry.get("score", 0.0)

                if prev_score > 0 and (prev_score - new_score) / prev_score > threshold:
                    regressions.append(
                        f"Regression detected for {bot_name} (Tier {new_entry.get('tier', 'N/A')}): "
                        f"Score dropped from {prev_score:.2f} to {new_score:.2f} (>{threshold:.0%} drop)."
                    )
        return regressions

    def publish_status_to_github(self, status: str, description: str, context: str, target_url: Optional[str] = None):
        """
        Publishes a GitHub commit status. Requires GITHUB_TOKEN environment variable.
        This is typically used in GitHub Actions.
        
        Args:
            status: "success", "failure", "pending", "error"
            description: Short description of the status.
            context: A string label to differentiate this status from others.
            target_url: URL for the status; typically a link to the leaderboard artifact.
        """
        github_token = os.getenv("GITHUB_TOKEN")
        github_sha = os.getenv("GITHUB_SHA")
        github_repo = os.getenv("GITHUB_REPOSITORY") # e.g., "owner/repo"

        if not github_token or not github_sha or not github_repo:
            print("GitHub token, SHA, or repository not found, skipping status update.")
            return

        api_url = f"https://api.github.com/repos/{github_repo}/statuses/{github_sha}"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "state": status,
            "description": description,
            "context": context
        }
        if target_url:
            payload["target_url"] = target_url

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            print(f"GitHub status published: {status} for {context}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to publish GitHub status: {e}")

# Example Integration Workflow within CI (conceptual):
# (This code would run in a GitHub Actions workflow, not directly from the main simulation)

# async def ci_workflow_example():
#     ci_integrator = CIIntegration()
#     
#     # 1. Get latest leaderboard data
#     current_leaderboard = ci_integrator.get_latest_leaderboard_data()
#     if not current_leaderboard:
#         print("No current leaderboard data found. Run simulations first.")
#         return

#     # 2. Generate badge markdown
#     from leaderboard.leaderboard_renderer import LeaderboardRenderer
#     renderer = LeaderboardRenderer(template_path="leaderboard/templates") # Ensure template exists
#     badge_markdown = renderer.generate_github_badge_markdown(current_leaderboard)
#     print(f"Generated badge markdown:\n{badge_markdown}")
#     
#     # In a real CI, you'd update a README.md file with this markdown
#     # Example for updating a file (requires read/write permissions for workflow)
#     # with open("README.md", "r+") as f:
#     #     content = f.read()
#     #     # Replace old badge with new, or insert if not present
#     #     new_content = update_badge_in_markdown(content, badge_markdown) # Implement this helper
#     #     f.seek(0)
#     #     f.truncate()
#     #     f.write(new_content)

#     # 3. Fetch previous leaderboard for regression detection (e.g., from an S3 bucket or previous artifact)
#     # For demonstration, let's mock a previous leaderbaord
#     # In a real scenario, you'd fetch this from a persistent store (e.g., S3, Google Cloud Storage, or GitHub Artifacts API)
#     previous_leaderboard = {
#         "generated_at": "2024-01-14T10:00:00Z",
#         "git_sha": "def456ghi",
#         "rankings": [
#             {"bot_name": "Claude 3.5 Sonnet", "score": 71.0, "expected_score": 70, "tier": "T2"},
#             {"bot_name": "GPT-4o mini-budget", "score": 59.0, "expected_score": 60, "tier": "T2"}
#         ]
#     }
#     regressions = ci_integrator.detect_regression(current_leaderboard, previous_leaderboard)
#     if regressions:
#         print("\n--- REGRESSIONS DETECTED ---")
#         for reg in regressions:
#             print(reg)
#             # In CI, you might post a commit status "failure" or open an issue
#             ci_integrator.publish_status_to_github(
#                 status="failure",
#                 description=f"Leaderboard regression: {reg.split(':')[0]}",
#                 context="fba-bench/leaderboard-status",
#                 target_url=f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
#             )
#     else:
#         print("\nNo significant regressions detected.")
#         ci_integrator.publish_status_to_github(
#             status="success",
#             description="Leaderboard scores are stable or improved.",
#             context="fba-bench/leaderboard-status",
#             target_url=f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
#         )

# if __name__ == "__main__":
#     # Example of how you might run this
#     # Make sure to set environment variables for testing GitHub API interaction:
#     # os.environ["GITHUB_TOKEN"] = "YOUR_GITHUB_TOKEN"
#     # os.environ["GITHUB_SHA"] = "SOME_COMMIT_SHA"
#     # os.environ["GITHUB_REPOSITORY"] = "your_username/your_repo"
#     # os.environ["GITHUB_SERVER_URL"] = "https://github.com"
#     # os.environ["GITHUB_RUN_ID"] = "12345" # A dummy run ID
#     import asyncio
#     asyncio.run(ci_workflow_example())