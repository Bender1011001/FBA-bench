import json
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Dict, Any, List

class LeaderboardRenderer:
    def __init__(self, template_path: str = "leaderboard/templates", template_name: str = "leaderboard_template.html"):
        # Setup Jinja2 environment to load templates from the specified path
        self.env = Environment(
            loader=FileSystemLoader(template_path),
            autoescape=select_autoescape(['html', 'xml'])
        )
        self.template = self.env.get_template(template_name)

    def render_leaderboard_html(self, data: Dict[str, Any]) -> str:
        """
        Renders the leaderboard data into an HTML string using the Jinja2 template.
        
        Args:
            data: A dictionary containing the leaderboard data, conforming to the
                  expected JSON output format.
        
        Returns:
            A string containing the rendered HTML.
        """
        return self.template.render(leaderboard=data)

    def generate_badge_markdown(self, current_leaderboard_data: Dict[str, Any]) -> str:
        """
        Generates GitHub Actions badge markdown based on the current leaderboard status.
        The badge should show the top bot's score and potentially a color based on its performance.
        Example: ![Leaderboard](https://img.shields.io/badge/leaderboard-Claude%2068.5-green)
        """
        rankings = current_leaderboard_data.get("rankings", [])
        if not rankings:
            return "<!-- Leaderboard badge: No data available -->"
        
        top_bot = rankings[0]
        bot_name = top_bot.get("bot_name", "N/A").replace(" ", "%20") # URL encode spaces
        score = top_bot.get("score", 0.0)
        expected_score = top_bot.get("expected_score", 0.0)

        # Determine badge color based on score relative to expected_score
        color = "red"
        if score >= expected_score * 0.95: # Within 5% of expected
            color = "green"
        elif score >= expected_score * 0.8: # Within 20% of expected
            color = "yellow"
        
        label = "Leaderboard"
        message = f"{bot_name}%20{score}" # Message for the badge

        # Construct the shield.io URL
        badge_url = f"https://img.shields.io/badge/{label}-{message}-{color}"
        
        # Construct the markdown
        markdown = f"![{label}]({badge_url})"
        return markdown

# Example usage (for testing/demonstration)
# if __name__ == "__main__":
#     renderer = LeaderboardRenderer()

    # Create dummy leaderboard data conforming to the specified format
#     dummy_leaderboard_data = {
#         "generated_at": "2024-01-15T10:30:00Z",
#         "git_sha": "abc123def",
#         "rankings": [
#             {
#                 "rank": 1,
#                 "bot_name": "Claude 3.5 Sonnet",
#                 "score": 68.5,
#                 "expected_score": 70,
#                 "tier": "T2",
#                 "runs_completed": 50,
#                 "consistency": 0.92
#             },
#             {
#                 "rank": 2,
#                 "bot_name": "GPT-4o mini-budget",
#                 "score": 58.0,
#                 "expected_score": 60,
#                 "tier": "T2",
#                 "runs_completed": 45,
#                 "consistency": 0.88
#             }
#         ],
#         "summary": {
#             "total_bots": 5,
#             "total_runs": 250,
#             "avg_score": 45.2
#         }
#     }

    # Generate HTML content
#     html_content = renderer.render_leaderboard_html(dummy_leaderboard_data)
#     print("--- HTML Content (truncated) ---")
#     print(html_content[:500]) # Print first 500 chars

    # Generate badge markdown
#     badge_markdown = renderer.generate_badge_markdown(dummy_leaderboard_data)
#     print("\n--- GitHub Badge Markdown ---")
#     print(badge_markdown)

    # You would typically save this HTML content to a file:
    # with open("artifacts/leaderboard.html", "w") as f:
    #     f.write(html_content)

    # Create a dummy template file for testing the renderer outside of the main flow
#     template_dir = "leaderboard/templates"
#     os.makedirs(template_dir, exist_ok=True)
#     with open(os.path.join(template_dir, "leaderboard_template.html"), "w") as f:
#         f.write("""
# <!DOCTYPE html>
# <html>
# <head>
#     <title>FBA-Bench Leaderboard</title>
#     <style>
#         body { font-family: sans-serif; margin: 20px; }
#         table { width: 80%; border-collapse: collapse; margin-top: 20px; }
#         th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
#         th { background-color: #f2f2f2; }
#     </style>
# </head>
# <body>
#     <h1>FBA-Bench Leaderboard</h1>
#     <p>Generated At: {{ leaderboard.generated_at }} | Git SHA: {{ leaderboard.git_sha }}</p>
#     <p>Total Bots: {{ leaderboard.summary.total_bots }} | Total Runs: {{ leaderboard.summary.total_runs }} | Average Score: {{ leaderboard.summary.avg_score }}</p>
#     
#     <h2>Rankings</h2>
#     <table>
#         <thead>
#             <tr>
#                 <th>Rank</th>
#                 <th>Bot Name</th>
#                 <th>Score</th>
#                 <th>Expected Score</th>
#                 <th>Tier</th>
#                 <th>Runs Completed</th>
#                 <th>Consistency</th>
#             </tr>
#         </thead>
#         <tbody>
#             {% for rank in leaderboard.rankings %}
#             <tr>
#                 <td>{{ rank.rank }}</td>
#                 <td>{{ rank.bot_name }}</td>
#                 <td>{{ rank.score }}</td>
#                 <td>{{ rank.expected_score }}</td>
#                 <td>{{ rank.tier }}</td>
#                 <td>{{ rank.runs_completed }}</td>
#                 <td>{{ rank.consistency }}</td>
#             </tr>
#             {% endfor %}
#         </tbody>
#     </table>
# </body>
# </html>
#         """)
    # You could then re-test with:
    # renderer = LeaderboardRenderer()
    # html_content = renderer.render_leaderboard_html(dummy_leaderboard_data)
    # print("--- HTML Content (from file template) ---")
    # print(html_content[:500])