import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from leaderboard.score_tracker import ScoreTracker
from leaderboard.leaderboard_renderer import LeaderboardRenderer
from metrics.metric_suite import MetricSuite # To potentially calculate scores
# Assuming a way to load bot configurations (e.g., from bot_factory or directly)
# Assuming a way to access historical run data (e.g., from a database or file storage)

class LeaderboardManager:
    def __init__(self, score_tracker: ScoreTracker, leaderboard_renderer: LeaderboardRenderer,
                 artifacts_dir: str = "artifacts", bot_configs_dir: str = "baseline_bots/configs"):
        self.score_tracker = score_tracker
        self.leaderboard_renderer = leaderboard_renderer
        self.artifacts_dir = artifacts_dir
        self.bot_configs_dir = bot_configs_dir
        
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.bot_expected_scores = self._load_bot_expected_scores()


    def _load_bot_expected_scores(self) -> Dict[str, float]:
        """Loads expected scores for bots from their configuration files."""
        expected_scores = {}
        for filename in os.listdir(self.bot_configs_dir):
            if filename.endswith(".yaml"):
                filepath = os.path.join(self.bot_configs_dir, filename)
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
                    bot_name = config.get("bot_name")
                    expected_score = config.get("expected_score")
                    if bot_name and expected_score is not None:
                        expected_scores[bot_name] = expected_score
        return expected_scores

    async def update_leaderboard(self, agent_id: str, tier: str, metric_results: Dict[str, Any]):
        """
        Updates the leaderboard with new metric results for a bot's run.
        Expected metric_results structure aligns with MetricSuite's output.
        """
        score = metric_results.get("score")
        if score is None:
            raise ValueError("Metric results must contain a 'score' field.")

        self.score_tracker.add_run_result(
            bot_name=agent_id,
            tier=tier,
            score=score,
            run_details=metric_results # Store full details for history/analysis
        )
        await self.generate_leaderboard_artifacts()

    async def generate_leaderboard_artifacts(self):
        """
        Generates static leaderboard artifacts (JSON and HTML) based on current tracked scores.
        """
        all_tracked_scores = self.score_tracker.get_all_tracked_scores()
        
        # Prepare data for rendering
        render_data = self._prepare_leaderboard_data(all_tracked_scores)

        # Generate JSON artifact
        json_output_path = os.path.join(self.artifacts_dir, "leaderboard.json")
        with open(json_output_path, 'w') as f:
            json.dump(render_data, f, indent=2)
        print(f"Generated JSON leaderboard artifact at {json_output_path}")

        # Generate HTML artifact
        html_output_path = os.path.join(self.artifacts_dir, "leaderboard.html")
        html_content = self.leaderboard_renderer.render_leaderboard_html(render_data)
        with open(html_output_path, 'w') as f:
            f.write(html_content)
        print(f"Generated HTML leaderboard artifact at {html_output_path}")

    def _prepare_leaderboard_data(self, tracked_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates and formats tracked scores into the final leaderboard output structure.
        """
        rankings = []
        for bot_name, tiers_data in tracked_scores.items():
            for tier, runs in tiers_data.items():
                if not runs:
                    continue
                
                # Calculate average score, consistency, etc.
                scores = [run['score'] for run in runs]
                avg_score = sum(scores) / len(scores)

                # Simple consistency: standard deviation (lower is more consistent)
                # Or could use min/max difference, or a custom metric
                import numpy as np
                consistency = 1.0 - (np.std(scores) / avg_score if avg_score > 0 else 0.0) # Invert std dev for "consistency" score

                total_runs_completed = len(runs)
                expected_score = self.bot_expected_scores.get(bot_name, 0)

                rankings.append({
                    "bot_name": bot_name,
                    "score": round(avg_score, 2),
                    "expected_score": expected_score,
                    "tier": tier,
                    "runs_completed": total_runs_completed,
                    "consistency": round(consistency, 3),
                    # Add more stats here like min/max score, cost, token usage from run_details
                    "last_run_details": runs[-1] if runs else None # Include details from the latest run
                })
        
        # Sort rankings by score (descending)
        rankings.sort(key=lambda x: x['score'], reverse=True)

        # Add rank based on sorted order
        for i, entry in enumerate(rankings):
            entry['rank'] = i + 1

        summary = {
            "total_bots": len(self.bot_configs), # Total configured bots
            "total_runs": sum(sum(len(runs) for runs in tiers.values()) for tiers in tracked_scores.values()),
            "avg_score": round(sum(r['score'] for r in rankings) / len(rankings), 2) if rankings else 0.0
        }

        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "git_sha": os.getenv("GITHUB_SHA", "unknown"), # For CI integration
            "rankings": rankings,
            "summary": summary
        }

# For example usage / instantiation:
# tracker = ScoreTracker()
# renderer = LeaderboardRenderer(template_path="leaderboard/leaderboard_template.html") # Need to create this template
# manager = LeaderboardManager(tracker, renderer)
# await manager.update_leaderboard("GPT-3.5", "T0", {"score": 36.5, "cost_usd": 0.01})
# await manager.generate_leaderboard_artifacts()