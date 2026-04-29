import unittest

from puzzle_runner.prompts import ScoreFeedback, compose_prompt


class PromptTests(unittest.TestCase):
    def test_initial_prompt_includes_remaining_tries(self) -> None:
        prompt = compose_prompt(
            ScoreFeedback(
                last_score=None,
                best_score=0,
                improved=None,
                stale_count=0,
                stale_limit=3,
                round_number=1,
            )
        )

        self.assertIn("Remaining no-progress tries before stop: 3.", prompt)

    def test_continue_prompt_includes_remaining_tries(self) -> None:
        prompt = compose_prompt(
            ScoreFeedback(
                last_score=47,
                best_score=47,
                improved=False,
                stale_count=2,
                stale_limit=3,
                round_number=4,
            )
        )

        self.assertIn("No-progress count: 2/3.", prompt)
        self.assertIn("Remaining no-progress tries before stop: 1.", prompt)


if __name__ == "__main__":
    unittest.main()
