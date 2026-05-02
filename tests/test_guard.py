import tempfile
import unittest
from pathlib import Path

from puzzle_runner.guard import ForbiddenGuard


class ForbiddenGuardTests(unittest.TestCase):
    def test_finding_includes_matching_pattern_for_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            level = root / "levels_public" / "101"
            level.parent.mkdir()
            level.write_text("old\n", encoding="utf-8")
            guard = ForbiddenGuard(root, ["levels_public/**"])

            level.write_text("new\n", encoding="utf-8")

            findings = guard.check()

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].path, "levels_public/101")
        self.assertEqual(findings[0].reason, "modified forbidden file")
        self.assertEqual(findings[0].pattern, "levels_public/**")


if __name__ == "__main__":
    unittest.main()
