import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation_v1_2.__main__ import load_local_env


class LocalEnvironmentTests(unittest.TestCase):
    def test_loads_only_supported_deepseek_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "DEEPSEEK_API_KEY='sk-test-only'\n"
                "export DEEPSEEK_MODEL=deepseek-v4-flash\n"
                "UNRELATED_SECRET=must-not-load\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_local_env(path)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-test-only")
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "deepseek-v4-flash")
                self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_shell_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("DEEPSEEK_MODEL=file-model\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_MODEL": "shell-model"}, clear=True):
                load_local_env(path)
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "shell-model")


if __name__ == "__main__":
    unittest.main()
