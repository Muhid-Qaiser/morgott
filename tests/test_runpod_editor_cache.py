import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITOR_CACHE = ROOT / "scripts" / "runpod-editor-cache.sh"


class RunpodEditorCacheTests(unittest.TestCase):
    def test_bootstrap_helper_hashes_match_reviewed_scripts(self) -> None:
        bootstrap = (ROOT / "scripts" / "runpod-bootstrap.sh").read_text()
        for variable, helper in (
            ("EDITOR_CACHE_SHA256", "runpod-editor-cache.sh"),
            ("SETUP_REMOTE_USER_SHA256", "runpod-setup-remote-user.sh"),
        ):
            expected = re.search(rf"^{variable}=([0-9a-f]{{64}})$", bootstrap, re.M)
            self.assertIsNotNone(expected)
            actual = hashlib.sha256(
                (ROOT / "scripts" / helper).read_bytes()
            ).hexdigest()
            self.assertEqual(expected[1], actual)

    def test_restore_uses_legacy_gzip_archive_when_zstd_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            cache = root / "volume" / "home" / "editor-cache"
            extension = home / ".vscode-server" / "extensions" / "publisher.ext"
            extension.mkdir(parents=True)
            (extension / "extension.txt").write_text("cached\n")
            cache.mkdir(parents=True)
            subprocess.run(
                [
                    "tar",
                    "-C",
                    str(home),
                    "-czf",
                    str(cache / ".vscode-server.tar.gz"),
                    ".vscode-server",
                ],
                check=True,
            )
            shutil.rmtree(home / ".vscode-server")

            bin_dir = root / "bin"
            bin_dir.mkdir()
            zstd = bin_dir / "zstd"
            zstd.write_text("#!/bin/sh\nexit 1\n")
            zstd.chmod(0o755)
            env = os.environ | {
                "HOME": str(home),
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "VOL": str(root / "volume"),
            }

            subprocess.run([EDITOR_CACHE, "restore"], check=True, env=env)

            self.assertEqual((extension / "extension.txt").read_text(), "cached\n")


if __name__ == "__main__":
    unittest.main()
