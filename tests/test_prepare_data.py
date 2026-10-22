import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from asr_lab.prepare_data import build_records, check_space, ensure_disjoint, safe_extract


class ArchiveTests(unittest.TestCase):
    def test_rejects_traversal_and_links_without_extracting(self):
        for name, kind in (("../outside", tarfile.REGTYPE), ("LibriSpeech/link", tarfile.SYMTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "unsafe.tar.gz"
                with tarfile.open(archive, "w:gz") as output:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "/etc/passwd"
                    output.addfile(member)
                with self.assertRaises(ValueError):
                    safe_extract(archive, root, "dev-clean")
                self.assertFalse((root / "LibriSpeech").exists())

    def test_safe_archive_and_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "safe.tar.gz"
            payload = b"1-2-3 HELLO\n"
            with tarfile.open(archive, "w:gz") as output:
                member = tarfile.TarInfo("LibriSpeech/dev-clean/1/2/1-2.trans.txt")
                member.size = len(payload)
                output.addfile(member, io.BytesIO(payload))
            extracted = safe_extract(archive, root, "dev-clean")
            self.assertEqual((extracted / "1/2/1-2.trans.txt").read_bytes(), payload)
            self.assertEqual(safe_extract(archive, root, "dev-clean"), extracted)

    def test_space_reserve_and_project_quota(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usage = type("Usage", (), {"free": 250 * 1024**3})()
            with patch("asr_lab.prepare_data.shutil.disk_usage", return_value=usage):
                check_space(root, 200, 120)
                with self.assertRaises(RuntimeError):
                    check_space(root, 200, 120, 51 * 1024**3)
                with self.assertRaises(RuntimeError):
                    check_space(root, 0, 1, 2 * 1024**3)


class ManifestTests(unittest.TestCase):
    def test_each_cross_split_identity_is_guarded(self):
        base = {"id": "1-2-3", "speaker_id": "1", "sha256": "a"}
        for key in base:
            other = {"id": "4-5-6", "speaker_id": "4", "sha256": "b"}
            other[key] = base[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "leakage"):
                ensure_disjoint({"train-clean-100": [base], "dev-clean": [other]})

    def test_same_text_is_retained_but_invalid_audio_is_reported(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.joinpath("1-2.trans.txt").write_text("1-2-3 HELLO\n1-2-4 HELLO\n1-2-5 LONG\n1-2-6 MISSING\n")
            sf.write(root / "1-2-3.flac", np.full(1600, 0.1), 16000)
            sf.write(root / "1-2-4.flac", np.full(1600, 0.2), 16000)
            sf.write(root / "1-2-5.flac", np.full(16000, 0.1), 16000)
            records, rejects = build_records(root, "dev-clean", max_duration=0.5)
            self.assertEqual([r["text"] for r in records], ["HELLO", "HELLO"])
            self.assertEqual(len(rejects), 2)
            self.assertTrue(any("duration_out_of_range" in r["reason"] for r in rejects))
            self.assertTrue(all(r["sample_rate"] == 16000 for r in records))
            self.assertNotEqual(records[0]["sha256"], records[1]["sha256"])


if __name__ == "__main__":
    unittest.main()
