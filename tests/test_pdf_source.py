"""PDF source fingerprints must survive Git's Windows newline conversion."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("reportlab"), "optional PDF dependencies are not installed")
class PDFSourceDigestTests(unittest.TestCase):
    def test_newlines_are_equivalent_but_content_changes_are_detected(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import build_learning_guide_pdf as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lesson.md"
            with patch.object(builder, "PROJECT_ROOT", root), patch.object(builder, "PDF_BUILD_INPUTS", (source,)):
                try:
                    source.write_bytes(b"# Lesson\nEOS ends a response.\n")
                    builder.source_digest.cache_clear()
                    expected = builder.source_digest()
                    source.write_bytes(b"# Lesson\r\nEOS ends a response.\r\n")
                    builder.source_digest.cache_clear()
                    self.assertEqual(builder.source_digest(), expected)
                    source.write_bytes(b"# Lesson\nChanged content.\n")
                    builder.source_digest.cache_clear()
                    self.assertNotEqual(builder.source_digest(), expected)
                finally:
                    builder.source_digest.cache_clear()
