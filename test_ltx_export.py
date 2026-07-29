"""Unit tests for LTX export pack."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image

from ltx_export import export_ltx_pack


class TestLtxExport(unittest.TestCase):
    def test_export_resizes_to_768_and_writes_dataset_json(self):
        root = Path(__file__).resolve().parent / "data" / "_test_ltx_char"
        dataset = root / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        (root / "meta.json").write_text(
            json.dumps({"name": "Test Char", "slug": "_test_ltx_char"}),
            encoding="utf-8",
        )
        for i in range(1, 3):
            stem = f"{i:04d}"
            img = Image.new("RGB", (1200, 900), (40 + i * 20, 80, 120))
            img.save(dataset / f"{stem}.png")
            item = {
                "id": stem,
                "status": "ok",
                "tag": f"tag_{i}",
                "ltxCaption": f"ohwx_test, shot {i}",
                "text": f"shot {i}",
            }
            (dataset / f"{stem}.json").write_text(json.dumps(item), encoding="utf-8")
            (dataset / f"{stem}.txt").write_text(item["ltxCaption"], encoding="utf-8")

        manifest = export_ltx_pack(root, trigger="ohwx_test")
        self.assertEqual(manifest["imageCount"], 2)
        self.assertEqual(manifest["trainSize"], 768)
        self.assertEqual(manifest["bucket"], "768x768x1")
        out = root / "ltx_train"
        img = Image.open(out / "images" / "0001.jpg")
        self.assertEqual(img.size, (768, 768))
        rows = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["caption"].startswith("ohwx_test"))
        self.assertTrue((out / "train_config.yaml").exists())
        self.assertTrue((out / "trigger.txt").exists())


if __name__ == "__main__":
    unittest.main()
