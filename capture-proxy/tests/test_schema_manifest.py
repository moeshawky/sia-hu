from __future__ import annotations

import json

from capture_proxy.schema import read_schema_manifest, write_schema_manifest


class TestSchemaManifestWriteFailure:
    """L-05: write_schema_manifest returns None on OSError, never raises.

    Pins: src/capture_proxy/schema.py:141-175 — the except OSError: return None
    at lines 173-174 must swallow write failures so a missing manifest never
    blocks capture.
    """

    def test_schema_write_oserror_returns_none_not_raises(
        self, tmp_path, monkeypatch
    ) -> None:
        def boom(*args, **kwargs) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.write_text", boom)
        assert write_schema_manifest(tmp_path, blobs_enabled=True, run_id=None) is None


class TestSchemaManifestHappyPath:
    """L-05 happy path: corpus + per-run manifests exist and are valid JSON dicts.

    Pins: src/capture_proxy/schema.py:163-175 — both schema.json and
    runs/{run_id}.schema.json are written when run_id is provided.
    """

    def test_schema_write_creates_corpus_and_per_run_manifest(self, tmp_path) -> None:
        run_id = "run-abc123"
        result = write_schema_manifest(tmp_path, blobs_enabled=True, run_id=run_id)

        assert result == tmp_path / "schema.json"

        corpus = json.loads((tmp_path / "schema.json").read_text(encoding="utf-8"))
        assert isinstance(corpus, dict)
        assert corpus["schema_version"] == 6

        per_run = json.loads(
            (tmp_path / "runs" / f"{run_id}.schema.json").read_text(encoding="utf-8")
        )
        assert isinstance(per_run, dict)
        assert per_run["run_id"] == run_id


class TestSchemaManifestReadFailure:
    """L-06: read_schema_manifest returns None on absent / corrupt / non-dict.

    Pins: src/capture_proxy/schema.py:178-193 — OSError and JSONDecodeError are
    caught, and non-dict payloads are rejected.
    """

    def test_read_schema_manifest_absent_returns_none(self, tmp_path) -> None:
        assert read_schema_manifest(tmp_path) is None

    def test_read_schema_manifest_corrupt_returns_none(self, tmp_path) -> None:
        (tmp_path / "schema.json").write_text("{not json", encoding="utf-8")
        assert read_schema_manifest(tmp_path) is None

    def test_read_schema_manifest_non_dict_returns_none(self, tmp_path) -> None:
        (tmp_path / "schema.json").write_text("[1,2,3]", encoding="utf-8")
        assert read_schema_manifest(tmp_path) is None
