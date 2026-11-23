from __future__ import annotations

import hashlib

from capture_proxy.blobs import BlobStore, canonical_bytes


class TestBlobStoreIdempotence:
    """L-10: put() is idempotent — existing blob is never rewritten.

    Pins: src/capture_proxy/blobs.py:161-163 — the early return when the
    target already exists and is non-empty, so a repeated turn costs one
    stat and no rewrite.
    """

    def test_blob_put_is_idempotent_no_rewrite(self, tmp_path) -> None:
        store = BlobStore(tmp_path, enabled=True)
        value = {"key": "value"}
        digest = store.put(value)

        first_stat = store.path_for(digest).stat()
        second_digest = store.put(value)
        second_stat = store.path_for(digest).stat()

        assert second_digest == digest
        assert first_stat.st_mtime_ns == second_stat.st_mtime_ns
        assert first_stat.st_ino == second_stat.st_ino


class TestBlobStoreWriteFailure:
    """L-11: digest-on-write-failure — return digest even when write fails.

    Pins: src/capture_proxy/blobs.py:166-167 — OSError from write_bytes is
    swallowed and the digest is still returned so the handle is recorded
    even though the blob is not on disk.
    """

    def test_blob_put_returns_digest_on_write_failure(
        self, tmp_path, monkeypatch
    ) -> None:
        store = BlobStore(tmp_path, enabled=True)
        value = {"key": "value"}
        expected_digest = hashlib.sha256(canonical_bytes(value)).hexdigest()

        def boom(*args, **kwargs) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.write_bytes", boom)
        assert store.put(value) == expected_digest


class TestBlobStoreDisabled:
    """L-10/L-11: disabled store writes nothing and returns None.

    Pins: src/capture_proxy/blobs.py:153-154 — put returns None immediately
    when the store is disabled, so has() is False and the blob dir stays
    empty.
    """

    def test_blob_put_disabled_returns_none_and_writes_nothing(self, tmp_path) -> None:
        store = BlobStore(tmp_path, enabled=False)
        value = {"key": "value"}

        assert store.put(value) is None

        digest = hashlib.sha256(canonical_bytes(value)).hexdigest()
        assert store.has(digest) is False
        assert not store.root.exists()


class TestBlobStoreErrorPaths:
    """Error-path coverage for has(), get(), and put() serialization."""

    def test_blob_has_returns_false_on_stat_oserror(
        self, tmp_path, monkeypatch
    ) -> None:
        store = BlobStore(tmp_path, enabled=True)
        monkeypatch.setattr(
            "pathlib.Path.stat",
            lambda *a, **kw: (_ for _ in ()).throw(OSError()),
        )
        assert store.has("0" * 64) is False

    def test_blob_get_returns_none_on_read_oserror(self, tmp_path, monkeypatch) -> None:
        store = BlobStore(tmp_path, enabled=True)
        monkeypatch.setattr(
            "pathlib.Path.read_bytes",
            lambda *a, **kw: (_ for _ in ()).throw(OSError()),
        )
        assert store.get("0" * 64) is None

    def test_blob_put_returns_none_for_unserializable_value(
        self, tmp_path, monkeypatch
    ) -> None:
        store = BlobStore(tmp_path, enabled=True)
        monkeypatch.setattr(
            "capture_proxy.blobs.canonical_bytes",
            lambda _v: (_ for _ in ()).throw(TypeError("unserializable")),
        )
        assert store.put({"key": "value"}) is None
