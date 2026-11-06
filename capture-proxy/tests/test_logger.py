"""Tests for CaptureLogger — JSONL file writing."""

import json
import uuid
from datetime import UTC, datetime

import pytest

from capture_proxy.logger import CaptureLogger


@pytest.fixture
def logger(tmp_path):
    """Provide CaptureLogger writing to a temporary directory."""
    return CaptureLogger(base_path=str(tmp_path))


@pytest.fixture
def sample_record():
    """Provide a minimal valid CaptureRecord dict."""
    return {
        "schema_version": 2,
        "session_id": str(uuid.uuid4()),
        "user_id": "user-tester",
        "turn_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "latency_ms": 12.5,
        "client_ip": "127.0.0.1",
        "client_user_agent": "test-agent",
        "model": "test-model",
        "run_id": "run_test",
        "runtime_settings": {"backend": "llama.cpp", "kv_cache_k": "q8_0"},
        "request": {"model": "test-model", "messages": []},
        "response": {"id": "chatcmpl-123", "choices": []},
        "tool_calls": None,
        "finish_reason": "stop",
        "upstream_status": 200,
        "error_body": None,
        "routing": None,
    }


class TestLog:
    """Tests for CaptureLogger.log()."""

    def test_log_creates_file(self, logger, sample_record, tmp_path):
        """log() creates a JSONL file in raw/YYYY-MM-DD/."""
        filepath = logger.log(sample_record)
        assert filepath.exists()
        assert "raw" in str(filepath)
        assert filepath.suffix == ".jsonl"

    def test_log_writes_one_line(self, logger, sample_record):
        """log() writes exactly 1 JSONL line."""
        filepath = logger.log(sample_record)
        content = filepath.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1

    def test_log_valid_jsonl(self, logger, sample_record):
        """log() writes valid JSON on the line."""
        filepath = logger.log(sample_record)
        line = filepath.read_text().strip()
        parsed = json.loads(line)
        assert parsed["turn_id"] == sample_record["turn_id"]
        assert parsed["model"] == "test-model"

    def test_log_separate_files_per_request(self, logger, sample_record):
        """Each log() call creates a separate file (different turn_id → different filename)."""
        sample_record["turn_id"] = str(uuid.uuid4())
        filepath1 = logger.log(sample_record)

        sample_record["turn_id"] = str(uuid.uuid4())
        filepath2 = logger.log(sample_record)

        # Different turn_ids → different filenames
        assert filepath1 != filepath2
        assert filepath1.exists()
        assert filepath2.exists()
        # Each file has exactly 1 line
        assert len(filepath1.read_text().strip().split("\n")) == 1
        assert len(filepath2.read_text().strip().split("\n")) == 1

    def test_log_creates_directory(self, logger, sample_record, tmp_path):
        """log() creates directory structure on first write."""
        filepath = logger.log(sample_record)
        assert filepath.parent.exists()
        assert filepath.parent.is_dir()

    def test_log_records_all_fields(self, logger, sample_record):
        """log() preserves all CaptureRecord fields in JSONL."""
        filepath = logger.log(sample_record)
        line = filepath.read_text().strip()
        parsed = json.loads(line)

        expected_fields = [
            "schema_version",
            "session_id",
            "user_id",
            "turn_id",
            "created_at",
            "latency_ms",
            "client_ip",
            "client_user_agent",
            "model",
            "run_id",
            "runtime_settings",
            "request",
            "response",
            "tool_calls",
            "finish_reason",
            "upstream_status",
            "error_body",
            "routing",
        ]
        for field in expected_fields:
            assert field in parsed, f"Missing field: {field}"

    def test_log_routing_always_none(self, logger, sample_record):
        """log() preserves routing=None (R-16 placeholder)."""
        sample_record["routing"] = None
        filepath = logger.log(sample_record)
        line = filepath.read_text().strip()
        parsed = json.loads(line)
        assert parsed["routing"] is None


class TestLogError:
    """Tests for CaptureLogger.log_error()."""

    def test_log_error_creates_file_in_errors(self, logger, sample_record, tmp_path):
        """log_error() creates file in errors/ not raw/."""
        sample_record["error_body"] = "upstream timeout"
        filepath = logger.log_error(sample_record)
        assert "errors" in str(filepath)
        assert filepath.exists()

    def test_log_error_writes_one_line(self, logger, sample_record):
        """log_error() writes exactly 1 JSONL line."""
        sample_record["error_body"] = "connection refused"
        filepath = logger.log_error(sample_record)
        content = filepath.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1

    def test_log_error_preserves_error_body(self, logger, sample_record):
        """log_error() preserves error_body in JSONL."""
        sample_record["error_body"] = "502 Bad Gateway"
        filepath = logger.log_error(sample_record)
        line = filepath.read_text().strip()
        parsed = json.loads(line)
        assert parsed["error_body"] == "502 Bad Gateway"


class TestMalformedTimestampFallback:
    """Adversarial: corrupted created_at falls back to datetime.now() for file path."""

    def test_malformed_timestamp_writes_to_today_directory(self, logger, sample_record):
        """Malformed created_at causes file to be written in today's date directory."""
        from datetime import UTC, datetime

        sample_record["created_at"] = "not-a-date-at-all"
        filepath = logger.log(sample_record)

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert today in str(filepath)
        assert filepath.exists()

    def test_malformed_timestamp_preserves_original_in_content(
        self, logger, sample_record
    ):
        """Malformed created_at is preserved verbatim in the JSONL record."""
        sample_record["created_at"] = "2026-13-45T99:99:99"
        filepath = logger.log(sample_record)

        line = filepath.read_text().strip()
        parsed = json.loads(line)
        assert parsed["created_at"] == "2026-13-45T99:99:99"

    def test_malformed_timestamp_filename_uses_current_time(
        self, logger, sample_record
    ):
        """Filename uses current UTC time when created_at is unparseable."""
        from datetime import UTC, datetime

        sample_record["created_at"] = "garbage"
        filepath = logger.log(sample_record)

        now_prefix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        assert filepath.name.startswith(now_prefix)
        assert sample_record["turn_id"] in filepath.name

    def test_attribute_error_timestamp_also_falls_back(self, logger, sample_record):
        """created_at that is not a string triggers AttributeError fallback."""
        sample_record["created_at"] = 12345
        filepath = logger.log(sample_record)
        assert filepath.exists()
        assert "raw" in str(filepath)
