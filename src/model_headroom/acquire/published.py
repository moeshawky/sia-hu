"""Published-evidence acquisition: metadata-first, revision-pinned, honest.

Acquires training-history evidence (loss trajectories) from a Hugging Face
Hub repository WITHOUT downloading the weights. The flow is
metadata-first: list the repo tree via the Hub REST API, identify candidate
training-history artifacts by name, and download ONLY the chosen small
artifact.

The mapping onto ``LossDataPoint`` is honest by construction:

- A training loss is mapped into the ``validation_loss`` field ONLY with an
  explicit limitation saying it is a TRAINING-split value; when a
  validation/eval loss is published, it is preferred over the training loss.
- A composite score is not a loss and is never mapped.
- A metric whose semantics the artifact does not declare (e.g. an
  undeclared ``pres_mse``) is never mapped — mapping it would be fabrication.
- Binary TensorBoard event files are refused (this extractor parses JSON,
  not protobuf).
- Network failure degrades to a ``"unavailable"`` bundle with the reason —
  never a fabricated number.

Only ``urllib`` from the standard library is used: no new dependency.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from model_headroom.acquire.provenance import (
    EvidenceBundle,
    EvidenceProvenance,
    ProvenanceClass,
)
from model_headroom.models.types import LossDataPoint

if TYPE_CHECKING:
    from collections.abc import Sequence

_HF_BASE = "https://huggingface.co"
_HF_API = "https://huggingface.co/api"

_CANDIDATE_BASENAMES = (
    "run_record.json",
    "trainer_state.json",
    "training_history.json",
)
_TFEVENTS_PREFIX = "events.out.tfevents."

_SIGNAL_NAME = "loss_saturation"


@dataclass(frozen=True)
class HubFile:
    """One file entry from a Hub repo tree listing.

    Attributes:
        path: In-repo path of the file.
        size: File size in bytes, when the API reports it.
        lfs: True when the file is stored in Git LFS.
    """

    path: str
    size: int
    lfs: bool


class PublishedAcquisitionError(Exception):
    """Raised when a Hub metadata or download step fails.

    The top-level flow catches this and degrades to an ``"unavailable"``
    evidence bundle; it is never converted into a fabricated value.
    """


def _quote_repo(repo_id: str) -> str:
    return urllib.parse.quote(repo_id, safe="/")


def _quote_path(file_path: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in file_path.split("/"))


def _resolve_url(repo_id: str, kind: str, file_path: str, revision: str) -> str:
    return (
        f"{_HF_BASE}/{kind}/{_quote_repo(repo_id)}/"
        f"resolve/{urllib.parse.quote(revision, safe='')}/{_quote_path(file_path)}"
    )


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_https(url: str) -> str:
    """Reject a non-HTTPS URL before it is opened.

    A published-evidence fetch must never follow a file or custom scheme. This
    is a scheme allow-list, not a lint suppression: a URL that is not HTTPS is
    a failure, not something to be opened anyway.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise PublishedAcquisitionError(
            f"refusing to open a non-HTTPS URL: {parsed.scheme!r}"
        )
    return url


def list_repo_tree(
    repo_id: str,
    *,
    kind: Literal["models", "datasets"] = "models",
    revision: str = "main",
    timeout: float = 30.0,
) -> list[HubFile]:
    """List every file in a Hub repo tree via the REST API (metadata only).

    Input:
        repo_id: The repo id, e.g. ``"moeshawky/RockAndEgg-313M-v2"``.
        kind: ``"models"`` or ``"datasets"``.
        revision: The branch/tag/commit to list (default ``"main"``).
        timeout: Per-request timeout in seconds.
    Output:
        One ``HubFile`` per file in the tree (directories excluded).
    Raises:
        PublishedAcquisitionError: On any network, HTTP, or decode failure.
    Invariants:
        - Metadata only: never downloads file content.
        - Revision-pinned: the listing is for exactly ``revision``.
    """
    url = (
        f"{_HF_API}/{kind}/{_quote_repo(repo_id)}/tree/"
        f"{urllib.parse.quote(revision, safe='')}?recursive=true"
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - scheme guarded
            _ensure_https(url), timeout=timeout
        ) as response:
            raw = response.read()
    except (OSError, ValueError) as e:
        raise PublishedAcquisitionError(
            f"cannot list {kind} repo {repo_id!r} at {revision!r}: {e}"
        ) from e
    try:
        entries: object = json.loads(raw)
    except ValueError as e:
        raise PublishedAcquisitionError(
            f"cannot decode the tree listing for {repo_id!r}: {e}"
        ) from e
    if not isinstance(entries, list):
        raise PublishedAcquisitionError(
            f"unexpected tree listing shape for {repo_id!r}: expected a JSON array"
        )
    files: list[HubFile] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "file":
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            continue
        size = entry.get("size")
        files.append(
            HubFile(
                path=path,
                size=int(size) if isinstance(size, (int, float)) else 0,
                lfs=bool(entry.get("lfs", False)),
            )
        )
    return files


def _is_tfevents(path: str) -> bool:
    basename = path.rsplit("/", 1)[-1]
    return basename.startswith(_TFEVENTS_PREFIX)


def identify_training_history_artifacts(files: Sequence[HubFile]) -> list[HubFile]:
    """Identify candidate training-history artifacts by name, in priority order.

    Candidates (highest priority first): ``run_record.json``,
    ``trainer_state.json``, ``training_history.json``, any ``*loss*.json``,
    and finally ``events.out.tfevents.*`` (binary — listed last so a JSON
    artifact always wins; a tfevents-only repo is refused by the caller).

    Input:
        files: The full tree listing from :func:`list_repo_tree`.
    Output:
        Candidates ordered by (priority, path); empty when nothing matches.
    """
    ranked: list[tuple[int, str, HubFile]] = []
    for f in files:
        basename = f.path.rsplit("/", 1)[-1]
        if basename in _CANDIDATE_BASENAMES:
            ranked.append((_CANDIDATE_BASENAMES.index(basename), f.path, f))
        elif basename.endswith(".json") and "loss" in basename.lower():
            ranked.append((3, f.path, f))
        elif _is_tfevents(f.path):
            ranked.append((4, f.path, f))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [f for _, _, f in ranked]


def download_artifact(
    repo_id: str,
    *,
    kind: Literal["models", "datasets"] = "models",
    file_path: str,
    revision: str = "main",
    timeout: float = 30.0,
) -> bytes:
    """Download one artifact from a Hub repo at a pinned revision.

    Downloads ONLY the named file — never the weights, never the repo.

    Input:
        repo_id: The repo id.
        kind: ``"models"`` or ``"datasets"``.
        file_path: In-repo path of the single artifact to fetch.
        revision: The revision to pin to (default ``"main"``).
        timeout: Per-request timeout in seconds.
    Output:
        The raw artifact bytes.
    Raises:
        PublishedAcquisitionError: On any network, HTTP, or decode failure.
    """
    url = _resolve_url(repo_id, kind, file_path, revision)
    try:
        with urllib.request.urlopen(  # noqa: S310 - scheme guarded
            _ensure_https(url), timeout=timeout
        ) as response:
            data: bytes = response.read()
            return data
    except (OSError, ValueError) as e:
        raise PublishedAcquisitionError(
            f"cannot download {file_path!r} from {kind} repo {repo_id!r}: {e}"
        ) from e


def _is_loss_key(key: str) -> bool:
    return "loss" in key.lower()


def _is_validation_key(key: str) -> bool:
    k = key.lower()
    return k.startswith("eval") or "val" in k


def _extract_points(
    series: object,
    key: str,
    *,
    root_is_loss: bool = False,
    root_is_validation: bool = False,
) -> tuple[list[tuple[int, float, bool]], str, bool] | None:
    """Extract (step, value, is_validation) points from one candidate series.

    Returns ``None`` when the candidate is not a loss-like series. A bare
    numeric list is a loss series ONLY under a loss-like key (or as a
    top-level list of an artifact whose file name is loss-like) — a bare
    numeric list under ``"composite"`` or ``"accuracy"`` is a score, not a
    loss. The returned tuple is (points, loss_key, step_was_logged).
    """
    if not isinstance(series, list) or not series:
        return None

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in series):
        if key == "root" and not root_is_loss:
            return None
        if key != "root" and not _is_loss_key(key):
            return None
        is_validation = root_is_validation if key == "root" else _is_validation_key(key)
        points = [
            (index, float(value), is_validation) for index, value in enumerate(series)
        ]
        return points, key, False

    if all(isinstance(v, dict) for v in series):
        loss_keys = {
            k
            for entry in series
            for k in entry
            if isinstance(k, str) and _is_loss_key(k)
        }
        if not loss_keys:
            return None
        loss_key = sorted(loss_keys)[0]
        dict_points: list[tuple[int, float, bool]] = []
        step_logged = False
        for index, entry in enumerate(series):
            raw_value = entry.get(loss_key)
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                continue
            raw_step = entry.get("step", entry.get("global_step", index))
            if not isinstance(raw_step, (int, float)) or isinstance(raw_step, bool):
                raw_step = index
            else:
                step_logged = True
            dict_points.append(
                (int(raw_step), float(raw_value), _is_validation_key(loss_key))
            )
        if not dict_points:
            return None
        return dict_points, loss_key, step_logged

    return None


def _top_level_rejections(artifact: dict[str, Any]) -> list[str]:
    """Describe top-level list-of-entries keys that carry no loss-like field."""
    rejections: list[str] = []
    for key, value in artifact.items():
        if not isinstance(value, list) or not value:
            continue
        if not all(isinstance(v, dict) for v in value):
            continue
        fields = sorted({k for v in value for k in v if isinstance(k, str)})
        if any(_is_loss_key(f) for f in fields):
            continue
        rejections.append(
            f"key '{key}' holds {len(value)} entries with fields "
            f"{fields}; no loss-like field, so it was not mapped "
            "(a composite score or undeclared metric is not a loss)"
        )
    return rejections


def extract_loss_trajectory(
    artifact: object,
    *,
    repo_id: str,
    file_path: str,
    kind: Literal["models", "datasets"] = "models",
    revision: str = "main",
    lineage_authenticated: bool = False,
    tokens_per_step: int | None = None,
) -> EvidenceBundle:
    """Map a discovered artifact onto LossDataPoint pairs, honestly.

    Input:
        artifact: The parsed artifact (the JSON-decoded dict or list).
        repo_id: The repo the artifact came from.
        file_path: The in-repo path of the artifact.
        kind: ``"models"`` or ``"datasets"``.
        revision: The revision the artifact was pinned to.
        lineage_authenticated: Operator attestation that the artifact
            demonstrably belongs to the checkpoint's training lineage.
            ``True`` -> ``"authenticated_ancestor"``; ``False`` (default)
            -> ``"published_context"``. The extractor never infers lineage
            on its own — that claim requires evidence the operator holds.
        tokens_per_step: When the artifact logged steps but not tokens, the
            token count is derived as ``step * tokens_per_step``. When
            omitted, ``tokens_trained`` carries the step ordinal and the
            limitation says so.
    Output:
        An ``EvidenceBundle`` for the ``loss_saturation`` signal. The payload
        is a ``list[LossDataPoint]`` (possibly empty) — empty with
        ``provenance_class="unavailable"`` when the artifact contains no
        loss-like series.
    Invariants:
        - A training loss is stored in ``validation_loss`` only with an
          explicit limitation naming it as a training-split value.
        - A validation/eval loss, when published, is preferred.
        - Non-loss fields (composite scores, undeclared metrics) are never
          mapped; they are named in the limitations.
    """
    url = _resolve_url(repo_id, kind, file_path, revision)
    limitations: list[str] = []
    assumptions: list[str] = []

    if isinstance(artifact, dict):
        candidates: list[tuple[str, object]] = list(artifact.items())
        rejections = _top_level_rejections(artifact)
    elif isinstance(artifact, list):
        candidates = [("root", artifact)]
        rejections = []
    else:
        candidates = []
        rejections = ["artifact is not a JSON object or array; nothing to map"]

    series: list[tuple[str, list[tuple[int, float, bool]], str, bool]] = []
    step_logged = False
    basename = file_path.rsplit("/", 1)[-1]
    root_is_loss = _is_loss_key(basename)
    root_is_validation = _is_validation_key(basename)
    for key, value in candidates:
        extracted = _extract_points(
            value, key, root_is_loss=root_is_loss, root_is_validation=root_is_validation
        )
        if extracted is None:
            continue
        points, loss_key, logged = extracted
        series.append((key, points, loss_key, logged))
        step_logged = step_logged or logged

    if not series:
        reasons = rejections or [
            "no loss-like series found in the artifact "
            "(looked for a loss/eval_loss/validation_loss field or a bare "
            "numeric series under a loss-like key)"
        ]
        return EvidenceBundle(
            signal_name=_SIGNAL_NAME,
            provenance=EvidenceProvenance(
                provenance_class="unavailable",
                source_url=url,
                source_revision=revision,
                acquisition_method="hub_metadata_tree",
                retrieved_at=_now_utc(),
                artifact_path=file_path,
                applicability="no loss evidence acquired from this artifact; "
                "nothing was fabricated",
                limitations=tuple(reasons),
            ),
            payload=None,
        )

    validation_series = [s for s in series if any(v for _, _, v in s[1])]
    if validation_series:
        chosen_key, all_points, loss_key, _ = validation_series[0]
    else:
        chosen_key, all_points, loss_key, _ = series[0]
    for key, _, other_loss_key, _ in series:
        if key == chosen_key:
            continue
        split = "validation/eval" if _is_validation_key(other_loss_key) else "training"
        limitations.append(
            f"the {other_loss_key!r} series (under key {key!r}) was skipped: "
            f"a single {('validation/eval' if validation_series else 'training')}"
            "-split trajectory is kept, and mixing splits in one trajectory "
            "would be a false curve"
        )

    finite_points = [p for p in all_points if math.isfinite(p[1])]
    skipped_nonfinite = len(all_points) - len(finite_points)
    if skipped_nonfinite:
        limitations.append(
            f"{skipped_nonfinite} non-finite loss value(s) in {loss_key!r} "
            "were skipped: a NaN/Inf is not a usable measurement"
        )
    all_points = finite_points
    if not all_points:
        return EvidenceBundle(
            signal_name=_SIGNAL_NAME,
            provenance=EvidenceProvenance(
                provenance_class="unavailable",
                source_url=url,
                source_revision=revision,
                acquisition_method="hub_metadata_tree",
                retrieved_at=_now_utc(),
                artifact_path=file_path,
                applicability="no usable loss evidence in this artifact; "
                "nothing was fabricated",
                limitations=(
                    *limitations,
                    "every loss value in the artifact was non-finite",
                ),
            ),
            payload=None,
        )

    has_validation = any(is_val for _, _, is_val in all_points)
    payload: list[LossDataPoint] = []
    for step, value, _ in all_points:
        if tokens_per_step is not None:
            tokens = int(step) * int(tokens_per_step)
            assumptions.append(
                f"tokens_trained derived as step * {tokens_per_step} because "
                "tokens were not logged"
            )
        else:
            tokens = int(step)
            assumptions.append(
                "tokens_trained set to the step ordinal because tokens were "
                "not logged and no tokens_per_step was supplied"
            )
        payload.append(LossDataPoint(tokens_trained=tokens, validation_loss=value))

    split = "validation/eval" if _is_validation_key(loss_key) else "training"
    limitations.append(
        f"mapped from the {loss_key!r} field (key {chosen_key!r}): "
        f"{len(all_points)}-point {split}-split loss log"
    )
    if split == "training":
        limitations.append(
            f"the {loss_key!r} values are TRAINING losses stored in the "
            "LossDataPoint validation_loss field; a training loss is not "
            "a validation loss"
        )
    if not has_validation:
        limitations.append("no validation-split loss was published")
    if not step_logged:
        limitations.append("step values were not logged; step taken as list index")
    limitations.extend(rejections)

    if lineage_authenticated:
        provenance_class: ProvenanceClass = "authenticated_ancestor"
        applicability = (
            "LossDataPoint series for the loss_saturation (R-01) signal; "
            "the operator attested the artifact belongs to the checkpoint's "
            "training lineage"
        )
    else:
        provenance_class = "published_context"
        applicability = (
            "LossDataPoint series for the loss_saturation (R-01) signal; "
            "establishes the shape and values of the published loss log "
            "only — NOT that it belongs to the checkpoint's training lineage"
        )

    return EvidenceBundle(
        signal_name=_SIGNAL_NAME,
        provenance=EvidenceProvenance(
            provenance_class=provenance_class,
            source_url=url,
            source_revision=revision,
            acquisition_method="hub_metadata_tree",
            artifact_path=file_path,
            retrieved_at=_now_utc(),
            applicability=applicability,
            limitations=tuple(limitations),
            assumptions=tuple(dict.fromkeys(assumptions)),
        ),
        payload=payload,
    )


def acquire_published_loss_evidence(
    repo_id: str,
    *,
    kind: Literal["models", "datasets"] = "models",
    revision: str = "main",
    file_path: str | None = None,
    lineage_authenticated: bool = False,
    tokens_per_step: int | None = None,
    timeout: float = 30.0,
) -> EvidenceBundle:
    """Acquire loss-trajectory evidence from a Hub repo, degrading gracefully.

    Lists the repo tree, identifies candidate training-history artifacts,
    downloads ONLY the chosen artifact, and maps it onto LossDataPoint pairs.
    Every failure mode — network, HTTP, decode, no candidate, binary file —
    degrades to an ``"unavailable"`` bundle with the reason; nothing is ever
    fabricated.

    Input:
        repo_id: The repo id, e.g. ``"moeshawky/rae-005-research-evidence"``.
        kind: ``"models"`` or ``"datasets"``.
        revision: The revision to pin to (default ``"main"``).
        file_path: When given, download exactly this file instead of
            auto-selecting a candidate.
        lineage_authenticated: Operator attestation of training lineage.
        tokens_per_step: Token derivation factor for step-logged artifacts.
        timeout: Per-request timeout in seconds.
    Output:
        An ``EvidenceBundle`` for the ``loss_saturation`` signal.
    """
    try:
        files = list_repo_tree(repo_id, kind=kind, revision=revision, timeout=timeout)
    except PublishedAcquisitionError as e:
        return _unavailable_bundle(repo_id, kind, revision, str(e))

    if file_path is None:
        candidates = identify_training_history_artifacts(files)
        if not candidates:
            return _unavailable_bundle(
                repo_id,
                kind,
                revision,
                f"no training-history artifact found among {len(files)} files; "
                "looked for run_record.json, trainer_state.json, "
                "training_history.json, *loss*.json, events.out.tfevents.*",
            )
        chosen = candidates[0]
    else:
        chosen = HubFile(path=file_path, size=0, lfs=False)

    if _is_tfevents(chosen.path):
        return _unavailable_bundle(
            repo_id,
            kind,
            revision,
            f"the only candidate is {chosen.path!r}, a binary TensorBoard "
            "event file; the JSON extractor refuses to parse protobuf, so no "
            "loss series was acquired (nothing fabricated)",
        )

    try:
        data = download_artifact(
            repo_id,
            kind=kind,
            file_path=chosen.path,
            revision=revision,
            timeout=timeout,
        )
    except PublishedAcquisitionError as e:
        return _unavailable_bundle(repo_id, kind, revision, str(e))

    try:
        parsed: object = json.loads(data)
    except ValueError as e:
        return _unavailable_bundle(
            repo_id, kind, revision, f"artifact {chosen.path!r} is not valid JSON: {e}"
        )

    return extract_loss_trajectory(
        parsed,
        repo_id=repo_id,
        file_path=chosen.path,
        kind=kind,
        revision=revision,
        lineage_authenticated=lineage_authenticated,
        tokens_per_step=tokens_per_step,
    )


def _unavailable_bundle(
    repo_id: str, kind: str, revision: str, reason: str
) -> EvidenceBundle:
    """Build the honest 'unavailable' bundle for a failed acquisition."""
    return EvidenceBundle(
        signal_name=_SIGNAL_NAME,
        provenance=EvidenceProvenance(
            provenance_class="unavailable",
            source_url=f"{_HF_BASE}/{kind}/{_quote_repo(repo_id)}",
            source_revision=revision,
            acquisition_method="hub_metadata_tree",
            retrieved_at=_now_utc(),
            applicability="no evidence acquired; nothing was fabricated",
            limitations=(reason,),
        ),
        payload=None,
    )
