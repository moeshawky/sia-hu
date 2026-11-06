#!/usr/bin/env python3
"""verify_routed_experts.py — proving-launch probe for the routed-experts sidecar.

Sends one short completion to a base URL, then asserts the response's
routed-experts blob (carrier: top-level sglext.routed_experts, per the
proving launch; meta_info kept as fallback) decodes under SGLang's
extract_routed_experts_
from_meta_info contract (base64 -> native int32, row-major
(num_tokens, num_layers, topk)). Prints measured numbers and exits
non-zero on any mismatch.

Layer count and topk width are validated against the DECODED tensor
(length divisibility, derived num_tokens); the expected LxK values are
CLI parameters, not in-code constants.

Exit codes:
  0  all checks passed
  1  usage/token-count mismatch, expert-id range violation, decode shape error
  2  transport / bad JSON / usage fields missing
  3  routed_experts absent from BOTH sglext AND meta_info —
     PROVING-LAUNCH FAILURE SIGNATURE
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

from capture_proxy.routing import decode_routed_experts, extract_routed_experts_b64


def parse_args() -> argparse.Namespace:
    """Parse probe CLI arguments (artifact expectations passed, never hardcoded).

    Args: None (reads sys.argv).

    Returns:
        argparse.Namespace: Parsed arguments.

    Raises:
        SystemExit: argparse usage errors.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url", required=True, help="Base URL, e.g. http://127.0.0.1:31000/v1"
    )
    p.add_argument("--model", required=True, help="Served model name")
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--num-experts", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0)
    return p.parse_args()


def post_completion(args: argparse.Namespace) -> dict[str, Any]:
    """POST one completion request with routed-experts return requested.

    Args:
        args: Parsed CLI arguments.

    Returns:
        dict: Parsed JSON response body.

    Raises:
        SystemExit: code 2 on transport error, non-JSON body, or HTTP error.
    """
    body = {
        "model": args.model,
        "prompt": "The capital of France is",
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        # Per-request opt-in (SGLang GenerateReqInput.return_routed_experts);
        # harmless when the server-side capturer already carries it.
        "return_routed_experts": True,
    }
    req = urllib.request.Request(  # noqa: S310 — operator-supplied base URL only
        f"{args.base_url.rstrip('/')}/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 — documented http(s) base URL
            req, timeout=args.timeout
        ) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:
        print(f"FAIL: HTTP {err.code} from upstream: {err.read()[:200]!r}")
        raise SystemExit(2) from None
    except (urllib.error.URLError, TimeoutError) as err:
        print(f"FAIL: cannot reach {args.base_url}: {err}")
        raise SystemExit(2) from None
    except json.JSONDecodeError as err:
        print(f"FAIL: response is not JSON: {err}")
        raise SystemExit(2) from None


def main() -> None:
    """Run the proving-launch probe and report measured numbers.

    Args: None (reads CLI args).

    Returns: None (raises SystemExit with the documented code).
    """
    args = parse_args()
    body = post_completion(args)

    raw = extract_routed_experts_b64(body)
    if raw is None:
        print(
            "PROVING-LAUNCH FAILURE SIGNATURE: routed_experts is ABSENT "
            "from BOTH sglext AND meta_info — the server carried no routing "
            "blob under either key for this request. Record this in ticket "
            "003 §6; if the server also failed graph capture with this "
            "profile, two same-signature failures make it a redesign "
            "signal, not a retry signal."
        )
        raise SystemExit(3)

    try:
        decoded = decode_routed_experts(raw, num_layers=args.num_layers, topk=args.topk)
    except ValueError as err:
        print(f"FAIL: decode mismatch: {err}")
        raise SystemExit(1) from None

    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        print("FAIL: response usage missing prompt_tokens/completion_tokens")
        raise SystemExit(2)

    expected_tokens = int(prompt_tokens) + int(completion_tokens)
    ok = True

    print(f"token count:      {decoded.num_tokens} (usage says {expected_tokens})")
    print(f"layer count:      {decoded.num_layers}")
    print(f"topk width:       {decoded.topk}")

    if decoded.num_tokens != expected_tokens:
        print("FAIL: num_tokens != prompt_tokens + completion_tokens")
        ok = False

    unit = args.num_layers * args.topk
    lo, hi = min(decoded.expert_ids), max(decoded.expert_ids)
    in_range = lo >= 0 and hi < args.num_experts
    print(
        f"expert id range:  [{lo}, {hi}] valid for 0..{args.num_experts - 1}: {in_range}"
    )
    if not in_range:
        print("FAIL: expert ids outside the architectural range")
        ok = False

    for layer in range(args.num_layers):
        seen = {
            decoded.expert_ids[t * unit + layer * args.topk + k]
            for t in range(decoded.num_tokens)
            for k in range(args.topk)
        }
        print(f"layer {layer:3d}: nonzero/distinct experts touched: {len(seen)}")

    if not ok:
        raise SystemExit(1)
    print("PASS: routed_experts present, decoded, and shape-consistent")


if __name__ == "__main__":
    main()
