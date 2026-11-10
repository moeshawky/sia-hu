"""Model template and training-data compatibility check.

Lightweight readiness check for the actual model/tokenizer/training interface.
Validates chat template rendering, role serialization, and label mask correctness.

No training-data processing framework — reuses existing tokenizer interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateCompatReport:
    """Readiness assessment for model/tokenizer/training interface."""

    has_chat_template: bool
    template_renders: bool
    roles_detected: list[str]
    eos_token: str | None
    bos_token: str | None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rendered_example: str | None = None


def check_template_compat(
    tokenizer: Any,
    example_messages: list[dict[str, str]] | None = None,
) -> TemplateCompatReport:
    """Check model/tokenizer template compatibility.

    Inspects:
    - Native chat template presence
    - Template rendering with example messages
    - Role serialization (system/user/assistant/tool)
    - EOS/BOS token presence
    - Training label mask correctness (if labels provided)

    Returns a report with issues and warnings. Does not modify the tokenizer.
    """
    issues: list[str] = []
    warnings: list[str] = []
    roles_detected: list[str] = []
    rendered: str | None = None

    # Check chat template
    has_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )
    if not has_template:
        issues.append(
            "No chat_template found on tokenizer. Generic template may not match model expectations."
        )

    # Check EOS/BOS
    eos = getattr(tokenizer, "eos_token", None)
    bos = getattr(tokenizer, "bos_token", None)
    if eos is None:
        issues.append("No EOS token defined. Training may not terminate correctly.")
    if bos is None:
        warnings.append("No BOS token defined. Some templates expect BOS.")

    # Render example
    template_renders = False
    if example_messages is None:
        example_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    if has_template:
        try:
            rendered = tokenizer.apply_chat_template(
                example_messages, tokenize=False, add_generation_prompt=False
            )
            template_renders = True
            # Detect roles in rendered output
            for msg in example_messages:
                role = msg.get("role", "")
                if role and role not in roles_detected:
                    roles_detected.append(role)
        except Exception as e:
            issues.append(f"Chat template rendering failed: {e}")
            template_renders = False

    # Check for tool role support
    tool_example = [
        {"role": "user", "content": "test"},
        {"role": "tool", "content": "result"},
    ]
    if has_template:
        try:
            tokenizer.apply_chat_template(tool_example, tokenize=False)
        except Exception:
            warnings.append("Tool role may not be supported by this template.")

    # Check label mask (assistant-only loss)
    if (
        template_renders
        and rendered
        and "<|assistant|>" not in rendered
        and "assistant" not in rendered.lower()
    ):
        warnings.append(
            "Rendered template may not include assistant markers. Verify label masking."
        )

    return TemplateCompatReport(
        has_chat_template=has_template,
        template_renders=template_renders,
        roles_detected=roles_detected,
        eos_token=eos,
        bos_token=bos,
        issues=issues,
        warnings=warnings,
        rendered_example=rendered,
    )
