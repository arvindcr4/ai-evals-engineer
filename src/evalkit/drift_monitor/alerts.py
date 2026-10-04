"""Alerts and pluggable sinks (stdout, JSONL file, Slack-style webhook)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx

SEVERITY_ICON = {"critical": ":rotating_light:", "warning": ":warning:"}


@dataclass
class Alert:
    date: str
    metric: str
    method: str
    severity: str
    value: float
    baseline: float
    score: float
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class AlertSink(Protocol):
    def send(self, day: str, alerts: list[Alert]) -> None: ...


class StdoutSink:
    def __init__(self, stream=None):
        self.stream = stream

    def send(self, day: str, alerts: list[Alert]) -> None:
        for a in alerts:
            print(f"[{a.severity.upper():8}] {day} {a.method:6} {a.message}", file=self.stream)


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, day: str, alerts: list[Alert]) -> None:
        with open(self.path, "a") as f:
            f.writelines(json.dumps(a.to_dict()) + "\n" for a in alerts)


def slack_payload(day: str, alerts: list[Alert]) -> dict:
    """Slack incoming-webhook body: a header plus one section per alert."""
    crit = sum(a.severity == "critical" for a in alerts)
    title = f"Drift monitor {day}: {crit} critical, {len(alerts) - crit} warning"
    blocks: list[dict] = [{"type": "header", "text": {"type": "plain_text", "text": title}}]
    for a in alerts:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{SEVERITY_ICON.get(a.severity, '')} *{a.metric}* ({a.method}) "
                    f"{a.message}",
                },
            }
        )
    return {"text": title, "blocks": blocks}


class WebhookSink:
    """POST a Slack-style payload; without a URL it only builds (and optionally echoes) it."""

    def __init__(self, url: str | None = None, timeout: float = 10.0, echo: bool = False):
        self.url = url
        self.timeout = timeout
        self.echo = echo
        self.sent: list[dict] = []

    def send(self, day: str, alerts: list[Alert]) -> None:
        if not alerts:
            return
        payload = slack_payload(day, alerts)
        self.sent.append(payload)
        if self.url:
            httpx.post(self.url, json=payload, timeout=self.timeout).raise_for_status()
        elif self.echo:
            print("webhook payload (dry run, no URL):\n" + json.dumps(payload, indent=2))


def parse_sink(spec: str) -> AlertSink:
    """``stdout`` | ``jsonl:<path>`` | ``webhook`` | ``webhook:<url>``."""
    kind, _, arg = spec.partition(":")
    if kind == "stdout":
        return StdoutSink()
    if kind == "jsonl":
        if not arg:
            raise ValueError("jsonl sink needs a path: jsonl:<path>")
        return JsonlSink(arg)
    if kind == "webhook":
        return WebhookSink(arg or None, echo=not arg)
    raise ValueError(f"unknown sink spec: {spec!r}")
