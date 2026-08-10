"""Structured logs, and what never enters them.

Section 25 asks for JSON logs in production and readable ones locally, which is
what ``SYNTRA_LOG_JSON`` switches between.

**A log line carries the shape of a request, never its content.** The exposure
rules of section 39 forbid writing the submitted document to the logs. What is
written instead is its length, which is what a latency has to be read against.
Nothing in this module has to enforce that: the call sites pass lengths, and
this is the module that makes it visible when one does not.

**Reconfiguring replaces our handler and leaves the others alone.** Building the
application twice in one process is what a test suite does all day. Clearing the
root handlers would take the test harness down with them, so exactly one handler
is owned here and only that one is ever removed.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from backend.app.core.config import Settings

#: Attributes :class:`logging.LogRecord` always carries. Anything else on a
#: record was passed through ``extra`` and belongs in the payload.
STANDARD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

#: Layout used when JSON is switched off. Readable on a terminal, and never in
#: production, where a log line is parsed rather than read.
TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

#: The single handler this module owns.
_HANDLER: logging.Handler | None = None


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record.

        Args:
            record: The record to render.

        Returns:
            A single line of JSON. Values that are not serialisable are
            rendered as strings rather than dropped: losing a field silently is
            worse than logging its repr.
        """
        payload: dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in STANDARD_FIELDS and not key.startswith("_")
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings) -> logging.Handler:
    """Install the application handler on the root logger.

    Args:
        settings: The settings of the running instance.

    Returns:
        The installed handler, so a caller can assert on it.
    """
    global _HANDLER  # noqa: PLW0603 - one handler per process, by design

    root = logging.getLogger()
    if _HANDLER is not None:
        root.removeHandler(_HANDLER)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else logging.Formatter(TEXT_FORMAT))
    root.addHandler(handler)
    root.setLevel(settings.log_level.value)

    _HANDLER = handler
    return handler
