"""One log line per request, and a correlation identifier.

Section 25 asks for the requests to be traced. What is traced is the method, the
route, the status and the duration.

**The path is logged, the query string is not.** A route is a shape, a query
string is content, and content is what the exposure rules of section 39 keep out
of the logs. A client that puts a document in a query parameter must not have it
written to disk by the middleware that was meant to time it.

**An identifier supplied by the client is reused.** A gateway that already
stamped the request has the correlation the operator will search on; generating
a second one here would break the chain at the door.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

LOGGER = logging.getLogger(__name__)

#: Header carrying the correlation identifier, in and out.
REQUEST_ID_HEADER = "X-Request-ID"


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Time every request, log it once, and correlate it."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Handle one request.

        Args:
            request: The incoming request.
            call_next: The rest of the application.

        Returns:
            The response, with the correlation header set.

        Raises:
            Exception: Whatever the application raised, once logged. The
                exception handlers of section 26 turn it into a payload; this
                middleware only makes sure the request that produced it is not
                missing from the log.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
        started = time.perf_counter()

        context = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        }

        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "request failed",
                extra={**context, "duration_ms": round((time.perf_counter() - started) * 1000, 1)},
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers[REQUEST_ID_HEADER] = request_id
        LOGGER.info(
            "request",
            extra={**context, "status": response.status_code, "duration_ms": duration_ms},
        )
        return response
