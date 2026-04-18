"""Request ID middleware for distributed tracing.

Attaches a unique request ID to every incoming request, either from
the client-supplied X-Request-ID header or auto-generated. The ID
is included in the response headers and available in log context.
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a unique request ID into each request/response cycle."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
