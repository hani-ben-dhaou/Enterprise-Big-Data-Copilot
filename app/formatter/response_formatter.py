"""
Response Formatter — assembles the final CopilotResponse.

Separating formatting from the orchestrator allows:
  - Different output formats in V2 (streaming, markdown, etc.)
  - Clean mapping from pipeline state → API response
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.core.models import (
    CopilotResponse,
    GeneratedSQL,
    PipelineState,
    ValidationResult,
    ValidationStatus,
)

logger = get_logger(__name__)


class ResponseFormatter:
    """Converts a completed PipelineState into a CopilotResponse."""

    def format(self, state: PipelineState) -> CopilotResponse:
        """
        Assemble the final structured response.

        Combines:
          - Generated SQL
          - Explanation
          - Confidence (possibly degraded if warnings present)
          - Warnings collected from validation
        """
        if state.error:
            return self._error_response(state)

        sql_result: GeneratedSQL = state.generated_sql  # type: ignore[assignment]
        validation: ValidationResult = state.validation_result  # type: ignore[assignment]

        # Degrade confidence if warnings exist
        confidence = sql_result.confidence
        if validation.status == ValidationStatus.WARNING:
            confidence = max(0.0, confidence - 0.1 * len(validation.warnings))

        warnings: list[str] = list(validation.warnings)

        # Add regeneration notice if applicable
        if state.regeneration_count > 0:
            warnings.append(
                f"SQL was regenerated {state.regeneration_count} time(s) "
                "due to validation failures."
            )

        # Attach query execution results (best-effort)
        results: list[list[Any]] | None = None
        execution: dict[str, Any] | None = None
        if state.execution_result is not None:
            execution = state.execution_result
            if execution.get("status") == "failed":
                warnings.append(
                    f"Query could not be executed: {execution.get('error', 'unknown error')}"
                )
            else:
                results = execution.get("rows")

        response = CopilotResponse(
            question=state.question,
            sql=sql_result.sql,
            explanation=sql_result.explanation,
            confidence=round(confidence, 3),
            warnings=warnings,
            dialect=sql_result.dialect,
            results=results,
            execution=execution,
        )

        logger.info(
            "response_formatted",
            confidence=response.confidence,
            warnings_count=len(response.warnings),
            execution_status=(execution or {}).get("status"),
        )

        return response

    @staticmethod
    def _error_response(state: PipelineState) -> CopilotResponse:
        return CopilotResponse(
            question=state.question,
            sql="",
            explanation=f"An error occurred: {state.error}",
            confidence=0.0,
            warnings=[state.error or "Unknown error"],
        )