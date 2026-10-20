"""Olive Insights: LLM-driven behavioral signal extraction over finalized transcripts.

Phase 1a covers the extraction core only: prompt/config loading, transcript chunking,
LLM-output validation, speaker identity resolution, and the ``InsightsService`` that
persists per-meeting signals and moments. Queueing, aggregation, and the API are later phases.
"""

from __future__ import annotations

from .extract import InsightsService

__all__ = ["InsightsService"]
