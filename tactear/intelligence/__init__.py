"""Intelligence layer for TactEar Phase 2: SOP retrieval and local advisory."""

from tactear.intelligence.advisor import (
    AdvisoryResult,
    build_advisory_prompt,
    check_llm_status,
    generate_advisory,
)
from tactear.intelligence.sop import SOPResult, retrieve_sop

__all__ = [
    "AdvisoryResult",
    "SOPResult",
    "build_advisory_prompt",
    "check_llm_status",
    "generate_advisory",
    "retrieve_sop",
]
