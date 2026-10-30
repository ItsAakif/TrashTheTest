"""Stage handlers package (T209).

Importing this package registers every additive job-type handler with
services.jobs.worker (ADR-008 / T009), so API, worker, and test importers
all see one registry.
"""
from .document_processing import process_job, STAGES, STAGE_PROGRESS_RANGES

from .structure_reprocess import process_job as structure_reprocess_handler
from services.jobs.worker import register_handler
from services.jobs.types import JobType

register_handler(JobType.STRUCTURE_REPROCESS.value,
                 structure_reprocess_handler)

__all__ = ["process_job", "STAGES", "STAGE_PROGRESS_RANGES",
           "structure_reprocess_handler"]
