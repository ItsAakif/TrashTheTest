"""Enum persistence contract: ORM must bind lowercase VALUES, not NAMES.

Live regression: every SQLEnum column used native_enum=False without
values_callable, so SQLAlchemy bound enum NAMES ("PROCESSING") while the
migrations create native PG enums with lowercase VALUES ("processing").
POST /api/projects failed live with::

    invalid input value for enum projectstatus: "PROCESSING"

No database is touched here: the test inspects the mapped column types
and their bind processors, which is exactly where the drift lived (the
existing suite only uses DummySession fakes, so nothing else catches it).
"""

import pytest

from database.models.chunk import Chunk  # noqa: F401  (import registers mappers)
from database.models.document import Document, DocumentStatus, FileType
from database.models.flashcard import Difficulty, Flashcard
from database.models.generated_content import GeneratedContentCache, OutputType
from database.models.job import Job, JobStatus, JobType
from database.models.project import Project, ProjectStatus
from database.models.study_session import StudySession, SessionType
from database.models.website import (
    Website,
    WebsiteStatus,
    WebsiteType,
    WebsiteVisibility,
)


@pytest.mark.parametrize(
    "model,attr,enum_cls",
    [
        (Project, "status", ProjectStatus),
        (Document, "file_type", FileType),
        (Document, "processing_status", DocumentStatus),
        (Job, "job_type", JobType),
        (Job, "status", JobStatus),
        (Flashcard, "difficulty", Difficulty),
        (GeneratedContentCache, "output_type", OutputType),
        (StudySession, "session_type", SessionType),
        (Website, "website_type", WebsiteType),
        (Website, "visibility", WebsiteVisibility),
        (Website, "status", WebsiteStatus),
    ],
)
def test_enum_column_persists_values(model, attr, enum_cls):
    column = model.__table__.c[attr]
    persisted = list(column.type.enums)
    assert persisted == [m.value for m in enum_cls], (
        f"{model.__name__}.{attr} persists {persisted}, "
        f"expected values {[m.value for m in enum_cls]}"
    )
    # No NAME may leak into the persisted set (PG native enums reject it).
    names = {m.name for m in enum_cls} - {m.value for m in enum_cls}
    assert not (set(persisted) & names), (
        f"{model.__name__}.{attr} would bind a NAME: {set(persisted) & names}"
    )


@pytest.mark.parametrize(
    "model,attr,enum_cls,member",
    [
        (Project, "status", ProjectStatus, ProjectStatus.PROCESSING),
        (Document, "processing_status", DocumentStatus, DocumentStatus.PENDING),
        (Job, "status", JobStatus, JobStatus.PENDING),
        (Job, "job_type", JobType, JobType.DOCUMENT_PROCESSING),
    ],
)
def test_enum_bind_processor_emits_value(model, attr, enum_cls, member):
    column = model.__table__.c[attr]
    processor = column.type.bind_processor(column.type)
    assert processor is not None
    assert processor(member) == member.value
