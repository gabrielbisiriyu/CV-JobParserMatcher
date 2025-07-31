from sqlalchemy import Column, String, JSON, TIMESTAMP, text, UniqueConstraint, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from db import Base



class ParsedCV(Base):
    __tablename__ = "parsed_cvs"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(String, nullable=False)  # ← Add this
    text_hash = Column(String, unique=True, nullable=False)
    parsed_fields = Column(JSON, nullable=False)
    cv_emb = Column(Vector(768))
    skill_emb = Column(Vector(768))
    exp_emb = Column(Vector(768))
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("user_id", name="uq_user_cv"),  # ← Only one CV per user
    )


class ParsedJob(Base):
    __tablename__ = "parsed_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    company_id = Column(String, nullable=False)  # NEW: Company identifier
    text_hash = Column(String, unique=True, nullable=False)
    job_text = Column(Text, nullable=False)
    parsed_fields = Column(JSON, nullable=False)
    cv_emb = Column(Vector(768))
    skill_emb = Column(Vector(768))
    exp_emb = Column(Vector(768))
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))



class JobApplication(Base):
    __tablename__ = "job_applications"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    job_id = Column(UUID(as_uuid=True), ForeignKey("parsed_jobs.id"), nullable=False)
    cv_id = Column(UUID(as_uuid=True), ForeignKey("parsed_cvs.id"), nullable=False)
    applied_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("job_id", "cv_id", name="uq_job_cv_application"),
    )  