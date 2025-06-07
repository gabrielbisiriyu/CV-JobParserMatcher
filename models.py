from sqlalchemy import Column, String, JSON, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from db import Base




class ParsedCV(Base):
    __tablename__ = "parsed_cvs"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    text_hash = Column(String, unique=True, nullable=False)
    parsed_fields = Column(JSON, nullable=False)
    cv_emb = Column(Vector(768))
    skill_emb = Column(Vector(768))
    exp_emb = Column(Vector(768))
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))




class ParsedJob(Base):
    __tablename__ = "parsed_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    text_hash = Column(String, unique=True, nullable=False)
    parsed_fields = Column(JSON, nullable=False)
    cv_emb = Column(Vector(768))
    skill_emb = Column(Vector(768))
    exp_emb = Column(Vector(768))
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))