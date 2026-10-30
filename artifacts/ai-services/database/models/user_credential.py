"""User provider credential model.

Per-user LLM API keys for the Settings > AI Provider surface. Keys are
stored encrypted at rest (Fernet, key derived from JWT_SECRET) and are
decrypted only inside the request that needs them. API responses carry
only a masked hint (last 4 chars), never the raw key.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from ..engine import Base


class UserProviderCredential(Base):
    """Encrypted per-user credential for one LLM provider."""

    __tablename__ = "user_provider_credentials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # "groq" | "openrouter" (validated in the service layer).
    provider = Column(String(32), nullable=False)

    # Fernet-encrypted API key (opaque to the DB). Never logged, never
    # returned, never placed in a URL or cache.
    key_encrypted = Column(Text, nullable=False)

    # Last 4 chars of the key for masked display ("....a91f").
    key_hint = Column(String(8), nullable=False, default="")

    # Model id selected by the user (catalog id or advanced custom id).
    model = Column(String(128), nullable=False, default="")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", backref="provider_credentials")

    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_user_provider_credentials_user_provider"),
    )

    def __repr__(self):
        return f"<UserProviderCredential(user_id={self.user_id}, provider={self.provider})>"
