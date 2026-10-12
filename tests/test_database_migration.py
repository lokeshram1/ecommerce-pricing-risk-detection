from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from app.core.config import settings

def test_alembic_upgrade_clean_sqlite(tmp_path,monkeypatch):
    database=tmp_path/"migration.db"
    monkeypatch.setattr(settings,"database_url",f"sqlite:///{database.as_posix()}")
    cfg=Config("alembic.ini")
    command.upgrade(cfg,"head")
    inspector=inspect(create_engine(f"sqlite:///{database.as_posix()}"))
    assert {"products","scores","reviews","alembic_version"} <= set(inspector.get_table_names())
    assert "flagged" in {column["name"] for column in inspector.get_columns("scores")}
