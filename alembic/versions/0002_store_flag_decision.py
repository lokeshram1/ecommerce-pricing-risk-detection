from alembic import op
import sqlalchemy as sa

revision="0002_store_flag_decision"
down_revision="0001_initial"
branch_labels=None
depends_on=None

def upgrade():
    op.add_column("scores",sa.Column("flagged",sa.Boolean(),nullable=False,server_default=sa.false()))
    op.create_index("ix_scores_flagged","scores",["flagged"])

def downgrade():
    op.drop_index("ix_scores_flagged",table_name="scores")
    op.drop_column("scores","flagged")
