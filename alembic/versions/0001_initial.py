from alembic import op
import sqlalchemy as sa
revision="0001_initial"; down_revision=None; branch_labels=None; depends_on=None
def upgrade():
    op.create_table("products",sa.Column("product_id",sa.String(128),primary_key=True),sa.Column("title",sa.Text(),nullable=False),sa.Column("category",sa.String(256),nullable=False),sa.Column("price",sa.Float()),sa.Column("list_price",sa.Float()),sa.Column("discount",sa.Float()),sa.Column("brand",sa.String(256),nullable=False),sa.Column("description",sa.Text(),nullable=False),sa.Column("features",sa.Text(),nullable=False),sa.Column("rating",sa.Float()),sa.Column("rating_count",sa.Float()),sa.Column("is_synthetic",sa.Boolean(),nullable=False),sa.Column("updated_at",sa.DateTime(timezone=True),nullable=False))
    op.create_index("ix_products_category","products",["category"])
    op.create_table("scores",sa.Column("id",sa.Integer(),primary_key=True),sa.Column("product_id",sa.String(128),sa.ForeignKey("products.product_id"),nullable=False),sa.Column("risk_probability",sa.Float(),nullable=False),sa.Column("risk_score",sa.Integer(),nullable=False),sa.Column("risk_category",sa.String(16),nullable=False),sa.Column("category_deviation",sa.Float()),sa.Column("evidence",sa.Text(),nullable=False),sa.Column("model_version",sa.String(64),nullable=False),sa.Column("created_at",sa.DateTime(timezone=True),nullable=False))
    for name,col in [("ix_scores_product_id","product_id"),("ix_scores_risk_category","risk_category"),("ix_scores_created_at","created_at")]: op.create_index(name,"scores",[col])
    op.create_index("ix_scores_created_category","scores",["created_at","risk_category"])
    op.create_table("reviews",sa.Column("id",sa.Integer(),primary_key=True),sa.Column("score_id",sa.Integer(),sa.ForeignKey("scores.id"),nullable=False,unique=True),sa.Column("decision",sa.String(64),nullable=False),sa.Column("notes",sa.Text(),nullable=False),sa.Column("reviewer",sa.String(128),nullable=False),sa.Column("created_at",sa.DateTime(timezone=True),nullable=False))
    op.create_index("ix_reviews_score_id","reviews",["score_id"])
def downgrade():
    op.drop_table("reviews"); op.drop_table("scores"); op.drop_index("ix_products_category",table_name="products"); op.drop_table("products")
