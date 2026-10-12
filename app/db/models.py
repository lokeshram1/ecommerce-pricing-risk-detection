from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, Boolean, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base

class Product(Base):
    __tablename__="products"
    product_id: Mapped[str]=mapped_column(String(128),primary_key=True)
    title: Mapped[str]=mapped_column(Text,default=""); category: Mapped[str]=mapped_column(String(256),default="Unknown",index=True)
    price: Mapped[float|None]=mapped_column(Float); list_price: Mapped[float|None]=mapped_column(Float); discount: Mapped[float|None]=mapped_column(Float)
    brand: Mapped[str]=mapped_column(String(256),default=""); description: Mapped[str]=mapped_column(Text,default=""); features: Mapped[str]=mapped_column(Text,default="")
    rating: Mapped[float|None]=mapped_column(Float); rating_count: Mapped[float|None]=mapped_column(Float); is_synthetic: Mapped[bool]=mapped_column(Boolean,default=False)
    updated_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
    scores: Mapped[list["Score"]]=relationship(back_populates="product")

class Score(Base):
    __tablename__="scores"
    id: Mapped[int]=mapped_column(Integer,primary_key=True); product_id: Mapped[str]=mapped_column(ForeignKey("products.product_id"),index=True)
    risk_probability: Mapped[float]=mapped_column(Float); risk_score: Mapped[int]=mapped_column(Integer); risk_category: Mapped[str]=mapped_column(String(16),index=True)
    flagged: Mapped[bool]=mapped_column(Boolean,default=False,index=True)
    category_deviation: Mapped[float|None]=mapped_column(Float); evidence: Mapped[str]=mapped_column(Text); model_version: Mapped[str]=mapped_column(String(64),default="0.1.0")
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc)); product: Mapped[Product]=relationship(back_populates="scores")
    review: Mapped["Review|None"]=relationship(back_populates="score",uselist=False)

class Review(Base):
    __tablename__="reviews"
    id: Mapped[int]=mapped_column(Integer,primary_key=True); score_id: Mapped[int]=mapped_column(ForeignKey("scores.id"),index=True,unique=True)
    decision: Mapped[str]=mapped_column(String(64)); notes: Mapped[str]=mapped_column(Text,default=""); reviewer: Mapped[str]=mapped_column(String(128),default="reviewer")
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc)); score: Mapped[Score]=relationship(back_populates="review")

Index("ix_scores_created_category","scores.created_at","scores.risk_category")
