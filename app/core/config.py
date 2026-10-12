from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite:///./pricing_risk.db"
    model_dir: str = "models"
    embedding_mode: str = "lightweight"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    demo_mode: bool = True
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
