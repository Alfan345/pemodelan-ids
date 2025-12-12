"""
Configuration settings for IDS middleware.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # API Settings
    api_prefix: str = Field(default="/api/v1", description="API prefix for all routes")
    app_name: str = Field(default="IDS Middleware API", description="Application name")
    version: str = Field(default="1.0.0", description="API version")
    
    # Model Settings
    artifacts_path: Path = Field(
        default=Path(__file__).parent.parent / "artifacts",
        description="Path to model artifacts directory"
    )
    model_state_file: str = Field(default="model_state.pt", description="Model state filename")
    config_file: str = Field(default="config.json", description="Config filename")
    label_map_file: str = Field(default="label_map.json", description="Label map filename")
    report_file: str = Field(default="report.json", description="Report filename")
    scaler_file: str = Field(default="scaler.pkl", description="Scaler filename")
    transform_meta_file: str = Field(default="transform_meta.json", description="Transform meta filename")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload")


# Global settings instance
settings = Settings()
