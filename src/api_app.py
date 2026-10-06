"""
Convenience entry point so you can run:

    uvicorn src.api_app:app --reload
"""
from src.api.main import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)