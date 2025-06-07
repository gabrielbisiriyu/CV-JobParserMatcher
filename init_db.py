# init_db.py
# init_db.py
import asyncio
from db import engine, Base

# ✅ Explicitly import models to register them with SQLAlchemy
from models import ParsedCV, ParsedJob

async def init_models():
    async with engine.begin() as conn:
        print("Connected to DB ✅")
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(init_models())
