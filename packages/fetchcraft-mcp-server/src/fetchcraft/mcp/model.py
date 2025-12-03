from pydantic import BaseModel

class QueryResponse(BaseModel):
    answer: str
    citations: dict
    model: str
