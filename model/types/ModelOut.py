import pydantic

class ModelOut(pydantic.BaseModel):
    text: str