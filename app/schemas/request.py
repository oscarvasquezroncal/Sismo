from pydantic import BaseModel
from typing import List

class SeismicSequence(BaseModel):
    sequence: List[List[float]] 
