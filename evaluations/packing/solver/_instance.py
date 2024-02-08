from pydantic import BaseModel  # pip install pydantic
from typing import Optional

class Rectangle(BaseModel):
    width: int
    height: int
    value: int = 1 # some instances may have a value

class Container(BaseModel):
    width: int
    height: int

class Instance(BaseModel):
    container: Container
    rectangles: list[Rectangle]

class Placement(BaseModel):
    x: int
    y: int
    rotated: bool = False  # not all variants allow rotation

class Solution(BaseModel):
    placements: list[Optional[Placement]]