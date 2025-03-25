from uuid import uuid4
from pydantic import BaseModel, Field
import random

# Constants
MAX_COORDINATE = 100

LocationIndex = int


class DropOff(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()))
    location_id: LocationIndex

    def __hash__(self) -> int:
        return hash(self.uid)


class Location(BaseModel):
    x: float
    y: float


class Instance(BaseModel):
    drop_offs: list[DropOff]
    vehicle_capacity: int
    locations: list[Location]
    depot_location: LocationIndex

    def distance(self, location_1: LocationIndex, location_2: LocationIndex) -> float:
        return (
            (self.locations[location_1].x - self.locations[location_2].x) ** 2
            + (self.locations[location_1].y - self.locations[location_2].y) ** 2
        ) ** 0.5

    def distance_to_depot(self, drop_off: DropOff) -> float:
        return self.distance(drop_off.location_id, self.depot_location)

    def distance_between_drop_offs(
        self, drop_off1: DropOff, drop_off2: DropOff
    ) -> float:
        return self.distance(drop_off1.location_id, drop_off2.location_id)


def generate_random(n_drop_offs: int, vehicle_capacity: int) -> Instance:
    # DropOff instances are created with location_id starting from 1,
    # while the depot_location is set to 0.
    drop_offs = [DropOff(location_id=i) for i in range(1, n_drop_offs + 1)]
    locations = [
        Location(x=MAX_COORDINATE * random.random(), y=MAX_COORDINATE * random.random())
        for _ in range(n_drop_offs + 1)
    ]
    depot_location = 0
    return Instance(
        drop_offs=drop_offs,
        vehicle_capacity=vehicle_capacity,
        locations=locations,
        depot_location=depot_location,
    )
