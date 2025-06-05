from collections.abc import Iterable
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)


class FacilityLocationInstance(BaseModel):
    """Pydantic model representing a facility location problem instance."""

    # Metadata
    instance_uid: str = Field(..., description="The unique identifier of the instance.")
    origin: str = Field(default="", description="The origin or source of the instance.")
    comment: str = Field(default="", description="Any comments to the instance.")

    # Instance statistics
    num_clients: PositiveInt = Field(
        ..., description="The number of clients to allocate."
    )
    num_facilities: PositiveInt = Field(
        ..., description="The number of potential facility locations."
    )
    is_integral: bool = Field(
        default=False,
        description="Specifies if the facility opening and connection costs are integral.",
    )

    # Instance data
    opening_cost: list[NonNegativeFloat | NonNegativeInt] = Field(
        ...,
        description="Opening cost of each facility.",
    )
    allocation_cost: list[list[NonNegativeFloat | NonNegativeInt]] = Field(
        ...,
        description=(
            "Cost to to allocating from each client (outer) to each facility (inner). "
            "`allocation_cost[i][k]` is the cost from serving client *i* via facility *k*."
        ),
    )

    schema_version: int = Field(
        default=0,
        description="Schema version of the instance. If the schema changes, this will be incremented.",
    )

    # The "after" mode is chosen to ensure validation occurs after all fields are populated.
    # This allows cross-field validation, such as checking relationships between multiple fields.
    @model_validator(mode="after")
    def validate_instance(self) -> "FacilityLocationInstance":
        if len(self.allocation_cost) != self.num_clients:
            raise ValueError(
                f"Expected {self.num_clients} rows in allocation_cost, got {len(self.allocation_cost)}"
            )
        if any(len(row) != self.num_facilities for row in self.allocation_cost):
            raise ValueError(
                "Each row in allocation_cost must have num_facilities columns."
            )
        if len(self.opening_cost) != self.num_facilities:
            raise ValueError(
                f"Expected {self.num_facilities} opening costs, got {len(self.opening_cost)}"
            )
        # Returning the class instance itself after validation to indicate successful validation
        # and to allow further processing of the validated instance.
        return self

    def get_allocation_cost(self, client: int, facility: int) -> NonNegativeFloat:
        """
        Get the allocation cost for a specific client and facility.

        :param client: Index of the client (0-based).
        :param facility: Index of the facility (0-based).
        :return: The allocation cost from the specified client to the specified facility.
        """
        return self.allocation_cost[client][facility]

    def iter_facilities(self) -> Iterable[int]:
        """
        Iterate over the indices of all facilities.

        :return: An iterable of facility indices.
        """
        return range(self.num_facilities)

    def iter_clients(self) -> Iterable[int]:
        """
        Iterate over the indices of all clients.

        :return: An iterable of client indices.
        """
        return range(self.num_clients)


def load_from_xz(path: Path) -> FacilityLocationInstance:
    """
    Load a FacilityLocationInstance from an xz-compressed JSON file.

    :param path: Path to the xz-compressed JSON file.
    :return: An instance of FacilityLocationInstance.
    """
    import lzma

    with lzma.open(path, "rt", encoding="utf-8") as f:
        return FacilityLocationInstance.model_validate_json(f.read())


def iter_all_instances(path: Path) -> Iterable[FacilityLocationInstance]:
    """
    Iterate over all FacilityLocationInstance files in a directory.

    :param path: Path to the directory containing xz-compressed JSON files.
    :return: A list of FacilityLocationInstance objects.
    """
    for file in path.rglob("*.json.xz"):
        yield load_from_xz(file)
