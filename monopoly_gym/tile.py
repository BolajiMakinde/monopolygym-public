# monopoly_gym/tile.py
from enum import Enum
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from monopoly_gym.player import Player
class ColorSet(Enum):
    BROWN = ("Brown", (102, 51, 0), 50, 50)
    LIGHT_BLUE = ("Light Blue", (153, 204, 255), 50, 50)
    PINK = ("Pink", (255, 192, 203), 100, 100)
    ORANGE = ("Orange", (255, 165, 0), 100, 100)
    RED = ("Red", (255, 0, 0), 150, 150)
    YELLOW = ("Yellow", (255, 255, 0), 150, 150)
    GREEN = ("Green", (0, 128, 0), 200, 200)
    DARK_BLUE = ("Dark Blue", (0, 0, 139), 200, 200)

    def __init__(self, name: str, rgb: Tuple[int, int, int], house_cost: int, hotel_cost: int):
        self.color_name = name
        self.rgb = rgb
        self.house_cost = house_cost
        self.hotel_cost = hotel_cost

class SpecialTileType(Enum):
    GO = "Go"
    JAIL = "Jail"
    FREE_PARKING = "Free Parking"
    GO_TO_JAIL = "Go to Jail"

class Tile:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index
        }

class Property(Tile):
    def __init__(self, name: str, purchase_cost: int, mortgage_price: int = None, unmortgage_price: int = None):
        super().__init__(name, None)
        self.purchase_cost = purchase_cost
        self.mortgage_price = mortgage_price
        self.unmortgage_price = unmortgage_price
        self.owner: Player = None
        self.is_mortgaged = False
        self.property_idx = None

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "purchase_cost": self.purchase_cost,
            "mortgage_price": self.mortgage_price,
            "unmortgage_price": self.unmortgage_price,
            "owner": self.owner.name if self.owner else None,
            "is_mortgaged": self.is_mortgaged
        }

    @property
    def value(self) -> int:
        return self.purchase_cost

    def max_sale_value(self) -> int:
        if self.is_mortgaged:
            return 0
        return self.mortgage_price + (self.houses * self.color_set.house_cost * 0.5) + (self.hotels * self.color_set.hotel_cost * 0.5) if isinstance(self, Street) else self.mortgage_price

    def __eq__(self, other):
        if not isinstance(other, Property):
            return NotImplemented
        return self.index == other.index

    def __lt__(self, other):
        if not isinstance(other, Property):
            return NotImplemented
        return self.value < other.value

    def __repr__(self):
        return f"{self.name} (Cost: {self.purchase_cost})"
    
    def __hash__(self):
        return hash(self.index)

class Railroad(Property):
    def __init__(self, name: str, purchase_cost: int, mortgage_price: int, unmortgage_price: int):
        super().__init__(name=name, purchase_cost=purchase_cost, mortgage_price=mortgage_price, unmortgage_price=unmortgage_price)
        self.rent = [25, 50, 100, 200]  # Rent increases with the number of railroads owned

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update({
            "type": "Railroad",
            "rent": self.rent,
        })
        return base_dict

class Utility(Property):
    def __init__(self, name: str, purchase_cost: int, mortgage_price: int, unmortgage_price: int):
        super().__init__(name=name, purchase_cost=purchase_cost, mortgage_price=mortgage_price, unmortgage_price=unmortgage_price)
        self.rent_multiplier = [4, 10]  # Rent multiplier depending on dice roll

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update({
            "type": "Utility",
            "rent_multiplier": self.rent_multiplier,
        })
        return base_dict

class Street(Property):
    def __init__(self, name: str, color_set: ColorSet, purchase_cost: int, mortgage_price: int, unmortgage_price: int,
                no_color_set_rent: int, color_set_rent: int, one_house_rent: int, two_house_rent: int,
                three_house_rent: int, four_house_rent: int, hotel_rent: int):
        super().__init__(name=name, purchase_cost=purchase_cost, mortgage_price=mortgage_price, unmortgage_price=unmortgage_price)
        self.color_set = color_set
        self.rent = {
            "no_color_set": no_color_set_rent,
            "color_set": color_set_rent,
            "one_house": one_house_rent,
            "two_house": two_house_rent,
            "three_house_rent": three_house_rent,
            "four_house_rent": four_house_rent,
            "hotel": hotel_rent,
        }
        self.houses = 0
        self.hotels = 0
        self.street_idx = None

    def build(self, quantity: int):
        if quantity <= 0:
            raise Exception(...)
        if self.hotels == 1 and quantity > 0:
            raise Exception("Already has a hotel. Can't build more.")

        # If currently houses=4 and we add 1 => 5 => hotel
        if self.houses + quantity == 5:
            self.houses = 0
            self.hotels = 1
        elif self.houses + quantity < 5:
            self.houses += quantity
        else:
            raise Exception(...)



    def sell(self, quantity: int):
        if quantity <= 0:
            raise Exception(f"Failed to sell quantity={quantity} on street={self}")
        if self.hotels == 1 and quantity <= 5:
            self.houses = 5 - quantity
        elif self.hotels == 0 and self.houses >= quantity:
            self.houses -= quantity
        else:
            raise Exception(f"Invalid quanity={quantity} provided for selling on street={self}")

    def __hash__(self):
        return hash((
            super().__hash__(),
            self.color_set,
            self.houses,
            self.hotels
        ))

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update({
            "type": "Street",
            "color_set": self.color_set.color_name,
            "rent": self.rent,
            "houses": self.houses,
            "hotels": self.hotels,
        })
        return base_dict

class Tax(Tile):
    def __init__(self, name: str, tax_amount: int):
        super().__init__(name, None)
        self.name = name
        self.tax_amount = tax_amount

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "type": "Tax",
            "tax_amount": self.tax_amount,
        }


class CommunityChest(Tile):
    def __init__(self, name: str):
        super().__init__(name, None)
        self.name = name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "type": "CommunityChest",
        }

class Chance(Tile):
    def __init__(self, name: str):
        super().__init__(name, None)
        self.name = name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "type": "Chance",
        }

class SpecialTile(Tile):

    def __init__(self, name: str, special_tile_type: SpecialTileType):
        super().__init__(name, None)
        self.name = name
        self.special_tile_type = special_tile_type

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "type": "SpecialTile",
            "special_tile_type": self.special_tile_type.value,
        }