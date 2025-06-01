# monopoly_gym/gym/board.py
from __future__ import annotations
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Type
from monopoly_gym.tile import Chance, ColorSet, CommunityChest, Property, Railroad, SpecialTile, SpecialTileType, Street, Tax, Tile, Utility

if TYPE_CHECKING:
    from monopoly_gym.player import Player
    from monopoly_gym.state import State

class Board:
    def __init__(self, houses_available: int, hotels_available: int):
        self.board : List[Tile] = [
            SpecialTile(name="GO", special_tile_type=SpecialTileType.GO),
            Street(name="Mediterranean Avenue", color_set=ColorSet.BROWN, purchase_cost=60, mortgage_price=30, 
                                unmortgage_price=33, no_color_set_rent=2, color_set_rent=4, one_house_rent=10, 
                                two_house_rent=30, three_house_rent=90, four_house_rent=160, hotel_rent=250),
            CommunityChest(name="Community Chest"),
            Street(name="Baltic Avenue", color_set=ColorSet.BROWN, purchase_cost=60, mortgage_price=30, 
                                unmortgage_price=33, no_color_set_rent=4, color_set_rent=8, one_house_rent=20, 
                                two_house_rent=60, three_house_rent=180, four_house_rent=320, hotel_rent=450),
            Tax(name="Income Tax", tax_amount=200),
            Railroad(name="Reading Railroad", purchase_cost=200, mortgage_price=100, unmortgage_price=110),
            Street(name="Oriental Avenue", color_set=ColorSet.LIGHT_BLUE, purchase_cost=100, mortgage_price=50, 
                                unmortgage_price=55, no_color_set_rent=6, color_set_rent=12, one_house_rent=30, 
                                two_house_rent=90, three_house_rent=270, four_house_rent=400, hotel_rent=550),
            Chance(name="Chance"),
            Street(name="Vermont Avenue", color_set=ColorSet.LIGHT_BLUE, purchase_cost=100, mortgage_price=50, 
                                unmortgage_price=55, no_color_set_rent=6, color_set_rent=12, one_house_rent=30, 
                                two_house_rent=90, three_house_rent=270, four_house_rent=400, hotel_rent=550),
            Street(name="Connecticut Avenue", color_set=ColorSet.LIGHT_BLUE, purchase_cost=120, mortgage_price=60, 
                                unmortgage_price=66, no_color_set_rent=8, color_set_rent=16, one_house_rent=40, 
                                two_house_rent=100, three_house_rent=300, four_house_rent=450, hotel_rent=600),
            SpecialTile(name="Jail", special_tile_type=SpecialTileType.JAIL),
            Street(name="St. Charles Place", color_set=ColorSet.PINK, purchase_cost=140, mortgage_price=70, 
                                unmortgage_price=77, no_color_set_rent=10, color_set_rent=20, one_house_rent=50, 
                                two_house_rent=150, three_house_rent=450, four_house_rent=625, hotel_rent=750),
            Utility(name="Electric Company", purchase_cost=150, mortgage_price=75, unmortgage_price=83),
            Street(name="States Avenue", color_set=ColorSet.PINK, purchase_cost=140, mortgage_price=70, 
                                unmortgage_price=77, no_color_set_rent=10, color_set_rent=20, one_house_rent=50, 
                                two_house_rent=150, three_house_rent=450, four_house_rent=625, hotel_rent=750),
            Street(name="Virginia Avenue", color_set=ColorSet.PINK, purchase_cost=160, mortgage_price=80, 
                                unmortgage_price=88, no_color_set_rent=12, color_set_rent=24, one_house_rent=60, 
                                two_house_rent=180, three_house_rent=500, four_house_rent=700, hotel_rent=900),
            Railroad(name="Pennsylvania Railroad", purchase_cost=200, mortgage_price=100, unmortgage_price=110),
            Street(name="St. James Place", color_set=ColorSet.ORANGE, purchase_cost=180, mortgage_price=90, 
                                unmortgage_price=99, no_color_set_rent=14, color_set_rent=28, one_house_rent=70, 
                                two_house_rent=200, three_house_rent=550, four_house_rent=750, hotel_rent=950),
            CommunityChest(name="Community Chest"),
            Street(name="Tennessee Avenue", color_set=ColorSet.ORANGE, purchase_cost=180, mortgage_price=90, 
                                unmortgage_price=99, no_color_set_rent=14, color_set_rent=28, one_house_rent=70, 
                                two_house_rent=200, three_house_rent=550, four_house_rent=750, hotel_rent=950),
            Street(name="New York Avenue", color_set=ColorSet.ORANGE, purchase_cost=200, mortgage_price=100, 
                                unmortgage_price=110, no_color_set_rent=16, color_set_rent=32, one_house_rent=80, 
                                two_house_rent=220, three_house_rent=600, four_house_rent=800, hotel_rent=1000),
            SpecialTile(name="Free Parking", special_tile_type=SpecialTileType.FREE_PARKING),
            Street(name="Kentucky Avenue", color_set=ColorSet.RED, purchase_cost=220, mortgage_price=110, 
                                unmortgage_price=121, no_color_set_rent=18, color_set_rent=36, one_house_rent=90, 
                                two_house_rent=250, three_house_rent=700, four_house_rent=875, hotel_rent=1050),
            Chance(name="Chance"),
            Street(name="Indiana Avenue", color_set=ColorSet.RED, purchase_cost=220, mortgage_price=110, 
                                unmortgage_price=121, no_color_set_rent=18, color_set_rent=36, one_house_rent=90, 
                                two_house_rent=250, three_house_rent=700, four_house_rent=875, hotel_rent=1050),
            Street(name="Illinois Avenue", color_set=ColorSet.RED, purchase_cost=240, mortgage_price=120, 
                                unmortgage_price=132, no_color_set_rent=20, color_set_rent=40, one_house_rent=100, 
                                two_house_rent=300, three_house_rent=750, four_house_rent=925, hotel_rent=1100),
            Railroad(name="B&O Railroad", purchase_cost=200, mortgage_price=100, unmortgage_price=110),
            Street(name="Atlantic Avenue", color_set=ColorSet.YELLOW, purchase_cost=260, mortgage_price=130, 
                                unmortgage_price=143, no_color_set_rent=22, color_set_rent=44, one_house_rent=110, 
                                two_house_rent=330, three_house_rent=800, four_house_rent=975, hotel_rent=1150),
            Street(name="Ventnor Avenue", color_set=ColorSet.YELLOW, purchase_cost=260, mortgage_price=130, 
                                unmortgage_price=143, no_color_set_rent=22, color_set_rent=44, one_house_rent=110, 
                                two_house_rent=330, three_house_rent=800, four_house_rent=975, hotel_rent=1150),
            Utility(name="Water Works", purchase_cost=150, mortgage_price=75, unmortgage_price=83),
            Street(name="Marvin Gardens", color_set=ColorSet.YELLOW, purchase_cost=280, mortgage_price=140, 
                                unmortgage_price=154, no_color_set_rent=24, color_set_rent=48, one_house_rent=120, 
                                two_house_rent=360, three_house_rent=850, four_house_rent=1025, hotel_rent=1200),
            SpecialTile(name="Go To Jail", special_tile_type=SpecialTileType.GO_TO_JAIL),
            Street(name="Pacific Avenue", color_set=ColorSet.GREEN, purchase_cost=300, mortgage_price=150, 
                                unmortgage_price=165, no_color_set_rent=26, color_set_rent=52, one_house_rent=130, 
                                two_house_rent=390, three_house_rent=900, four_house_rent=1100, hotel_rent=1275),
            Street(name="North Carolina Avenue", color_set=ColorSet.GREEN, purchase_cost=300, mortgage_price=150, 
                                unmortgage_price=165, no_color_set_rent=26, color_set_rent=52, one_house_rent=130, 
                                two_house_rent=390, three_house_rent=900, four_house_rent=1100, hotel_rent=1275),
            CommunityChest(name="Community Chest"),
            Street(name="Pennsylvania Avenue", color_set=ColorSet.GREEN, purchase_cost=320, mortgage_price=160, 
                                unmortgage_price=176, no_color_set_rent=28, color_set_rent=56, one_house_rent=150, 
                                two_house_rent=450, three_house_rent=1000, four_house_rent=1200, hotel_rent=1400),
            Railroad(name="Short Line", purchase_cost=200, mortgage_price=100, unmortgage_price=110),
            Chance(name="Chance"),
            Street(name="Park Place", color_set=ColorSet.DARK_BLUE, purchase_cost=350, mortgage_price=175, 
                                unmortgage_price=193, no_color_set_rent=35, color_set_rent=70, one_house_rent=175, 
                                two_house_rent=500, three_house_rent=1100, four_house_rent=1300, hotel_rent=1500),
            Tax(name="Luxury Tax", tax_amount=100),
            Street(name="Boardwalk", color_set=ColorSet.DARK_BLUE, purchase_cost=400, mortgage_price=200, 
                                unmortgage_price=220, no_color_set_rent=50, color_set_rent=100, one_house_rent=200, 
                                two_house_rent=600, three_house_rent=1400, four_house_rent=1700, hotel_rent=2000),
        ]

        self.chance_cards: List[Tuple[int, str, Callable[[State], None]]] = [
            (
                1,
                "Advance to Go (Collect $200).",
                lambda state: (
                    setattr(state.current_player(), "position", 0),
                    setattr(state.current_player(), "balance", state.current_player().balance + 200 if state.current_player().balance + 200 >= 0 else 0),
                ),

            ),
            (
                2,
                "Advance to Illinois Avenue. If you pass Go, collect $200.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 200 if state.current_player().position > 24 else state.current_player().balance),
                    setattr(state.current_player(), "position", 24),
                ),
            ),
            (
                3,
                "Advance to St. Charles Place. If you pass Go, collect $200.",
                lambda state: (
                    setattr(state.current_player(), "position", 11),
                    setattr(state.current_player(), "balance", state.current_player().balance + 200 if state.current_player().position > 11 else state.current_player().balance),
                ),
            ),
            (
                4,
                "Advance token to the nearest Utility. If unowned, you may buy it from the Bank.",
                lambda state: (
                    setattr(state.current_player(), "position", self._find_nearest(state, state.current_player(), Utility)),
                ),
            ),
            (
                5,
                "Advance token to the nearest Railroad and pay owner twice the rental to which they are otherwise entitled.",
                lambda state: (
                    setattr(state.current_player(), "position", self._find_nearest(state, state.current_player(), Railroad)),
                ),
            ),
            (
                6,
                "Bank pays you dividend of $50.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 50),
                ),
            ),
            (
                7,
                "Get Out of Jail Free. This card may be kept until needed, or traded/sold.",
                lambda state: (
                    setattr(state.current_player(), "jail_free_cards", state.current_player().jail_free_cards + 1),
                ),
            ),
            (
                8,
                "Go Back 3 Spaces.",
                lambda state: (
                    setattr(state.current_player(), "position", max(state.current_player().position - 3, 0)),
                ),
            ),
            (
                9,
                "Go to Jail. Go directly to jail, do not pass Go, do not collect $200.",
                lambda state: (
                    setattr(state.current_player(), "position", 10),
                    setattr(state.current_player(), "in_jail", True),
                    setattr(state.current_player(), "jail_turns", 0),
                ),
            ),
            (
                10,
                "Make general repairs on all your property: For each house pay $25, for each hotel pay $100.",
                lambda state: (
                    setattr(
                        state.current_player(),
                        "balance",
                        max(
                            state.current_player().balance
                            - sum(0 if not isinstance(prop, Street) else prop.houses for prop in state.current_player().properties) * 25
                            - sum(0 if not isinstance(prop, Street) else prop.hotels for prop in state.current_player().properties) * 100,
                            0,
                        ),
                    ),
                ),
            ),
            (
                11,
                "Pay poor tax of $15.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance - 15),
                ),
            ),
            (
                12,
                "Take a trip to Reading Railroad. If you pass Go, collect $200.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 200 if state.current_player().position > 5 else state.current_player().balance),
                    setattr(state.current_player(), "position", 5),
                ),
            ),
            (
                13,
                "Take a walk on the Boardwalk. Advance token to Boardwalk.",
                lambda state: (
                    setattr(state.current_player(), "position", 39),
                ),
            ),
            (
                14,
                "You have been elected Chairman of the Board. Pay each player $50.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance - (50 * (len(state.players) - 1))),
                    *[
                        (
                            setattr(p, "balance", p.balance + 50),
                        )
                        for p in state.players
                        if p != state.current_player()
                    ],
                ),
            ),
            (
                15,
                "Your building loan matures. Collect $150.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 150),
                ),
            ),
            (
                16,
                "Receive for services $25.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 25),
                ),
            ),
        ]

        self.community_chest_cards: List[Tuple[int, str, Callable[[State], None]]] = [
            (
                1,
                "Advance to Go (Collect $200).",
                lambda state: (
                    setattr(state.current_player(), "position", 0),
                    setattr(state.current_player(), "balance", state.current_player().balance + 200 if state.current_player().balance + 200 >= 0 else 0),
                ),
            ),
            (
                2,
                "Bank error in your favor. Collect $200.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 200 if state.current_player().balance + 200 >= 0 else 0),
                ),
            ),
            (
                3,
                "Doctor's fees. Pay $50.",
                lambda state: (
                    setattr(state.current_player(), "balance", max(state.current_player().balance - 50, 0)),
                ),
            ),
            (
                4,
                "From sale of stock you get $50.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 50 if state.current_player().balance + 50 >= 0 else 0),
                ),
            ),
            (
                5,
                "Get Out of Jail Free. This card may be kept until needed, or traded/sold.",
                lambda state: (
                    setattr(state.current_player(), "jail_free_cards", state.current_player().jail_free_cards + 1),
                ),
            ),
            (
                6,
                "Go to Jail. Go directly to jail, do not pass Go, do not collect $200.",
                lambda state: (
                    setattr(state.current_player(), "position", 10),
                    setattr(state.current_player(), "in_jail", True),
                    setattr(state.current_player(), "jail_turns", 0),
                ),
            ),
            (
                7,
                "Grand Opera Night. Collect $50 from every player for opening night seats.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 50 if state.current_player().balance + 50 >= 0 else 0),
                    *[
                        (
                            setattr(p, "balance", max(p.balance - 50, 0)),
                        )
                        for p in state.players
                        if p != state.current_player()
                    ],
                ),
            ),
            (
                8,
                "Holiday Fund matures. Receive $100.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 100 if state.current_player().balance + 100 >= 0 else 0),
                ),
            ),
            (
                9,
                "Income tax refund. Collect $20.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 20 if state.current_player().balance + 20 >= 0 else 0),
                ),
            ),
            (
                10,
                "It is your birthday. Collect $10 from every player.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 10 if state.current_player().balance + 10 >= 0 else 0),
                    *[
                        (
                            setattr(p, "balance", max(p.balance - 10, 0)),
                        )
                        for p in state.players
                        if p != state.current_player()
                    ],
                ),
            ),
            (
                11,
                "Life insurance matures. Collect $100.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 100 if state.current_player().balance + 100 >= 0 else 0),
                ),
            ),
            (
                12,
                "Pay hospital fees of $100.",
                lambda state: (
                    setattr(state.current_player(), "balance", max(state.current_player().balance - 100, 0)),
                ),
            ),
            (
                13,
                "Pay school fees of $150.",
                lambda state: (
                    setattr(state.current_player(), "balance", max(state.current_player().balance - 150, 0)),
                ),
            ),
            (
                14,
                "Receive $25 consultancy fee.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 25 if state.current_player().balance + 25 >= 0 else 0),
                ),
            ),
            (
                15,
                "You inherit $100.",
                lambda state: (
                    setattr(state.current_player(), "balance", state.current_player().balance + 100 if state.current_player().balance + 100 >= 0 else 0),
                ),
            ),
            (
                16,
                "You are assessed for street repairs: Pay $40 per house and $115 per hotel.",
                lambda state: (
                    setattr(
                        state.current_player(),
                        "balance",
                        max(
                            state.current_player().balance
                            - sum(0 if not isinstance(prop, Street) else prop.houses for prop in state.current_player().properties) * 40
                            - sum(0 if not isinstance(prop, Street) else prop.hotels for prop in state.current_player().properties) * 115,
                            0,
                        ),
                    ),
                ),
            ),
        ]

        self.houses_available = houses_available
        self.hotels_available = hotels_available
        self.properties = []
        self.streets = []
        property_idx = 0
        street_idx = 0
        for i, tile in enumerate(self.board):
            tile.index = i
            if isinstance(tile, Property):
                tile.property_idx = property_idx
                self.properties.append(tile)
                property_idx += 1
            if isinstance(tile, Street):
                tile.street_idx = street_idx
                self.streets.append(tile)
                street_idx += 1

    def _find_nearest(self, state: State, player: Player, tile_type: Type[Tile]) -> int:
        current_pos = state.current_player().position
        for offset in range(1, len(self.board) + 1):
            next_pos = (current_pos + offset) % len(self.board)
            if isinstance(self.board[next_pos], tile_type):
                return next_pos
        return current_pos
    
    def generate_board_from_tiles(self, tiles: List[Tile], houses_available: int, hotels_available: int):
        self.board = []
        for tile_idx, tile in enumerate(tiles):
            if isinstance(tile, Property):
                tile.index = tile_idx
            self.board.append(tile)
        self.houses_available = houses_available
        self.hotels_available = hotels_available

    def get_property_by_index(self, index: int) -> Optional[Property]:
        return self.board[index]

    def get_properties_by_color(self, color_set: ColorSet) -> List[Property]:
        properties = [
            tile for tile in self.board
            if isinstance(tile, Property) and tile.color_set == color_set
        ]
        return properties