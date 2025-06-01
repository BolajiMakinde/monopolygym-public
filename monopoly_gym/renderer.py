# monopoly_gym/renderer.py

# Board geometry
from typing import List, Tuple
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from monopoly_gym.state import State

from monopoly_gym.tile import Chance, CommunityChest, Railroad, SpecialTile, SpecialTileType, Street, Tax, Utility


BOARD_SIZE = 700
CORNER_SIZE = 100
EDGE_TILE_COUNT = 9
EDGE_TILE_SIZE = (BOARD_SIZE - 2 * CORNER_SIZE) / EDGE_TILE_COUNT
TOKEN_RADIUS = 14

TILE_COLORS = {
    "Property": (220, 220, 220),
    "Railroad": (230, 230, 230),
    "Utility":  (230, 250, 250),
    "Chance":   (255, 255, 190),
    "CommunityChest": (255, 220, 255),
    "Tax":      (255, 220, 220),
    "SpecialTile": (230, 230, 230),
}

CORNER_COLORS = {
    SpecialTileType.GO:            (180, 255, 180),
    SpecialTileType.JAIL:          (255, 200, 200),
    SpecialTileType.FREE_PARKING:  (210, 210, 210),
    SpecialTileType.GO_TO_JAIL:    (255, 230, 150),
}

ASCII_CHANCE = """CHANCE
  _____
 / ??? \\
 \\ ??? /
  -----"""

ASCII_COMM_CHEST = """COMM. CHEST
  ______
 | $$$$ |
 | $$$$ |
  ------"""

ASCII_RAILROAD = """
   (  )____
  (____   \\
==O=====O==O==
"""

ASCII_UTILITY_ELEC = """
    /\\
   /  \\
  /    \\
 (      )
  \\    /
   \\  /
    \\/
"""

ASCII_UTILITY_WATER = """
   ~~~~ ~~~
  ~~~~~ ~~~
 ~~~~~~ ~~~
   ~~~~ ~~~
"""

ASCII_TAX = """
   ___
  (   )
  /   \\
  \\___/
"""

ASCII_CORNER_GO = """
   >> 
 >>    
>>>---> 
 >>     
   >>
"""

ASCII_CORNER_FREEPARKING = """
  __ 
 (  )
--====--
(      )
 '----'
"""

ASCII_CORNER_GOTOJAIL = """
  +----+
  |  | |
  |  | |
  |__|_|
"""

class Renderer():
    def __init__(self, name: str, state: State):
        pygame.init()
        self.state = state
        self.screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))

        pygame.display.set_caption(name)
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.running = True
        self.font_corner = pygame.font.SysFont("Arial", 16, bold=True)   # For big corner text
        self.font_tile   = pygame.font.SysFont("Arial", 14)             # For tile name
        self.font_tile_small   = pygame.font.SysFont("Arial", 10)             # For tile name
        self.font_cost   = pygame.font.SysFont("Arial", 13, italic=True)
        self.font_center = pygame.font.SysFont("Arial", 38, bold=True)  # "MonopolyGym" center
        self.font_ascii  = pygame.font.SysFont("Courier", 12)           # ASCII art
        self.font_ascii_small = pygame.font.SysFont("Courier", 6)  # A smaller Courier font
        self.font_player = pygame.font.SysFont("Arial", 8, bold=True)  # Player codes
    
    def get_tile_rect(self, tile_index: int) -> pygame.Rect:
        """Compute tile Rect for standard Monopoly ring layout."""
        if tile_index < 10:
            if tile_index == 0:
                return pygame.Rect(0, BOARD_SIZE - CORNER_SIZE, CORNER_SIZE, CORNER_SIZE)
            else:
                x = CORNER_SIZE + (tile_index - 1) * EDGE_TILE_SIZE
                y = BOARD_SIZE - CORNER_SIZE
                return pygame.Rect(x, y, EDGE_TILE_SIZE, CORNER_SIZE)
        elif tile_index < 20:
            if tile_index == 10:
                return pygame.Rect(BOARD_SIZE - CORNER_SIZE, BOARD_SIZE - CORNER_SIZE, CORNER_SIZE, CORNER_SIZE)
            else:
                step = tile_index - 10 - 1
                x = BOARD_SIZE - CORNER_SIZE
                y = (BOARD_SIZE - CORNER_SIZE) - (step * EDGE_TILE_SIZE) - EDGE_TILE_SIZE
                return pygame.Rect(x, y, CORNER_SIZE, EDGE_TILE_SIZE)
        elif tile_index < 30:
            if tile_index == 20:
                return pygame.Rect(BOARD_SIZE - CORNER_SIZE, 0, CORNER_SIZE, CORNER_SIZE)
            else:
                rev_index = 29 - tile_index
                x = CORNER_SIZE + rev_index * EDGE_TILE_SIZE
                y = 0
                return pygame.Rect(x, y, EDGE_TILE_SIZE, CORNER_SIZE)
        else:
            if tile_index == 30:
                return pygame.Rect(0, 0, CORNER_SIZE, CORNER_SIZE)
            step = tile_index - 30 - 1
            x = 0
            y = CORNER_SIZE + (step * EDGE_TILE_SIZE)
            return pygame.Rect(x, y, CORNER_SIZE, EDGE_TILE_SIZE)


    def draw_multiline_ascii(self, ascii_text: str, rect: pygame.Rect, color=(0, 0, 0)):
        lines = ascii_text.split("\n")
        line_height = self.font_ascii.get_linesize()
        total_height = line_height * len(lines)
        start_y = rect.centery - total_height // 2

        for line in lines:
            line_surf = self.font_ascii.render(line, True, color)
            lw, _ = self.font_ascii.size(line)
            line_x = rect.centerx - lw // 2
            self.screen.blit(line_surf, (line_x, start_y))
            start_y += line_height

    def draw_multiline_ascii_small(self, ascii_text: str, rect: pygame.Rect, color=(0, 0, 0)):
        lines = ascii_text.split("\n")
        line_height = self.font_ascii_small.get_linesize()
        total_height = line_height * len(lines)
        start_y = rect.centery - (total_height // 2)

        for line in lines:
            line_surf = self.font_ascii_small.render(line, True, color)
            lw, _ = self.font_ascii_small.size(line)
            line_x = rect.centerx - (lw // 2)
            self.screen.blit(line_surf, (line_x, start_y))
            start_y += line_height

    def draw_houses_hotels(self, street: Street, tile_rect: pygame.Rect):
        top_y = tile_rect.y + 14
        left_x = tile_rect.x + 3
        icon_size = 9

        if street.hotels == 1:
            hotel_rect = pygame.Rect(left_x, top_y, tile_rect.width - 6, icon_size + 2)
            pygame.draw.rect(self.screen, (200, 0, 0), hotel_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), hotel_rect, 1)
        else:
            for h in range(street.houses):
                hx = left_x + h * (icon_size + 2)
                house_rect = pygame.Rect(hx, top_y, icon_size, icon_size)
                pygame.draw.rect(self.screen, (0, 150, 0), house_rect)
                pygame.draw.rect(self.screen, (0, 0, 0), house_rect, 1)

    def draw_mortgage_overlay(self, tile_rect: pygame.Rect):
        overlay = pygame.Surface((tile_rect.width, tile_rect.height), pygame.SRCALPHA)
        overlay.fill((50, 50, 50, 100))
        self.screen.blit(overlay, (tile_rect.x, tile_rect.y))

    def draw_single_tile(self, tile, tile_rect: pygame.Rect):
        bg_color = TILE_COLORS["SpecialTile"]
        if isinstance(tile, SpecialTile) and tile.special_tile_type in CORNER_COLORS:
            bg_color = CORNER_COLORS[tile.special_tile_type]
        elif isinstance(tile, Street):
            bg_color = TILE_COLORS["Property"]
        elif isinstance(tile, Railroad):
            bg_color = TILE_COLORS["Railroad"]
        elif isinstance(tile, Utility):
            bg_color = TILE_COLORS["Utility"]
        elif isinstance(tile, Chance):
            bg_color = TILE_COLORS["Chance"]
        elif isinstance(tile, CommunityChest):
            bg_color = TILE_COLORS["CommunityChest"]
        elif isinstance(tile, Tax):
            bg_color = TILE_COLORS["Tax"]

        pygame.draw.rect(self.screen, bg_color, tile_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), tile_rect, 2)

        if isinstance(tile, Street):
            color_bar = pygame.Rect(tile_rect.x, tile_rect.y, tile_rect.width, 12)
            pygame.draw.rect(self.screen, tile.color_set.rgb, color_bar)
        elif isinstance(tile, Railroad):
            band = pygame.Rect(tile_rect.x, tile_rect.y, tile_rect.width, 12)
            pygame.draw.rect(self.screen, (0, 0, 0), band)

        if isinstance(tile, SpecialTile) and tile.special_tile_type in CORNER_COLORS:
            corner_label = tile.special_tile_type.value 

            lines = self.wrap_text(corner_label, tile_rect.width - 12, self.font_corner)

            line_height = self.font_corner.get_linesize()
            total_height = len(lines) * line_height

            current_y = tile_rect.centery - (total_height // 2)

            for line in lines:
                surf = self.font_corner.render(line, True, (0, 0, 0))
                line_width, _ = self.font_corner.size(line)
                x_pos = tile_rect.centerx - (line_width // 2)
                self.screen.blit(surf, (x_pos, current_y))
                current_y += line_height

        elif isinstance(tile, Street):
            self.draw_houses_hotels(tile, tile_rect)
            name_lines = self.wrap_text(tile.name, tile_rect.width - 6, self.font_tile)
            cost_line = f"${tile.purchase_cost}"

            total_line_height = (len(name_lines) * self.font_tile.get_linesize()) + self.font_cost.get_linesize()
            start_y = tile_rect.centery - total_line_height // 2

            for line in name_lines:
                if self.font_tile.size(line)[0] > tile_rect.width:
                    line_surf = self.font_tile_small.render(line, True, (0, 0, 0))
                    lw, lh = self.font_tile_small.size(line)
                    x = tile_rect.centerx - lw // 2
                    self.screen.blit(line_surf, (x, start_y))
                    start_y += lh

                else:
                    line_surf = self.font_tile.render(line, True, (0, 0, 0))
                    lw, lh = self.font_tile.size(line)
                    x = tile_rect.centerx - lw // 2
                    self.screen.blit(line_surf, (x, start_y))
                    start_y += lh

            cost_surf = self.font_cost.render(cost_line, True, (20, 20, 20))
            cw, _ = self.font_cost.size(cost_line)
            cx = tile_rect.centerx - cw // 2
            self.screen.blit(cost_surf, (cx, start_y))

            if tile.is_mortgaged:
                self.draw_mortgage_overlay(tile_rect)

        elif isinstance(tile, Railroad):
            self.draw_multiline_ascii_small(ASCII_RAILROAD, tile_rect, (0, 0, 0))
            name_lines = self.wrap_text(tile.name, tile_rect.width - 6, self.font_tile)
            cost_line = f"${tile.purchase_cost}"
            all_height = (len(name_lines) * self.font_tile.get_linesize()) + self.font_cost.get_linesize()
            start_y = tile_rect.centery - all_height // 2
            for line in name_lines:
                surf = self.font_tile.render(line, True, (0, 0, 0))
                lw, lh = self.font_tile.size(line)
                x = tile_rect.centerx - lw // 2
                self.screen.blit(surf, (x, start_y))
                start_y += lh

            c_surf = self.font_cost.render(cost_line, True, (20, 20, 20))
            cw, ch = self.font_cost.size(cost_line)
            cx = tile_rect.centerx - cw // 2
            self.screen.blit(c_surf, (cx, start_y))

            if tile.is_mortgaged:
                self.draw_mortgage_overlay(tile_rect)

        elif isinstance(tile, Utility):
            if "Water" in tile.name:
                self.draw_multiline_ascii_small(ASCII_UTILITY_WATER, tile_rect, (0, 0, 0))
            else:
                self.draw_multiline_ascii_small(ASCII_UTILITY_ELEC, tile_rect, (0, 0, 0))
            name_lines = self.wrap_text(tile.name, tile_rect.width - 6, self.font_tile)
            cost_line = f"${tile.purchase_cost}"
            all_height = (len(name_lines) * self.font_tile.get_linesize()) + self.font_cost.get_linesize()
            start_y = tile_rect.centery - all_height // 2
            for line in name_lines:
                surf = self.font_tile.render(line, True, (0, 0, 0))
                lw, lh = self.font_tile.size(line)
                x = tile_rect.centerx - lw // 2
                self.screen.blit(surf, (x, start_y))
                start_y += lh

            c_surf = self.font_cost.render(cost_line, True, (20, 20, 20))
            cw, ch = self.font_cost.size(cost_line)
            cx = tile_rect.centerx - cw // 2
            self.screen.blit(c_surf, (cx, start_y))

            if tile.is_mortgaged:
                self.draw_mortgage_overlay(tile_rect)
        elif isinstance(tile, Chance):
            self.draw_multiline_ascii(ASCII_CHANCE, tile_rect, (0, 0, 0))

        elif isinstance(tile, CommunityChest):
            self.draw_multiline_ascii(ASCII_COMM_CHEST, tile_rect, (0, 0, 0))

        elif isinstance(tile, Tax):
            self.draw_multiline_ascii_small(ASCII_TAX, tile_rect, (0, 0, 0))
            tax_name = tile.name + "?"
            tax_surf = self.font_tile.render(tax_name, True, (0, 0, 0))
            cost_surf = self.font_cost.render(f"${tile.tax_amount}", True, (150, 0, 0))

            total_h = self.font_tile.get_linesize() + self.font_cost.get_linesize()
            start_y = tile_rect.centery - total_h // 2
            tw, th = self.font_tile.size(tax_name)
            tx = tile_rect.centerx - tw // 2
            self.screen.blit(tax_surf, (tx, start_y))
            start_y += th
            cw, ch = self.font_cost.size(f"${tile.tax_amount}")
            cx = tile_rect.centerx - cw // 2
            self.screen.blit(cost_surf, (cx, start_y))

        elif isinstance(tile, SpecialTile):
            if tile.special_tile_type == SpecialTileType.GO:
                self.draw_multiline_ascii(ASCII_CORNER_GO, tile_rect, (0, 0, 0))
            elif tile.special_tile_type == SpecialTileType.FREE_PARKING:
                self.draw_multiline_ascii(ASCII_CORNER_FREEPARKING, tile_rect, (0, 0, 0))
            elif tile.special_tile_type == SpecialTileType.GO_TO_JAIL:
                self.draw_multiline_ascii(ASCII_CORNER_GOTOJAIL, tile_rect, (0, 0, 0))
            lines = self.wrap_text(tile.name, tile_rect.width - 6, self.font_tile)
            total_height = len(lines) * self.font_tile.get_linesize()
            start_y = tile_rect.centery - total_height // 2
            for line in lines:
                surf = self.font_tile.render(line, True, (0, 0, 0))
                lw, lh = self.font_tile.size(line)
                x = tile_rect.centerx - lw // 2
                self.screen.blit(surf, (x, start_y))
                start_y += lh

        else:
            lines = self.wrap_text(tile.name, tile_rect.width - 6, self.font_tile)
            total_height = len(lines) * self.font_tile.get_linesize()
            start_y = tile_rect.centery - total_height // 2
            for line in lines:
                surf = self.font_tile.render(line, True, (0, 0, 0))
                lw, lh = self.font_tile.size(line)
                x = tile_rect.centerx - lw // 2
                self.screen.blit(surf, (x, start_y))
                start_y += lh

    def wrap_text(self, text: str, max_width: int, font: pygame.font.Font) -> List[str]:
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        lines = []
        current_line = words[0]

        for word in words[1:]:
            test_line = current_line + " " + word
            w, _ = font.size(test_line)
            if w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def draw_board(self):
        self.screen.fill((245, 245, 245))

        for i, tile in enumerate(self.state.board.board):
            tile_rect = self.get_tile_rect(i)
            self.draw_single_tile(tile, tile_rect)

        center_surf = self.font_center.render("MonopolyGym", True, (60, 60, 60))
        center_rect = center_surf.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2))
        self.screen.blit(center_surf, center_rect)

        deck_width = 100
        deck_height = 70
        gap = 20
        left_deck_x = (BOARD_SIZE // 2) - deck_width - (gap // 2)
        right_deck_x = (BOARD_SIZE // 2) + (gap // 2)
        center_y = (BOARD_SIZE // 2) + 80

        chance_deck_rect = pygame.Rect(left_deck_x, center_y, deck_width + 10, deck_height)
        chest_deck_rect = pygame.Rect(right_deck_x, center_y, deck_width + 10, deck_height)

        pygame.draw.rect(self.screen, (255, 255, 190), chance_deck_rect)
        pygame.draw.rect(self.screen, (255, 220, 255), chest_deck_rect)

        pygame.draw.rect(self.screen, (0, 0, 0), chance_deck_rect, 2)
        pygame.draw.rect(self.screen, (0, 0, 0), chest_deck_rect, 2)

        chance_label = self.font_tile.render("CHANCE DECK", True, (0, 0, 0))
        chest_label = self.font_tile.render("COMM. CHEST", True, (0, 0, 0))

        chance_label_rect = chance_label.get_rect(center=(chance_deck_rect.centerx, chance_deck_rect.centery))
        chest_label_rect = chest_label.get_rect(center=(chest_deck_rect.centerx, chest_deck_rect.centery))

        self.screen.blit(chance_label, chance_label_rect)
        self.screen.blit(chest_label, chest_label_rect)


    def get_tile_center(self, tile_index: int) -> Tuple[int, int]:
        rect = self.get_tile_rect(tile_index)
        return rect.centerx, rect.centery


    def draw_players(self):
        token_colors = [
            (255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 165, 0),  # orange
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
            (128, 0, 128),  # purple
            (128, 128, 128) # gray
        ]
        for i, player in enumerate(self.state.players):
            center_x, center_y = self.get_tile_center(player.position)
            pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), TOKEN_RADIUS)
            color = token_colors[i % len(token_colors)]
            pygame.draw.circle(self.screen, color, (center_x, center_y), TOKEN_RADIUS - 2)

            # Draw mgn_code text in the circle
            code_surf = self.font_player.render(player.mgn_code, True, (255, 255, 255))
            code_rect = code_surf.get_rect(center=(center_x, center_y))
            self.screen.blit(code_surf, code_rect)

    def tick(self, framerate: float):
        # Limit the frame rate
        self.clock.tick(framerate)

    def render(self):
        """Render the current state."""
        self.draw_board()
        self.draw_players()
        pygame.display.flip()