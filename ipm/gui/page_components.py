import pygame
import time
import torch
import random
from ipm.models.idct_helpers import convert_decision_to_leaf, convert_leaf_to_decision
from typing import Callable
from abc import ABC, abstractmethod


class GUIItem(ABC):
    @abstractmethod
    def show(self):
        pass

    def show_children(self):
        pass

    def process_event(self, event):
        pass

    def hide(self):
        self.showing = False

    def process_standby(self):
        self.show()


class GUIButton(GUIItem):
    def __init__(self, surface: pygame.Surface, position: tuple, size: tuple, event_fn: Callable,
                 text: str, font_size: int = 18, text_color: str = 'white', transparent: bool = True,
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.text = text
        self.position = position
        self.pos_x, self.pos_y = self.position
        self.size = size
        self.surface = surface
        self.size_x, self.size_y = self.size
        self.font = pygame.font.Font('freesansbold.ttf', font_size)
        self.rect_color = rect_color
        self.border_color = border_color
        self.border_width = border_width
        self.transparent = transparent
        self.text_color = text_color
        self.event_fn = event_fn

        if self.rect_color is None:
            self.rect_color = (0, 0, 0, 255)
        if self.border_color is None:
            self.border_color = (255, 255, 255, 255)

        self.rectangle = pygame.Rect((self.pos_x, self.pos_y, self.size_x, self.size_y))

        if self.transparent:
            self.rect_shape = pygame.Surface(self.rectangle.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rectangle.size)
        self.highlighting = False
        self.showing = False

    def show(self):
        self.showing = True
        if not self.highlighting:
            self.rect_shape.fill(self.rect_color)
            self.surface.blit(self.rect_shape, self.position)
            if self.border_width > 0:
                pygame.draw.rect(self.surface, self.border_color, self.rectangle, width=self.border_width)
            text_rendered = self.font.render(self.text, True, pygame.Color(self.text_color))
            text_rect = text_rendered.get_rect(center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
            self.cursor = pygame.Rect(text_rect.topright, (3, text_rect.height + 2))
            self.surface.blit(text_rendered, text_rect)
        else:
            highlight_color = (69, 69, 69)
            highlight_border_color = (2, 168, 2)
            highlight_text_color = (2, 168, 2)
            self.rect_shape.fill(highlight_color)
            self.surface.blit(self.rect_shape, self.position)
            self.rect_shape.fill(self.rect_color)
            self.surface.blit(self.rect_shape, self.position)
            if self.border_width > 0:
                pygame.draw.rect(self.surface, highlight_border_color, self.rectangle, width=self.border_width)
            text_rendered = self.font.render(self.text, True, highlight_text_color)
            text_rect = text_rendered.get_rect(
                center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
            self.surface.blit(text_rendered, text_rect)

    def process_event(self, event):
        mouse_position = pygame.mouse.get_pos()
        if self.showing and self.rectangle.collidepoint(mouse_position):
            self.highlighting = True
            if event.type == pygame.MOUSEBUTTONUP:
                self.event_fn()
        else:
            self.highlighting = False
        return True, None


class GUITriggerButton(GUIButton):
    def __init__(self, surface: pygame.Surface, position: tuple, size: tuple,
                 text: str, font_size: int = 18, text_color: str = 'white', transparent: bool = True,
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        super().__init__(surface=surface, position=position, size=size, event_fn=None, text=text,
                         font_size=font_size, text_color=text_color, transparent=transparent,
                         rect_color=rect_color, border_color=border_color, border_width=border_width)
        self.return_text = text

    def process_event(self, event):
        mouse_position = pygame.mouse.get_pos()
        if self.showing and self.rectangle.collidepoint(mouse_position):
            self.highlighting = True
            if event.type == pygame.MOUSEBUTTONUP:
                return self.return_text, None
        else:
            self.highlighting = False
        return True, None


class Legend(GUIItem):
    def __init__(self, surface, x, y, w, h, decision_color, action_color,
                 decision_border_color, action_border_color, highlight_color, font,
                 option_list, selected=-1, transparent=True):
        self.decision_color = decision_color
        self.decision_border_color = decision_border_color
        self.action_color = action_color
        self.action_border_color = action_border_color
        self.position = (x, y)

        self.font = pygame.font.Font('freesansbold.ttf', 24)
        self.small_font = pygame.font.Font('freesansbold.ttf', 18)
        self.surface = surface
        #
        # self.rect = pygame.Rect(x, y, w, h)
        # self.rect_shape = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        # self.rect_text = pygame.Rect(x + w + 30, y, w, h)
        # self.rect_text_shape = pygame.Surface(self.rect.size)

        self.rect_decision = pygame.Rect(x, y, w, h)
        self.rect_decision_shape = pygame.Surface(self.rect_decision.size, pygame.SRCALPHA)
        self.rect_decision_text = pygame.Rect(x + w + 30, y, w, h)
        self.rect_decision_text_shape = pygame.Surface(self.rect_decision.size)

        y2 = y + h + 10
        self.position2 = (x, y2)
        self.rect_action = pygame.Rect(x, y2, w, h)
        self.rect_action_shape = pygame.Surface(self.rect_action.size, pygame.SRCALPHA)
        self.rect_action_text = pygame.Rect(x + w + 30, y2, w, h)
        self.rect_action_text_shape = pygame.Surface(self.rect_action_text.size)

    def show(self):
        # self.rect_shape.fill((255, 255, 255))
        # self.surface.blit(self.rect_shape, self.position)
        # pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect, width=2)
        # msg = self.font.render(" = Modifiable", 1, (0, 0, 0))
        # x, y = self.position
        # x += 130
        # y += 8
        # self.surface.blit(msg, (x, y))

        self.rect_decision_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_decision_shape, self.position)
        self.rect_decision_shape.fill(self.decision_color)
        self.surface.blit(self.rect_decision_shape, self.position)
        pygame.draw.rect(self.surface, self.decision_border_color, self.rect_decision, width=2)
        msg = self.font.render(" = Decision Node", 1, (0, 0, 0))
        x, y = self.position
        x += 130
        y += 8
        self.surface.blit(msg, (x, y))

        self.rect_action_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_action_shape, self.position2)
        self.rect_action_shape.fill(self.action_color)
        self.surface.blit(self.rect_action_shape, self.position2)
        pygame.draw.rect(self.surface, self.action_border_color, self.rect_action, width=2)
        msg = self.font.render(" = Action Node", 1, (0, 0, 0))
        x, y = self.position2
        x += 130
        y += 8
        self.surface.blit(msg, (x, y))



        msg = self.small_font.render("Key", 1, (0, 0, 0))
        x += 100
        y += 85
        self.some_rect = pygame.Rect(x-5, y-5, 150, 100)
        self.some_rect_shape = pygame.Surface(self.some_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(self.surface, (0, 0, 0), self.some_rect, width=2)
        self.surface.blit(msg, (x, y))

        msg = self.small_font.render("---------------------", 1, (0, 0, 0))
        y += 25
        self.surface.blit(msg, (x, y))

        msg = self.small_font.render(" C: Counter", 1, (0, 0, 0))
        y += 25
        self.surface.blit(msg, (x, y))

        msg = self.small_font.render(" D: Dispenser", 1, (0, 0, 0))
        y += 25
        self.surface.blit(msg, (x, y))

    def process_event(self, event):
        return 'continue', None


class OptionBox(GUIItem):
    def __init__(self, surface, x, y, w, h, settings, domain_idx, gui_node_idx, color, highlight_color, font,
                 option_list, lock_menu=False, selected=-1, transparent=True, max_len=20, num_visible_options=20,
                 frozen=False):
        self.color = color
        self.highlight_color = highlight_color
        self.settings = settings
        self.domain_idx = domain_idx
        self.gui_node_idx = gui_node_idx
        self.lock_menu = lock_menu
        self.frozen = frozen
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.max_len = max_len
        self.rect = pygame.Rect(x, y, w, h)
        self.transparent = transparent
        if transparent:
            self.rect_shape = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rect.size)
        self.position = (x, y)
        self.font = font
        self.option_list = option_list
        if selected == -1:
            self.selected = random.randint(0, len(option_list) - 1)
        else:
            self.selected = selected
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1
        self.previously_selected = self.selected
        self.previous_action_option = -1
        self.surface = surface
        self.scroll_y = self.selected
        self.num_visible_options = num_visible_options

    def show(self):
        self.rect_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_shape, self.position)
        self.rect_shape.fill(self.highlight_color if self.menu_active else self.color)
        self.surface.blit(self.rect_shape, self.position)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect, width=2)
        text = self.option_list[self.selected]
        if len(text) > self.max_len:
            text = text[:self.max_len] + '..'
        else:
            text = text

        msg = self.font.render(text, 1, (0, 0, 0))
        x, y = self.rect.center
        self.surface.blit(msg, msg.get_rect(center=(x - 5, y)))
        # draw triangle, upside down at the right
        # pygame.draw.polygon(self.surface, (0, 0, 0, 128), ((self.rect.right - 12, self.rect.bottom - 10),
        #                                                       (self.rect.right - 7, self.rect.top + 20),
        #                                                       (self.rect.right - 17, self.rect.top + 20)))

        pygame.draw.polygon(self.surface, (0, 0, 0, 128), ((self.rect.right - 12 // 2, self.rect.bottom - 5),
                                                           (self.rect.right - 7 // 2, self.rect.top + 10),
                                                           (self.rect.right - 17 // 2, self.rect.top + 10)))

        # pygame.draw.polygon(self.surface, (0, 0, 0), ((self.rect.x + self.rect.w - 10, self.rect.y + 10),
        #                                                         (self.rect.x + self.rect.w - 10, self.rect.y + self.rect.h - 10),
        #                                                         (self.rect.x + self.rect.w - 20, self.rect.y + self.rect.h / 2)), 0)

        if self.draw_menu:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)

            if num_visible_options < max_len:
                lower_idx = min(self.scroll_y, max_len - num_visible_options)
                upper_idx = min(self.scroll_y + num_visible_options, max_len)
            else:
                lower_idx = 0
                upper_idx = max_len

            if num_visible_options < max_len:
                rect = self.rect.copy()
                w, h = rect.size
                w += 45
                new_rect = pygame.Rect(rect.x, rect.y, w, h)

                # we want a little progress bar on the right
                bar_height = 10
                bar_width = bar_height // 2

                total_y_len = self.rect.height * (num_visible_options) - bar_height
                assert 0 <= self.scroll_y <= max_len - num_visible_options
                percentage_done = self.scroll_y / (max_len - num_visible_options)
                bar_y = self.rect.y + h + total_y_len * percentage_done
                bar_rect = pygame.Rect(new_rect.right - bar_width, bar_y, bar_width, bar_height)

            for i, item_idx in enumerate(range(lower_idx, upper_idx)):
                text = self.option_list[item_idx]
                rect = self.rect.copy()
                rect.y += (i + 1) * self.rect.height
                w, h = rect.size
                w += 45
                new_rect_size = (w, h)
                new_rect = pygame.Rect(rect.x, rect.y, w, h)
                if self.transparent:
                    rect_shape = pygame.Surface(new_rect_size, pygame.SRCALPHA)
                else:
                    rect_shape = pygame.Surface(new_rect_size)

                rect_shape.fill((255, 255, 255, 255))
                self.surface.blit(rect_shape, (rect.x, rect.y))
                rect_shape.fill(self.highlight_color if item_idx == self.active_option else self.color)
                self.surface.blit(rect_shape, (rect.x, rect.y))
                pygame.draw.rect(self.surface, (0, 0, 0, 128), new_rect, width=2)

                msg = self.font.render(text, 1, (0, 0, 0))
                self.surface.blit(msg, msg.get_rect(center=new_rect.center))

            if num_visible_options < max_len:
                rect_shape = pygame.Surface(bar_rect.size)
                rect_shape.fill((0, 0, 0))
                self.surface.blit(rect_shape, (bar_rect.x, bar_rect.y))
            # pygame.draw.rect(self.surface, (0, 0, 0, 128), bar_rect, width=2)

        else:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)
            self.scroll_y = min(self.scroll_y, max_len - num_visible_options)

    def process_event(self, event):
        if self.frozen:
            return -1
        x = int(self.x * self.settings.zoom) - self.settings.offset_x
        y = int(self.y * self.settings.zoom) - self.settings.offset_y
        current_w = int(self.w * self.settings.zoom)
        current_h = int(self.h * self.settings.zoom)
        current_rect = pygame.Rect(x, y, current_w, current_h)
        in_bounds = 0 < x + current_w < self.settings.width and 0 < y + current_h < self.settings.height
        if not in_bounds:
            return -1

        mpos = pygame.mouse.get_pos()
        self.menu_active = current_rect.collidepoint(mpos)

        num_visible_options = self.num_visible_options
        max_len = len(self.option_list)
        if num_visible_options < max_len:
            lower_idx = min(self.scroll_y, max_len - num_visible_options)
            upper_idx = min(self.scroll_y + num_visible_options, max_len)
        else:
            lower_idx = 0
            upper_idx = max_len

        for i, item_idx in enumerate(range(lower_idx, upper_idx)):
            rect = current_rect.copy()
            rect.y += (i + 1) * current_rect.height
            if rect.collidepoint(mpos):
                self.active_option = item_idx
                break

        # if not self.menu_active and self.active_option == -1:
        #     self.draw_menu = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.menu_active:
                self.draw_menu = not self.draw_menu
            elif self.draw_menu and self.active_option >= 0:
                self.previous_selected = self.selected
                self.selected = self.active_option
                self.draw_menu = False
                return self.active_option

        if self.lock_menu:
            self.settings.options_menus_per_domain[self.domain_idx][self.gui_node_idx] = self.draw_menu

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 5:
                self.scroll_y += 1
                self.scroll_y = min(self.scroll_y, len(self.option_list) - self.num_visible_options)
            elif event.button == 4:
                self.scroll_y -= 1
                self.scroll_y = max(self.scroll_y, 0)
        return -1

    def process_standby(self):
        self.show()


class TextBox(GUIItem):
    def __init__(self, surface, settings, x, y, w, h, color, highlight_color,
                 font, value='0.0', transparent=True):
        self.color = color
        self.highlight_color = highlight_color
        self.rect = pygame.Rect(x, y, w, h)
        self.transparent = transparent
        self.settings = settings
        if transparent:
            self.rect_shape = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rect.size)
        self.position = (x, y)
        self.font = font
        self.value = value
        self.previous_value = value
        self.menu_active = False
        self.surface = surface
        self.main_font = font
        self.currently_editing = False
        self.x = x
        self.y = y
        self.h = h
        self.w = w

        text_rendered = self.main_font.render(value, True, pygame.Color((0, 0, 0)))
        self.text_rect = text_rendered.get_rect(center=(x + w // 2, y + h // 2))
        self.surface.blit(text_rendered, self.text_rect)

        self.cursor = pygame.Rect(self.text_rect.topright, (3, self.text_rect.height + 2))

    def show(self):
        self.rect_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_shape, self.position)

        mpos = pygame.mouse.get_pos()
        x = int(self.x * self.settings.zoom) - self.settings.offset_x
        y = int(self.y * self.settings.zoom) - self.settings.offset_y
        current_w = int(self.w * self.settings.zoom)
        current_h = int(self.h * self.settings.zoom)
        current_rect = pygame.Rect(x, y, current_w, current_h)

        self.rect_shape.fill(self.highlight_color if current_rect.collidepoint(mpos) else self.color)

        self.surface.blit(self.rect_shape, self.position)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect, width=2)
        text_rendered = self.main_font.render(self.value, True, pygame.Color((0, 0, 0)))
        text_rect = text_rendered.get_rect(
            center=(self.x + self.w // 2, self.y + self.h // 2))
        self.surface.blit(text_rendered, text_rect)

        if self.currently_editing:
            if time.time() % 1 > 0.5:
                text_rendered = self.main_font.render(self.value, True, pygame.Color((0, 0, 0)))
                text_rect = text_rendered.get_rect(center=(self.x + self.w // 2, self.y + self.h // 2))
                self.cursor.midleft = text_rect.midright
                pygame.draw.rect(self.surface, (0, 0, 0), self.cursor)

    def process_event(self, event):
        x = int(self.x * self.settings.zoom) - self.settings.offset_x
        y = int(self.y * self.settings.zoom) - self.settings.offset_y
        current_w = int(self.w * self.settings.zoom)
        current_h = int(self.h * self.settings.zoom)
        current_rect = pygame.Rect(x, y, current_w, current_h)

        mpos = pygame.mouse.get_pos()
        self.menu_active = current_rect.collidepoint(mpos)

        if self.menu_active:
            if event.type == pygame.MOUSEBUTTONUP:
                self.currently_editing = True

        if self.currently_editing:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.value = self.value[:-1]
                elif event.key == pygame.K_RETURN and len(self.value) > 0:
                    self.currently_editing = False
                elif '%' in self.value:
                    self.currently_editing = False
                elif event.type == pygame.QUIT:
                    return False
                else:
                    self.value = self.value + event.unicode


class TextObjectBox(GUIItem):
    def __init__(self, surface, x, y, w, h, settings, domain_idx, gui_node_idx, color, highlight_color, font,
                 option_list, lock_menu=False, selected=-1, transparent=True, max_len=20, num_visible_options=20,
                 value=[0, 0, 0], frozen=True):
        self.color = color
        self.highlight_color = highlight_color
        self.settings = settings
        self.domain_idx = domain_idx
        self.gui_node_idx = gui_node_idx
        self.lock_menu = lock_menu
        self.frozen = frozen
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.max_len = max_len
        self.rect = pygame.Rect(x, y, w, h)
        self.shifter = 50
        self.rect1 = pygame.Rect(x, y + self.shifter, w, h)
        self.rect2 = pygame.Rect(x, y + 2 * self.shifter, w, h)
        self.transparent = transparent
        if transparent:
            self.rect_shape = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            self.rect_shape1 = pygame.Surface(self.rect1.size, pygame.SRCALPHA)
            self.rect_shape2 = pygame.Surface(self.rect2.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rect.size)

        self.position = (x, y)
        self.position1 = (x, y + self.shifter)
        self.position2 = (x, y + 2 * self.shifter)
        self.font = font
        self.value = str(round(value[0], 2))
        self.value1 = str(round(value[1], 2))
        self.value2 = str(round(value[2], 2))
        self.previous_value = value
        self.menu_active = False
        self.surface = surface
        self.main_font = font
        self.currently_editing = False
        self.currently_editing1 = False
        self.currently_editing2 = False

        text_rendered = self.main_font.render(self.value, True, pygame.Color((0, 0, 0)))
        self.text_rect = text_rendered.get_rect(center=(x + w // 2, y + h // 2))

        text_rendered2 = self.main_font.render(self.value1, True, pygame.Color((0, 0, 0)))
        self.text_rect2 = text_rendered.get_rect(center=(x + w // 2, (y + self.shifter) + h // 2))

        text_rendered3 = self.main_font.render(self.value2, True, pygame.Color((0, 0, 0)))
        self.text_rect3 = text_rendered.get_rect(center=(x + w // 2, (y + 2 * self.shifter) + h // 2))

        # show each text box
        self.surface.blit(text_rendered, self.text_rect)
        self.surface.blit(text_rendered2, self.text_rect2)
        self.surface.blit(text_rendered3, self.text_rect3)

        self.position_ = (x, y + self.shifter / 2)
        self.position_1 = (x, y + 3 * self.shifter / 2)
        self.position_2 = (x, y + 5 * self.shifter / 2)
        self.rect_ = pygame.Rect(x, y + self.shifter / 2, w, h)
        self.rect_1 = pygame.Rect(x, y + 3 * self.shifter / 2, w, h)
        self.rect_2 = pygame.Rect(x, y + 5 * self.shifter / 2, w, h)
        self.rect_shape_ = pygame.Surface(self.rect_.size, pygame.SRCALPHA)
        self.rect_shape_1 = pygame.Surface(self.rect_1.size, pygame.SRCALPHA)
        self.rect_shape_2 = pygame.Surface(self.rect_2.size, pygame.SRCALPHA)
        self.cursor = pygame.Rect(self.text_rect.topright, (3, self.text_rect.height + 2))
        self.option_list = option_list
        if selected == -1:
            self.selected = random.randint(0, len(option_list) - 1)
            self.selected_1 = random.randint(0, len(option_list) - 1)
            self.selected_2 = random.randint(0, len(option_list) - 1)
        else:
            self.selected = selected[0]
            self.selected_1 = selected[1]
            self.selected_2 = selected[2]
        self.draw_menu = False
        self.draw_menu_1 = False
        self.draw_menu_2 = False
        self.menu_active = False
        self.menu_active1 = False
        self.menu_active2 = False
        self.menu_active_ = False
        self.menu_active_1 = False
        self.menu_active_2 = False
        self.active_option = -1
        self.active_option_1 = -1
        self.active_option_2 = -1
        self.previously_selected = self.selected
        self.previously_selected_1 = self.selected_1
        self.previously_selected_2 = self.selected_2
        self.previous_action_option = -1
        self.surface = surface
        self.scroll_y = self.selected
        self.scroll_y_1 = self.selected_2
        self.scroll_y_2 = self.selected_2
        self.num_visible_options = num_visible_options

    def show(self):
        self.rect_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_shape, self.position)

        self.rect_shape1.fill((255, 255, 255))
        self.surface.blit(self.rect_shape1, self.position1)

        self.rect_shape2.fill((255, 255, 255))
        self.surface.blit(self.rect_shape2, self.position2)

        mpos = pygame.mouse.get_pos()
        x = int(self.x * self.settings.zoom) - self.settings.offset_x
        y = int(self.y * self.settings.zoom) - self.settings.offset_y
        current_w = int(self.w * self.settings.zoom)
        current_h = int(self.h * self.settings.zoom)
        current_rect = pygame.Rect(x, y, current_w, current_h)
        current_rect1 = pygame.Rect(x, y + self.shifter, current_w, current_h)
        current_rect2 = pygame.Rect(x, y + 2 * self.shifter, current_w, current_h)

        colliding_current_rect = current_rect.collidepoint(mpos) and not self.frozen
        self.rect_shape.fill(self.highlight_color if colliding_current_rect else self.color)


        self.surface.blit(self.rect_shape, self.position)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect, width=2)
        text_rendered = self.main_font.render(self.value, True, pygame.Color((0, 0, 0)))
        text_rect = text_rendered.get_rect(
            center=(self.x + self.w // 2, self.y + self.h // 2))
        self.surface.blit(text_rendered, text_rect)

        colliding_current_rect1 = current_rect1.collidepoint(mpos) and not self.frozen
        self.rect_shape1.fill(self.highlight_color if colliding_current_rect1 else self.color)

        self.surface.blit(self.rect_shape1, self.position1)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect1, width=2)
        text_rendered = self.main_font.render(self.value1, True, pygame.Color((0, 0, 0)))
        text_rect = text_rendered.get_rect(
            center=(self.x + self.w // 2, (self.y + self.shifter) + self.h // 2))
        self.surface.blit(text_rendered, text_rect)

        colliding_current_rect2 = current_rect2.collidepoint(mpos) and not self.frozen
        self.rect_shape2.fill(self.highlight_color if colliding_current_rect2 else self.color)

        self.surface.blit(self.rect_shape2, self.position2)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect2, width=2)
        text_rendered = self.main_font.render(self.value2, True, pygame.Color((0, 0, 0)))
        text_rect = text_rendered.get_rect(
            center=(self.x + self.w // 2, (self.y + 2 * self.shifter) + self.h // 2))
        self.surface.blit(text_rendered, text_rect)

        if self.currently_editing and not self.frozen:
            if time.time() % 1 > 0.5:
                text_rendered = self.main_font.render(self.value, True, pygame.Color((0, 0, 0)))
                text_rect = text_rendered.get_rect(center=(self.x + self.w // 2, self.y + self.h // 2))
                self.cursor.midleft = text_rect.midright
                pygame.draw.rect(self.surface, (0, 0, 0), self.cursor)
        if self.currently_editing1 and not self.frozen:
            if time.time() % 1 > 0.5:
                text_rendered = self.main_font.render(self.value1, True, pygame.Color((0, 0, 0)))
                text_rect = text_rendered.get_rect(center=(self.x + self.w // 2, (self.y + self.shifter) + self.h // 2))
                self.cursor.midleft = text_rect.midright
                pygame.draw.rect(self.surface, (0, 0, 0), self.cursor)
        if self.currently_editing2 and not self.frozen:
            if time.time() % 1 > 0.5:
                text_rendered = self.main_font.render(self.value2, True, pygame.Color((0, 0, 0)))
                text_rect = text_rendered.get_rect(
                    center=(self.x + self.w // 2, (self.y + 2 * self.shifter) + self.h // 2))
                self.cursor.midleft = text_rect.midright
                pygame.draw.rect(self.surface, (0, 0, 0), self.cursor)

        # starting the option stuff
        self.rect_shape_.fill((255, 255, 255))
        self.surface.blit(self.rect_shape_, self.position_)
        self.rect_shape_.fill(self.highlight_color if self.menu_active_ else self.color)
        self.surface.blit(self.rect_shape_, self.position_)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect_, width=2)
        text = self.option_list[self.selected]
        if len(text) > self.max_len:
            text = text[:self.max_len] + '..'
        else:
            text = text
        msg = self.font.render(text, 1, (0, 0, 0))
        x, y = self.rect_.center
        self.surface.blit(msg, msg.get_rect(center=(x - 5, y)))

        pygame.draw.polygon(self.surface, (0, 0, 0, 128), ((self.rect_.right - 12 // 2, self.rect_.bottom - 5),
                                                           (self.rect_.right - 7 // 2, self.rect_.top + 10),
                                                           (self.rect_.right - 17 // 2, self.rect_.top + 10)))

        # starting the option stuff for option 2
        self.rect_shape_1.fill((255, 255, 255))
        self.surface.blit(self.rect_shape_1, self.position_1)
        self.rect_shape_1.fill(self.highlight_color if self.menu_active_1 else self.color)
        self.surface.blit(self.rect_shape_1, self.position_1)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect_1, width=2)
        text = self.option_list[self.selected_1]
        if len(text) > self.max_len:
            text = text[:self.max_len] + '..'
        else:
            text = text
        msg = self.font.render(text, 1, (0, 0, 0))
        x, y = self.rect_1.center
        self.surface.blit(msg, msg.get_rect(center=(x - 5, y)))

        pygame.draw.polygon(self.surface, (0, 0, 0, 128), ((self.rect_1.right - 12 // 2, self.rect_1.bottom - 5),
                                                           (self.rect_1.right - 7 // 2, self.rect_1.top + 10),
                                                           (self.rect_1.right - 17 // 2, self.rect_1.top + 10)))

        self.rect_shape_2.fill((255, 255, 255))
        self.surface.blit(self.rect_shape_2, self.position_2)
        self.rect_shape_2.fill(self.highlight_color if self.menu_active_2 else self.color)
        self.surface.blit(self.rect_shape_2, self.position_2)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect_2, width=2)
        text = self.option_list[self.selected_2]
        if len(text) > self.max_len:
            text = text[:self.max_len] + '..'
        else:
            text = text
        msg = self.font.render(text, 1, (0, 0, 0))
        x, y = self.rect_2.center
        self.surface.blit(msg, msg.get_rect(center=(x - 5, y)))

        pygame.draw.polygon(self.surface, (0, 0, 0, 128), ((self.rect_2.right - 12 // 2, self.rect_2.bottom - 5),
                                                           (self.rect_2.right - 7 // 2, self.rect_2.top + 10),
                                                           (self.rect_2.right - 17 // 2, self.rect_2.top + 10)))

        if self.draw_menu:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)

            if num_visible_options < max_len:
                lower_idx = min(self.scroll_y, max_len - num_visible_options)
                upper_idx = min(self.scroll_y + num_visible_options, max_len)
            else:
                lower_idx = 0
                upper_idx = max_len

            if num_visible_options < max_len:
                rect = self.rect_.copy()
                w, h = rect.size
                w += 45
                new_rect = pygame.Rect(rect.x, rect.y, w, h)

                # we want a little progress bar on the right
                bar_height = 10
                bar_width = bar_height // 2

                total_y_len = self.rect_.height * (num_visible_options) - bar_height
                assert 0 <= self.scroll_y <= max_len - num_visible_options
                percentage_done = self.scroll_y / (max_len - num_visible_options)
                bar_y = self.rect_.y + h + total_y_len * percentage_done
                bar_rect = pygame.Rect(new_rect.right - bar_width, bar_y, bar_width, bar_height)

            for i, item_idx in enumerate(range(lower_idx, upper_idx)):
                text = self.option_list[item_idx]
                rect = self.rect_.copy()
                rect.y += (i + 1) * self.rect_.height
                w, h = rect.size
                w += 45
                new_rect_size = (w, h)
                new_rect = pygame.Rect(rect.x, rect.y, w, h)
                if self.transparent:
                    rect_shape = pygame.Surface(new_rect_size, pygame.SRCALPHA)
                else:
                    rect_shape = pygame.Surface(new_rect_size)

                rect_shape.fill((255, 255, 255, 255))
                self.surface.blit(rect_shape, (rect.x, rect.y))
                rect_shape.fill(self.highlight_color if item_idx == self.active_option else self.color)
                self.surface.blit(rect_shape, (rect.x, rect.y))
                pygame.draw.rect(self.surface, (0, 0, 0, 128), new_rect, width=2)

                msg = self.font.render(text, 1, (0, 0, 0))
                self.surface.blit(msg, msg.get_rect(center=new_rect.center))

            if num_visible_options < max_len:
                rect_shape = pygame.Surface(bar_rect.size)
                rect_shape.fill((0, 0, 0))
                self.surface.blit(rect_shape, (bar_rect.x, bar_rect.y))
            # pygame.draw.rect(self.surface, (0, 0, 0, 128), bar_rect, width=2)

        else:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)
            self.scroll_y = min(self.scroll_y, max_len - num_visible_options)

        if self.draw_menu_1:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)

            if num_visible_options < max_len:
                lower_idx = min(self.scroll_y_1, max_len - num_visible_options)
                upper_idx = min(self.scroll_y_1 + num_visible_options, max_len)
            else:
                lower_idx = 0
                upper_idx = max_len

            if num_visible_options < max_len:
                rect = self.rect_1.copy()
                w, h = rect.size
                w += 45
                new_rect = pygame.Rect(rect.x, rect.y, w, h)

                # we want a little progress bar on the right
                bar_height = 10
                bar_width = bar_height // 2

                total_y_len = self.rect_1.height * (num_visible_options) - bar_height
                assert 0 <= self.scroll_y_1 <= max_len - num_visible_options
                percentage_done = self.scroll_y_1 / (max_len - num_visible_options)
                bar_y = self.rect_1.y + h + total_y_len * percentage_done
                bar_rect = pygame.Rect(new_rect.right - bar_width, bar_y, bar_width, bar_height)

            for i, item_idx in enumerate(range(lower_idx, upper_idx)):
                text = self.option_list[item_idx]
                rect = self.rect_1.copy()
                rect.y += (i + 1) * self.rect_1.height
                w, h = rect.size
                w += 45
                new_rect_size = (w, h)
                new_rect = pygame.Rect(rect.x, rect.y, w, h)
                if self.transparent:
                    rect_shape = pygame.Surface(new_rect_size, pygame.SRCALPHA)
                else:
                    rect_shape = pygame.Surface(new_rect_size)

                rect_shape.fill((255, 255, 255, 255))
                self.surface.blit(rect_shape, (rect.x, rect.y))
                rect_shape.fill(self.highlight_color if item_idx == self.active_option_1 else self.color)
                self.surface.blit(rect_shape, (rect.x, rect.y))
                pygame.draw.rect(self.surface, (0, 0, 0, 128), new_rect, width=2)

                msg = self.font.render(text, 1, (0, 0, 0))
                self.surface.blit(msg, msg.get_rect(center=new_rect.center))

            if num_visible_options < max_len:
                rect_shape = pygame.Surface(bar_rect.size)
                rect_shape.fill((0, 0, 0))
                self.surface.blit(rect_shape, (bar_rect.x, bar_rect.y))
            # pygame.draw.rect(self.surface, (0, 0, 0, 128), bar_rect, width=2)

        else:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)
            self.scroll_y_1 = min(self.scroll_y_1, max_len - num_visible_options)

        if self.draw_menu_2:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)

            if num_visible_options < max_len:
                lower_idx = min(self.scroll_y_2, max_len - num_visible_options)
                upper_idx = min(self.scroll_y_2 + num_visible_options, max_len)
            else:
                lower_idx = 0
                upper_idx = max_len

            if num_visible_options < max_len:
                rect = self.rect_2.copy()
                w, h = rect.size
                w += 45
                new_rect = pygame.Rect(rect.x, rect.y, w, h)

                # we want a little progress bar on the right
                bar_height = 10
                bar_width = bar_height // 2

                total_y_len = self.rect_2.height * (num_visible_options) - bar_height
                assert 0 <= self.scroll_y_2 <= max_len - num_visible_options
                percentage_done = self.scroll_y_2 / (max_len - num_visible_options)
                bar_y = self.rect_2.y + h + total_y_len * percentage_done
                bar_rect = pygame.Rect(new_rect.right - bar_width, bar_y, bar_width, bar_height)

            for i, item_idx in enumerate(range(lower_idx, upper_idx)):
                text = self.option_list[item_idx]
                rect = self.rect_2.copy()
                rect.y += (i + 1) * self.rect_2.height
                w, h = rect.size
                w += 45
                new_rect_size = (w, h)
                new_rect = pygame.Rect(rect.x, rect.y, w, h)
                if self.transparent:
                    rect_shape = pygame.Surface(new_rect_size, pygame.SRCALPHA)
                else:
                    rect_shape = pygame.Surface(new_rect_size)

                rect_shape.fill((255, 255, 255, 255))
                self.surface.blit(rect_shape, (rect.x, rect.y))
                rect_shape.fill(self.highlight_color if item_idx == self.active_option_2 else self.color)
                self.surface.blit(rect_shape, (rect.x, rect.y))
                pygame.draw.rect(self.surface, (0, 0, 0, 128), new_rect, width=2)

                msg = self.font.render(text, 1, (0, 0, 0))
                self.surface.blit(msg, msg.get_rect(center=new_rect.center))

            if num_visible_options < max_len:
                rect_shape = pygame.Surface(bar_rect.size)
                rect_shape.fill((0, 0, 0))
                self.surface.blit(rect_shape, (bar_rect.x, bar_rect.y))
            # pygame.draw.rect(self.surface, (0, 0, 0, 128), bar_rect, width=2)

        else:
            num_visible_options = self.num_visible_options
            max_len = len(self.option_list)
            self.scroll_y_2 = min(self.scroll_y_2, max_len - num_visible_options)

    def process_event(self, event):
        if self.frozen:
            return

        x = int(self.x * self.settings.zoom) - self.settings.offset_x
        y = int(self.y * self.settings.zoom) - self.settings.offset_y
        current_w = int(self.w * self.settings.zoom)
        current_h = int(self.h * self.settings.zoom)
        current_rect = pygame.Rect(x, y, current_w, current_h)
        current_rect1 = pygame.Rect(x, y + self.shifter, current_w, current_h)
        current_rect2 = pygame.Rect(x, y + 2 * self.shifter, current_w, current_h)

        mpos = pygame.mouse.get_pos()
        self.menu_active = current_rect.collidepoint(mpos)

        mpos = pygame.mouse.get_pos()
        self.menu_active2 = current_rect1.collidepoint(mpos)

        mpos = pygame.mouse.get_pos()
        self.menu_active3 = current_rect2.collidepoint(mpos)

        if self.menu_active:
            if event.type == pygame.MOUSEBUTTONUP:
                self.currently_editing = True

        if self.menu_active2:
            if event.type == pygame.MOUSEBUTTONUP:
                self.currently_editing1 = True

        if self.menu_active3:
            if event.type == pygame.MOUSEBUTTONUP:
                self.currently_editing2 = True

        if self.currently_editing:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    if len(self.value) > 2:
                        self.value = self.value[:-1]
                elif event.key == pygame.K_RETURN and len(self.value) > 0:
                    self.currently_editing = False
                elif '%' in self.value:
                    self.currently_editing = False
                elif event.type == pygame.QUIT:
                    return False
                else:
                    if len(self.value) < 4:
                        self.value = self.value + event.unicode
        if self.currently_editing1:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    # make sure we don't delete 0.
                    if len(self.value1) > 2:
                        self.value1 = self.value1[:-1]
                elif event.key == pygame.K_RETURN and len(self.value1) > 0:
                    self.currently_editing1 = False
                elif '%' in self.value1:
                    self.currently_editing1 = False
                elif event.type == pygame.QUIT:
                    return False
                else:
                    # max length example: 0.02
                    if len(self.value1) < 4:
                        self.value1 = self.value1 + event.unicode

        if self.currently_editing2:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    if len(self.value2) > 2:
                        self.value2 = self.value2[:-1]
                elif event.key == pygame.K_RETURN and len(self.value2) > 0:
                    self.currently_editing2 = False
                elif '%' in self.value2:
                    self.currently_editing2 = False
                elif event.type == pygame.QUIT:
                    return False
                else:
                    if len(self.value2) < 4:
                        self.value2 = self.value2 + event.unicode

        current_rect_ = pygame.Rect(x, y + self.shifter / 2, current_w + 45, current_h)
        current_rect_1 = pygame.Rect(x, y + 3 * self.shifter / 2, current_w + 45, current_h)
        current_rect_2 = pygame.Rect(x, y + 5 * self.shifter / 2, current_w + 45, current_h)
        self.menu_active_ = current_rect_.collidepoint(mpos)
        self.menu_active_1 = current_rect_1.collidepoint(mpos)
        self.menu_active_2 = current_rect_2.collidepoint(mpos)

        num_visible_options = self.num_visible_options
        max_len = len(self.option_list)
        if num_visible_options < max_len:
            lower_idx = min(self.scroll_y, max_len - num_visible_options)
            upper_idx = min(self.scroll_y + num_visible_options, max_len)
        else:
            lower_idx = 0
            upper_idx = max_len

        for i, item_idx in enumerate(range(lower_idx, upper_idx)):
            rect = current_rect_.copy()
            rect.y += (i + 1) * current_rect_.height
            if rect.collidepoint(mpos):
                self.active_option = item_idx
                break

        for i, item_idx in enumerate(range(lower_idx, upper_idx)):
            rect = current_rect_1.copy()
            rect.y += (i + 1) * current_rect_1.height
            if rect.collidepoint(mpos):
                self.active_option_1 = item_idx
                break

        for i, item_idx in enumerate(range(lower_idx, upper_idx)):
            rect = current_rect_2.copy()
            rect.y += (i + 1) * current_rect_2.height
            if rect.collidepoint(mpos):
                self.active_option_2 = item_idx
                break
        # if not self.menu_active and self.active_option == -1:
        #     self.draw_menu = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.draw_menu_1 and not self.draw_menu_2:
            if self.menu_active_:
                self.draw_menu = not self.draw_menu
            elif self.draw_menu and self.active_option >= 0:
                self.previously_selected = self.selected
                self.selected = self.active_option
                self.draw_menu = False
                return self.active_option

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.draw_menu and not self.draw_menu_2:
            if self.menu_active_1:
                self.draw_menu_1 = not self.draw_menu_1
            elif self.draw_menu_1 and self.active_option_1 >= 0:
                self.previously_selected_1 = self.selected_1
                self.selected_1 = self.active_option_1
                self.draw_menu_1 = False
                return self.active_option_1

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.draw_menu and not self.draw_menu_1:
            if self.menu_active_2:
                self.draw_menu_2 = not self.draw_menu_2
            elif self.draw_menu_2 and self.active_option_2 >= 0:
                self.previously_selected_2 = self.selected_2
                self.selected_2 = self.active_option_2
                self.draw_menu_2 = False
                return self.active_option_2

        #if self.lock_menu:
        #    self.settings.options_menus_per_domain[self.domain_idx][self.gui_node_idx] = self.draw_menu

    def process_standby(self):
        self.show()


class GUITreeNode(GUIItem):
    def __init__(self, surface: pygame.Surface, position: tuple, size: tuple,
                 font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.position = position
        self.pos_x, self.pos_y = self.position
        self.size = size
        self.surface = surface
        self.size_x, self.size_y = self.size
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        self.secondary_font = pygame.font.Font('freesansbold.ttf', font_size - 4)
        self.rect_color = rect_color
        self.border_color = border_color
        self.border_width = border_width
        self.transparent = transparent
        self.text_color = text_color

        if self.rect_color is None:
            self.rect_color = (255, 255, 255, 255)
        if self.border_color is None:
            self.border_color = (0, 0, 0, 255)

        if self.__class__.__name__ == 'GUIActionNodeDT':
            self.rectangle = pygame.Rect((self.pos_x, self.pos_y, self.size_x, self.size_y*3))
        else:
            self.rectangle = pygame.Rect((self.pos_x, self.pos_y, self.size_x, self.size_y))


        if self.transparent:
            self.rect_shape = pygame.Surface(self.rectangle.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rectangle.size)

        self.child_elements = []

    def show(self):
        self.rect_shape.fill(self.rect_color)
        self.surface.blit(self.rect_shape, self.position)
        if self.border_width > 0:
            pygame.draw.rect(self.surface, self.border_color, self.rectangle, width=self.border_width)

    def show_children(self):
        for child in self.child_elements:
            child.show()

    def process_event(self, event):
        for child in self.child_elements:
            child.process_event(event)
        return True


class Arrow(GUIItem):
    def __init__(self, surface: pygame.Surface, start: pygame.Vector2, end: pygame.Vector2, color=pygame.Color('black'),
                 body_width: int = 5, head_width: int = 10, head_height: int = 10, text=None, text_left=True):
        self.surface = surface
        self.start = start
        self.end = end
        self.color = color
        self.body_width = body_width
        self.head_width = head_width
        self.head_height = head_height
        self.text = text
        self.main_font = pygame.font.Font('freesansbold.ttf', 18)

        arrow = self.start - self.end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - self.head_height

        self.head_vertices = [pygame.Vector2(0, self.head_height / 2),
                              pygame.Vector2(self.head_width / 2, -self.head_height / 2),
                              pygame.Vector2(-self.head_width / 2, -self.head_height / 2)]
        translation = pygame.Vector2(0, arrow.length() - (self.head_height / 2)).rotate(-angle)
        for i in range(len(self.head_vertices)):
            self.head_vertices[i].rotate_ip(-angle)
            self.head_vertices[i] += translation
            self.head_vertices[i] += self.start

        if arrow.length() >= self.head_height:
            self.body_verts = [pygame.Vector2(-self.body_width / 2, body_length / 2),
                               pygame.Vector2(self.body_width / 2, body_length / 2),
                               pygame.Vector2(self.body_width / 2, -body_length / 2),
                               pygame.Vector2(-self.body_width / 2, -body_length / 2)]
            translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
            for i in range(len(self.body_verts)):
                self.body_verts[i].rotate_ip(-angle)
                self.body_verts[i] += translation
                self.body_verts[i] += self.start

        # in the middle of the arrow, add text to the left
        if self.text is not None:
            # depending on whether text is left or right, we need to adjust the position
            if text_left:
                self.text_pos = (self.start + self.end) / 2
                self.text_pos.x -= 25
                self.text_pos.y -= 10
                self.text_surface = self.main_font.render(self.text, True, self.color)
                self.text_rect = self.text_surface.get_rect(center=self.text_pos)
            else:
                self.text_pos = (self.start + self.end) / 2
                self.text_pos.x += 25
                self.text_pos.y -= 10
                self.text_surface = self.main_font.render(self.text, True, self.color)
                self.text_rect = self.text_surface.get_rect(center=self.text_pos)

    def process_event(self, event):
        return 'continue', None

    def show(self):
        pygame.draw.polygon(self.surface, self.color, self.head_vertices)
        pygame.draw.polygon(self.surface, self.color, self.body_verts)
        if self.text is not None:
            self.surface.blit(self.text_surface, self.text_rect)


class Multiplier(GUIItem):
    def __init__(self, env_wrapper, multiplier_idx, surface: pygame.Surface, settings, position: tuple):
        self.env_wrapper = env_wrapper
        self.multiplier_idx = multiplier_idx
        self.settings = settings

        option_color = (137, 207, 240, 128)
        option_highlight_color = (137, 207, 240, 255)
        x, y = position
        node_options_h = 24
        node_options_w = 160
        choices = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0']
        self.choices = choices
        self.child_elements = []

        # option_list, lock_menu=False, selected=-1, transparent=True, max_len=20, num_visible_options=20):

        self.node_box = OptionBox(surface=surface,
                                  x=x + 200, y=y,
                                  w=node_options_w, h=node_options_h,
                                  settings=self.settings,
                                  domain_idx=1,
                                  gui_node_idx=1,
                                  color=option_color,
                                  highlight_color=option_highlight_color,
                                  font=pygame.font.SysFont(None, 30),
                                  option_list=self.choices,
                                  selected=self.choices.index(str(float(self.env_wrapper.multipliers[multiplier_idx]))))
        self.child_elements.append(self.node_box)

    def show(self):
        pass

    def process_event(self, event):
        for item in self.child_elements:
            if item.selected != item.previously_selected:
                self.env_wrapper.multipliers[self.multiplier_idx] = float(self.choices[item.selected])
                self.env_wrapper.initialize_env()
                item.previously_selected = item.selected
        for child in self.child_elements:
            child.process_event(event)
        return 'continue', None

    def show_children(self):
        for child in self.child_elements:
            child.show()


class GUIDecisionNode(GUITreeNode):
    def __init__(self, icct, node_idx: int, env_feat_names: [], surface: pygame.Surface, settings, position: tuple,
                 size: tuple,
                 font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                 variable_idx: int = -1, compare_sign='<', comparator_value='1.0',
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.icct = icct
        self.node_idx = node_idx
        self.env_feat_names = env_feat_names
        self.settings = settings
        super(GUIDecisionNode, self).__init__(surface=surface, position=position,
                                              size=size, font_size=font_size,
                                              text_color=text_color, transparent=transparent,
                                              rect_color=rect_color, border_color=border_color,
                                              border_width=border_width)

        option_color = (137, 207, 240, 128)
        option_highlight_color = (137, 207, 240, 255)

        x, y = position

        node_options_h = 35
        node_options_w = 180
        node_options_y = 10 + y
        node_options_x = self.pos_x + self.size_x // 2 - node_options_w // 2

        # below assumes that root node will be idx 0
        if node_idx != 0:
            choices = ['Decision Node', 'Action Node']
        else:
            choices = ['Decision Node']

        variable_options_h = 35
        variable_options_w = 190
        variable_options_y = 10 + node_options_y + node_options_h
        variable_options_x = 10 + x

        self.variables_box = OptionBox(surface,
                                       variable_options_x, variable_options_y,
                                       variable_options_w, variable_options_h,
                                       self.settings,
                                       option_color,
                                       option_highlight_color,
                                       pygame.font.SysFont(None, 30),
                                       env_feat_names,
                                       selected=variable_idx)
        self.child_elements.append(self.variables_box)

        sign_options_h = 35
        sign_options_w = 60
        sign_options_y = 10 + node_options_y + node_options_h
        sign_options_x = 10 + variable_options_x + variable_options_w

        signs = ['<', '>']

        self.sign_box = OptionBox(surface,
                                  sign_options_x, sign_options_y,
                                  sign_options_w, sign_options_h,
                                  self.settings,
                                  option_color,
                                  option_highlight_color,
                                  pygame.font.SysFont(None, 30), signs,
                                  selected=signs.index(compare_sign))
        self.child_elements.append(self.sign_box)

        compare_options_h = 35
        compare_options_w = 70
        compare_options_y = 10 + node_options_y + node_options_h
        compare_options_x = 10 + sign_options_x + sign_options_w

        self.comparator_box = TextBox(surface,
                                      settings,
                                      compare_options_x,
                                      compare_options_y,
                                      compare_options_w,
                                      compare_options_h,
                                      option_color,
                                      option_highlight_color,
                                      pygame.font.Font('freesansbold.ttf', 20),
                                      value=comparator_value)
        self.child_elements.append(self.comparator_box)

        self.node_box = OptionBox(surface,
                                  node_options_x, node_options_y,
                                  node_options_w, node_options_h,
                                  self.settings,
                                  option_color,
                                  option_highlight_color,
                                  pygame.font.SysFont(None, 30),
                                  choices,
                                  selected=0)
        self.child_elements.append(self.node_box)

    def process_event(self, event):
        super(GUIDecisionNode, self).process_event(event)
        if self.variables_box.selected != self.variables_box.previously_selected:
            with torch.no_grad():
                weights = torch.abs(self.icct.layers.cpu())
                max_weight = torch.max(weights[self.node_idx])
                for i in range(len(self.icct.layers[self.node_idx])):
                    if i != self.variables_box.selected:
                        self.icct.layers[self.node_idx, i] = 1
                    else:
                        self.icct.layers[self.node_idx, i] = 2

                # think we need to update comparators here

                print('New var value!')
            self.variables_box.previously_selected = self.variables_box.selected
        if self.sign_box.selected != self.sign_box.previously_selected:
            with torch.no_grad():
                is_greater_than = (self.icct.alpha.cpu() * self.icct.layers.cpu() > 0)[
                    self.node_idx, self.variables_box.selected]
                sign_for_new_var = '>' if is_greater_than else '<'
                if self.variables_box.option_list[self.variables_box.selected] != sign_for_new_var:
                    self.icct.layers[self.node_idx, self.variables_box.selected] *= -1
                print('New comparator value!')
                self.sign_box.previously_selected = self.sign_box.selected
        if not self.comparator_box.currently_editing and \
                (self.comparator_box.value != self.comparator_box.previous_value):
            multiplier = float(self.comparator_box.value) / float(self.comparator_box.previous_value)
            with torch.no_grad():
                self.icct.layers[self.node_idx, self.variables_box.selected] /= multiplier
        if self.node_box.selected != self.node_box.previously_selected:
            if self.node_box.selected == 1:
                new_tree = convert_decision_to_leaf(self.icct, self.node_idx)
                return 'new_tree', new_tree
        return 'continue', None


class GUIActionNodeICCT(GUITreeNode):
    def __init__(self, tree, surface: pygame.Surface, settings, position: tuple, size: tuple, name: str,
                 text: str, font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.tree = tree
        super(GUIActionNodeICCT, self).__init__(surface, position, size,
                                                font_size, text_color, transparent,
                                                rect_color, border_color, border_width)

    def process_event(self, event):
        super(GUIActionNodeICCT, self).process_event(event)
        return 'continue', None


class GUIActionNodeIDCT(GUITreeNode):
    def __init__(self, tree, surface: pygame.Surface, settings, position: tuple, size: tuple,
                 leaf_idx: int, action_idx: int, actions_list: [], font_size: int = 12,
                 text_color: str = 'black', transparent: bool = True,
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0, frozen=False):
        self.tree = tree
        self.leaf_idx = leaf_idx
        self.settings = settings
        super(GUIActionNodeIDCT, self).__init__(surface, position, size,
                                                font_size, text_color, transparent,
                                                rect_color, border_color, border_width)

        option_color = (240, 128, 101, 128)
        option_highlight_color = (240, 128, 101, 255)

        x, y = position

        node_options_h = 35
        node_options_w = 160
        node_options_y = 10 + y
        node_options_x = self.pos_x + self.size_x // 2 - node_options_w // 2

        choices = ['Decision Node', 'Action Node']

        variable_options_h = 35
        variable_options_w = 160
        variable_options_y = 5 + node_options_y + node_options_h
        variable_options_x = 10 + x

        self.actions_box = OptionBox(surface,
                                     variable_options_x, variable_options_y,
                                     variable_options_w, variable_options_h,
                                     self.settings,
                                     option_color,
                                     option_highlight_color,
                                     pygame.font.SysFont(None, 30),
                                     actions_list,
                                     selected=action_idx)
        self.child_elements.append(self.actions_box)

        self.node_box = OptionBox(surface,
                                  node_options_x, node_options_y,
                                  node_options_w, node_options_h,
                                  self.settings,
                                  option_color,
                                  option_highlight_color,
                                  pygame.font.SysFont(None, 30),
                                  choices,
                                  selected=1)
        self.child_elements.append(self.node_box)

    def process_event(self, event):
        super(GUIActionNodeIDCT, self).process_event(event)
        if self.actions_box.selected != self.actions_box.previously_selected:
            with torch.no_grad():
                n_actions = len(self.tree.leaf_init_information[self.leaf_idx][2])
                for i in range(n_actions):
                    if i == self.actions_box.selected:
                        self.tree.leaf_init_information[self.leaf_idx][2][i] = 2
                        self.tree.action_mus[self.leaf_idx, i] = 2
                    else:
                        self.tree.leaf_init_information[self.leaf_idx][2][i] = -2
                        self.tree.action_mus[self.leaf_idx, i] = -2
                print('New action value!')
            self.actions_box.previously_selected = self.actions_box.selected
        if self.node_box.selected != self.node_box.previously_selected:
            if self.node_box.selected == 0:
                new_tree = convert_leaf_to_decision(self.tree, self.leaf_idx)
                return 'new_tree', new_tree
        return 'continue', None


class GUIDecisionNodeDT(GUITreeNode):
    def __init__(self, decision_tree, dt_node, env_feat_names: [], surface: pygame.Surface, settings, domain_idx: int,
                 position: tuple, size: tuple,
                 font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                 variable_idx: int = -1, compare_sign='<=',
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0, frozen=False):
        self.decision_tree = decision_tree
        self.dt_node = dt_node
        self.feat_val = dt_node.comp_val
        self.domain_idx = domain_idx
        self.frozen = frozen
        comparator_value = self.dt_node.comp_val
        self.env_feat_names = env_feat_names
        # self.feature_values = feature_values
        self.settings = settings
        self.gui_node_idx = len(self.settings.options_menus_per_domain[self.domain_idx])
        self.settings.options_menus_per_domain[self.domain_idx].append(False)

        super(GUIDecisionNodeDT, self).__init__(surface=surface, position=position,
                                                size=size, font_size=font_size,
                                                text_color=text_color, transparent=transparent,
                                                rect_color=rect_color, border_color=border_color,
                                                border_width=border_width)

        option_color = (137, 207, 240, 128)
        option_highlight_color = (137, 207, 240, 255)

        x, y = position

        node_options_h = 40 // 2
        node_options_w = 300 // 2
        node_options_y = (10 + y)
        node_options_x = (self.pos_x + self.size_x // 2 - node_options_w // 2)

        if not self.dt_node.is_root:
            choices = ['Decision Node', 'Action Node']
        else:
            choices = ['Decision Node']

        variable_options_h = 40 // 2
        variable_options_w = 430 // 2
        variable_options_y = 15 // 2 + node_options_y + node_options_h
        variable_options_x = (self.pos_x + self.size_x // 2 - variable_options_w // 2)

        self.variables_box = OptionBox(surface,
                                       variable_options_x, variable_options_y,
                                       variable_options_w, variable_options_h,
                                       self.settings,
                                       domain_idx,
                                       self.gui_node_idx,
                                       option_color,
                                       option_highlight_color,
                                       pygame.font.SysFont(None, 24),
                                       env_feat_names,
                                       max_len=30,
                                       lock_menu=True,
                                       selected=variable_idx)
        self.child_elements.append(self.variables_box)

        sign_options_h = 35 // 2
        sign_options_w = 60 // 2
        sign_options_y = 10 // 2 + node_options_y + node_options_h
        sign_options_x = 10 // 2 + variable_options_x + variable_options_w

        signs = ['<=', '>']

        # self.sign_box = OptionBox(surface,
        #                           sign_options_x, sign_options_y,
        #                           sign_options_w, sign_options_h,
        #                           self.settings,
        #                           option_color,
        #                           option_highlight_color,
        #                           pygame.font.SysFont(None, 15), signs,
        #                           selected=signs.index(compare_sign))
        # self.child_elements.append(self.sign_box)

        compare_options_h = 35 // 2
        compare_options_w = 70 // 2
        compare_options_y = 10 // 2 + node_options_y + node_options_h
        compare_options_x = 10 // 2 + sign_options_x + sign_options_w

        # self.comparator_box = TextBox(surface,
        #                               settings,
        #                               compare_options_x,
        #                               compare_options_y,
        #                               compare_options_w,
        #                               compare_options_h,
        #                               option_color,
        #                               option_highlight_color,
        #                               pygame.font.Font('freesansbold.ttf', 10),
        #                               value=str(comparator_value))
        # self.child_elements.append(self.comparator_box)

        self.node_box = OptionBox(surface,
                                  node_options_x, node_options_y,
                                  node_options_w, node_options_h,
                                  self.settings,
                                  domain_idx,
                                  self.gui_node_idx,
                                  option_color,
                                  option_highlight_color,
                                  pygame.font.SysFont(None, 24),
                                  choices,
                                  max_len=20,
                                  selected=0)
        self.child_elements.append(self.node_box)

    def process_event(self, event):
        if self.frozen:
            return 'continue', None
        node_showing, node_idx = self.settings.check_if_options_menu_open(domain_idx=self.domain_idx)
        if node_showing and node_idx != self.gui_node_idx:
            return 'continue', None
        super(GUIDecisionNodeDT, self).process_event(event)
        if self.variables_box.selected != self.variables_box.previously_selected:
            self.dt_node.var_idx = self.variables_box.selected
            # also needs to change the possible values for this feature
            self.variables_box.previously_selected = self.variables_box.selected
            return 'new_tree', None
        # if self.sign_box.selected != self.sign_box.previously_selected:
        #     self.dt_node.normal_ordering = self.sign_box.selected
        #     self.sign_box.previously_selected = self.sign_box.selected
        # if not self.comparator_box.currently_editing and \
        #     (self.comparator_box.value != self.comparator_box.previous_value):
        #     self.dt_node.comp_val = float(self.comparator_box.value)
        if self.node_box.selected != self.node_box.previously_selected:
            if self.node_box.selected == 1:
                self.decision_tree.convert_dt_decision_to_leaf(self.dt_node)
                return 'new_tree', None
        return 'continue', None

    def show(self):
        # check to see if we any other node is showing
        node_showing, node_idx = self.settings.check_if_options_menu_open(domain_idx=self.domain_idx)
        if node_showing and node_idx != self.gui_node_idx:
            # then we don't update
            pass
        else:
            super(GUIDecisionNodeDT, self).show()

    def show_children(self):
        # check to see if we any other node is showing
        node_showing, node_idx = self.settings.check_if_options_menu_open(domain_idx=self.domain_idx)
        if node_showing and node_idx != self.gui_node_idx:
            # then we don't update
            pass
        else:
            super(GUIDecisionNodeDT, self).show_children()


class GUIActionNodeDT(GUITreeNode):
    def __init__(self, decision_tree, dt_node, surface: pygame.Surface, settings, domain_idx: int, position: tuple,
                 size: tuple,
                 leaf_idx: int, action_idx: int, actions_list: [], font_size: int = 12,
                 text_color: str = 'black', transparent: bool = True,
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0, action_prob=None,
                 first_one=False, frozen=False):
        self.decision_tree = decision_tree
        self.dt_node = dt_node
        self.leaf_idx = leaf_idx
        self.domain_idx = domain_idx
        self.settings = settings
        self.frozen = frozen
        self.gui_node_idx = len(self.settings.options_menus_per_domain[self.domain_idx])
        self.settings.options_menus_per_domain[self.domain_idx].append(False)
        super(GUIActionNodeDT, self).__init__(surface, position, size,
                                              font_size, text_color, transparent,
                                              rect_color, border_color, border_width)

        option_color = (240, 128, 101, 128)
        option_highlight_color = (240, 128, 101, 255)

        x, y = position

        node_options_h = 45 // 2
        node_options_w = 200 // 2
        node_options_y = 5 + y
        node_options_x = self.pos_x + self.size_x // 2 - node_options_w // 2

        max_depth = 3
        if dt_node.depth >= max_depth:
            choices = ['Action Node']
        else:
            choices = ['Action Node', 'Decision Node']

        variable_options_h = 45 // 2
        variable_options_w = 220 // 2
        variable_options_y = 5 + node_options_y + node_options_h
        variable_options_x = self.pos_x + self.size_x // 2 - variable_options_w // 2

        # self.actions_box = OptionBox(surface,
        #                              variable_options_x, variable_options_y,
        #                              variable_options_w, variable_options_h,
        #                              self.settings,
        #                              domain_idx,
        #                              self.gui_node_idx,
        #                              option_color,
        #                              option_highlight_color,
        #                              pygame.font.SysFont(None, 18),
        #                              actions_list,
        #                              lock_menu=True,
        #                              selected=action_idx,
        #                              max_len=12)
        # self.child_elements.append(self.actions_box)

        use_stochastic_leaves = True
        if use_stochastic_leaves:
            # get top k actions
            # then we want to instantiate a TextBox for each actiom
            # probability distribution
            self.actions_box = TextObjectBox(surface,
                                             variable_options_x, variable_options_y,
                                             variable_options_w, variable_options_h,
                                             self.settings,
                                             domain_idx,
                                             self.gui_node_idx,
                                             option_color,
                                             option_highlight_color,
                                             pygame.font.SysFont(None, 15),
                                             actions_list,
                                             lock_menu=True,
                                             selected=action_idx,
                                             max_len=17,
                                             value=action_prob,
                                             frozen=self.frozen)
            self.child_elements.append(self.actions_box)
            self.actions_box.value_tracker = [float(self.actions_box.value), float(self.actions_box.value1),
                                              float(self.actions_box.value2)]
            # self.node_box = TextBox(surface,
            #                         settings,
            #                         node_options_x, node_options_y,
            #                         node_options_w, node_options_h,
            #                         option_color,
            #                         option_highlight_color,
            #                         pygame.font.Font('freesansbold.ttf', 20),
            #                         value=str(action_prob))
            # self.child_elements.append(self.node_box)
            self.node_box = OptionBox(surface,
                                      node_options_x, node_options_y,
                                      node_options_w, node_options_h,
                                      self.settings,
                                      domain_idx,
                                      self.gui_node_idx,
                                      option_color,
                                      option_highlight_color,
                                      pygame.font.SysFont(None, 18),
                                      choices,
                                      selected=0,
                                      frozen=self.frozen)
            self.child_elements.append(self.node_box)

            self.drop_down_only = False


        else:
            self.actions_box = OptionBox(surface,
                                         variable_options_x, variable_options_y,
                                         variable_options_w, variable_options_h,
                                         self.settings,
                                         domain_idx,
                                         self.gui_node_idx,
                                         option_color,
                                         option_highlight_color,
                                         pygame.font.SysFont(None, 18),
                                         actions_list,
                                         lock_menu=True,
                                         selected=action_idx,
                                         max_len=12)
            self.child_elements.append(self.actions_box)
            self.node_box = OptionBox(surface,
                                      node_options_x, node_options_y,
                                      node_options_w, node_options_h,
                                      self.settings,
                                      domain_idx,
                                      self.gui_node_idx,
                                      option_color,
                                      option_highlight_color,
                                      pygame.font.SysFont(None, 18),
                                      choices,
                                      selected=0)
            self.child_elements.append(self.node_box)
            self.drop_down_only = False

    def process_event(self, event):
        if self.frozen:
            return 'continue', None
        super(GUIActionNodeDT, self).process_event(event)
        if self.actions_box.selected != self.actions_box.previously_selected:
            self.dt_node.action.indices[0] = self.actions_box.selected
            self.actions_box.previously_selected = self.actions_box.selected
            return 'new_tree', None
        if self.actions_box.selected_1 != self.actions_box.previously_selected_1:
            self.dt_node.action.indices[1] = self.actions_box.selected_1
            self.actions_box.previously_selected_1 = self.actions_box.selected
            return 'new_tree', None
        if self.actions_box.selected_2 != self.actions_box.previously_selected_2:
            self.dt_node.action.indices[2] = self.actions_box.selected_2
            self.actions_box.previously_selected_2 = self.actions_box.selected_2
            return 'new_tree', None


        import numpy as np
        print(self.actions_box.position, self.actions_box.value, self.actions_box.value1, self.actions_box.value2)
        if np.sum([float(self.actions_box.value), float(self.actions_box.value1),
                   float(self.actions_box.value2)]) == 1 and [float(self.actions_box.value),
                                                              float(self.actions_box.value1),
                                                              float(
                                                                  self.actions_box.value2)] != self.actions_box.value_tracker and not self.actions_box.currently_editing and not self.actions_box.currently_editing1 and not self.actions_box.currently_editing2:
            self.actions_box.value_tracker = [float(self.actions_box.value), float(self.actions_box.value1),
                                              float(self.actions_box.value2)]
            self.dt_node.action.values = [float(self.actions_box.value), float(self.actions_box.value1),
                                                          float(self.actions_box.value2)]
            return 'new_tree', None
        if self.node_box.selected != self.node_box.previously_selected:
            self.decision_tree.convert_dt_leaf_to_decision(self.dt_node)
            return 'new_tree', None
        return 'continue', None

    def show(self):
        # check to see if we any other node is showing
        node_showing, node_idx = self.settings.check_if_options_menu_open(domain_idx=self.domain_idx)
        if node_showing and node_idx != self.gui_node_idx:
            # then we don't update
            pass
        else:
            super(GUIActionNodeDT, self).show()

    def show_children(self):
        # check to see if we any other node is showing
        node_showing, node_idx = self.settings.check_if_options_menu_open(domain_idx=self.domain_idx)
        if node_showing and node_idx != self.gui_node_idx:
            # then we don't update
            pass
        else:
            for child in self.child_elements:
                # if self.drop_down_only:
                #     if type(child)
                child.show()
