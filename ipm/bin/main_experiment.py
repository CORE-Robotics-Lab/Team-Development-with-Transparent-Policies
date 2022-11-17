import math
import os
import pygame
import cv2
import time
import torch
from typing import Callable


class GUIButton:
    def __init__(self, surface: pygame.Surface, position: tuple, size: tuple, event_fn: Callable,
                    text: str, font_size: int = 12, text_color: str = 'black', transparent: bool = True,
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
            self.rect_color = (255, 255, 255, 255)
        if self.border_color is None:
            self.border_color = (0, 0, 0, 255)

        self.rectangle = pygame.Rect((self.pos_x, self.pos_y, self.size_x, self.size_y))

        if self.transparent:
            self.rect_shape = pygame.Surface(self.rectangle.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rectangle.size)

        self.showing = False

    def show(self):
        self.rect_shape.fill(self.rect_color)
        self.surface.blit(self.rect_shape, self.position)
        if self.border_width > 0:
            pygame.draw.rect(self.surface, self.border_color, self.rectangle, width=self.border_width)
        text_rendered = self.font.render(self.text, True, pygame.Color(self.text_color))
        text_rect = text_rendered.get_rect(center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
        self.cursor = pygame.Rect(text_rect.topright, (3, text_rect.height + 2))
        self.surface.blit(text_rendered, text_rect)
        self.highlighting = False
        self.showing = True

    def hide(self):
        self.showing = False

    def process_event(self, event):
        mouse_position = pygame.mouse.get_pos()
        if self.rectangle.collidepoint(mouse_position):
            # Highlight on hover

            if not self.highlighting:
                highlight_color = (69, 69, 69)
                highlight_border_color = (255, 255, 0)
                highlight_text_color = (255, 255, 0)
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
                pygame.display.update()

                self.highlighting = True

            if event.type == pygame.MOUSEBUTTONUP:
                self.event_fn()
        return True

    def process_standby(self):
        mouse_position = pygame.mouse.get_pos()
        if not self.rectangle.collidepoint(mouse_position) and self.highlighting:
            self.rect_shape.fill(self.rect_color)
            self.surface.blit(self.rect_shape, self.position)
            self.rect_shape.fill(self.rect_color)
            self.surface.blit(self.rect_shape, self.position)
            if self.border_width > 0:
                pygame.draw.rect(self.surface, self.border_color, self.rectangle, width=self.border_width)
            text_rendered = self.font.render(self.text, True, self.text_color)
            text_rect = text_rendered.get_rect(
                center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
            self.surface.blit(text_rendered, text_rect)
            self.highlighting = False
            pygame.display.update()

class GUIPage:
    def __init__(self):
        pygame.init()
        self.X, self.Y = 1800, 800
        self.screen = None
        self.gui_items = []
        self.showing = False

    def show(self):
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        for item in self.gui_items:
            item.show()
        self.showing = True

    def hide(self):
        for item in self.gui_items:
            item.hide()
        self.screen.fill((0, 0, 0))
        self.showing = False

class GUIPageCenterText(GUIPage):
    def __init__(self, text, font_size, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn = None, bottom_right_fn = None):
        GUIPage.__init__(self)
        self.text = text
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        self.text_render = self.main_font.render(text, True, (255, 255, 255))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

    def show(self):
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)
        self.screen.blit(self.text_render, self.text_render.get_rect(center=self.screen.get_rect().center))
        button_size = (100, 50)
        button_size_x, button_size_y = button_size

        bottom_left_pos = (5 * button_size_x, self.Y - 4 * button_size_y)
        bottom_right_pos = (self.X - 5 * button_size_x, self.Y - 4 * button_size_y)

        def get_button(pos, button_text, button_fn):
            # surface: pygame.Surface, position: tuple, size: tuple, event_fn: Callable,
            return GUIButton(surface=self.screen, position=pos, event_fn=button_fn,
                             size=button_size, text=button_text, rect_color=(69, 69, 69),
                             text_color='white',
                             transparent=False,
                             border_color=(255, 255, 255), border_width=3)

        if self.bottom_left_button:
            self.gui_items.append(get_button(bottom_left_pos, 'Previous', self.bottom_left_fn))
        if self.bottom_right_button:
            self.gui_items.append(get_button(bottom_right_pos, 'Next', self.bottom_right_fn))

        for item in self.gui_items:
            item.show()

        self.showing = True


class GUIOvercookedPage(GUIPage):
    def __init__(self, text, font_size, bottom_left_button=False, bottom_right_button=False,
                 bottom_left_fn = None, bottom_right_fn = None):
        GUIPage.__init__(self)
        self.text = text
        self.main_font = pygame.font.Font('freesansbold.ttf', font_size)
        self.text_render = self.main_font.render(text, True, (255, 255, 255))
        self.bottom_left_button = bottom_left_button
        self.bottom_right_button = bottom_right_button
        self.bottom_left_fn = bottom_left_fn
        self.bottom_right_fn = bottom_right_fn

    def show(self):
        self.screen = pygame.display.set_mode((self.X, self.Y), pygame.SRCALPHA)


        # self.screen.blit(self.text_render, self.text_render.get_rect(center=self.screen.get_rect().center))
        # button_size = (100, 50)
        # button_size_x, button_size_y = button_size
        #
        # bottom_left_pos = (2 * button_size_x, self.Y - 2 * button_size_y)
        # bottom_right_pos = (self.X - 5 * button_size_x, self.Y - 4 * button_size_y)
        #
        # def get_button(pos, button_text, button_fn):
        #     # surface: pygame.Surface, position: tuple, size: tuple, event_fn: Callable,
        #     return GUIButton(surface=self.screen, position=pos, event_fn=button_fn,
        #                      size=button_size, text=button_text, rect_color=(69, 69, 69),
        #                      text_color='white',
        #                      transparent=False,
        #                      border_color=(255, 255, 255), border_width=3)
        #
        # if self.bottom_left_button:
        #     self.gui_items.append(get_button(bottom_left_pos, 'Previous', self.bottom_left_fn))
        # if self.bottom_right_button:
        #     self.gui_items.append(get_button(bottom_right_pos, 'Next', self.bottom_right_fn))
        #
        # for item in self.gui_items:
        #     item.show()
        #
        # self.showing = True


if __name__ == '__main__':
    pages = []
    current_page = 0

    def next_page():
        global current_page
        pages[current_page].hide()
        current_page += 1
        pages[current_page].show()

    def previous_page():
        global current_page
        pages[current_page].hide()
        current_page -= 1
        pages[current_page].show()

    pages.append(GUIPageCenterText('Welcome to our experiment investigating the performance'
                                 ' of our AI-based overcooked player.', 24,
                                 bottom_left_button=False, bottom_right_button=True,
                                   bottom_right_fn=next_page))

    pages.append(GUIPageCenterText('More tutorial text will go here...', 24,
                                 bottom_left_button=True, bottom_right_button=True,
                                   bottom_left_fn=previous_page, bottom_right_fn=next_page))

    pages.append(GUIPageCenterText('Are you ready to proceed?', 24,
                                 bottom_left_button=True, bottom_right_button=True,
                                   bottom_left_fn=previous_page, bottom_right_fn=next_page))

    pages.append(GUIPageCenterText('overcooked-ai env goes here', 24,
                                 bottom_left_button=False, bottom_right_button=False))

    clock = pygame.time.Clock()
    is_running = True

    pages[0].show()
    pygame.display.flip()

    while is_running:
        # time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                break
            for gui_item in pages[current_page].gui_items:
                is_running = gui_item.process_event(event)
        for gui_item in pages[current_page].gui_items:
            gui_item.process_standby()

    # pygame.display.flip()
    pygame.display.update()
    clock.tick(60)
