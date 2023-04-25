import os

import pygame


class SettingsWrapper:
    def __init__(self):
        self.zoom = 1
        self.old_zoom = 1
        self.max_zoom = 3
        self.min_zoom = 1
        self.width, self.height = 1920, 1080
        self.offset_x, self.offset_y = 0, 0
        self.absolute_x, self.absolute_y = self.width // 2, self.height // 2
        self.options_menus_per_domain = {0: [], 1: [], 2: [], 3: []}

    def check_if_options_menu_open(self, domain_idx) -> (bool, int):
        for i, menu_open in enumerate(self.options_menus_per_domain[domain_idx]):
            if menu_open:
                return True, i
        return False, -1

    def zoom_out(self):
        self.old_zoom = self.zoom
        self.zoom = max(self.zoom - 0.1, self.min_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom

    def zoom_in(self):
        self.old_zoom = self.zoom
        self.zoom = min(self.zoom + 0.1, self.max_zoom)
        assert self.max_zoom >= self.zoom >= self.min_zoom

def get_next_user_id():
    # get the next user id
    # we can get this info by looking at the folders in the data/experiments/conditions folder
    # need to iterate through each folder in each condition and get the max number
    latest_user_id = 0

    if not os.path.exists(os.path.join('data', 'experiments')):
        os.mkdir(os.path.join('data', 'experiments'))

    for condition in os.listdir(os.path.join('data', 'experiments')):
        if not condition.startswith('.'):
            for user_folder in os.listdir(os.path.join('data', 'experiments', condition)):
                if not user_folder.startswith('.'):
                    user_id = int(user_folder.split('_')[-1])
                    latest_user_id = max(latest_user_id, int(user_id))
    return latest_user_id + 1


def process_zoom(screen, settings):
    # create pygame subsurface
    wnd_w, wnd_h = screen.get_size()
    zoom_size = (round(wnd_w / settings.zoom), round(wnd_h / settings.zoom))
    # when fully zoomed in, make sure it is in bounds
    x = settings.absolute_x
    y = settings.absolute_y
    x = (x + settings.offset_x) // settings.zoom
    y = (y + settings.offset_y) // settings.zoom

    # prevent any black borders
    x = max(x, zoom_size[0] // 2)
    y = max(y, zoom_size[1] // 2)
    x = min(x, wnd_w - zoom_size[0] // 2)
    y = min(y, wnd_h - zoom_size[1] // 2)

    if settings.zoom == 1:
        x = wnd_w // 2
        y = wnd_h // 2

    settings.absolute_x = x
    settings.absolute_y = y
    settings.offset_x = int(settings.absolute_x * settings.zoom) - settings.width // 2
    settings.offset_y = int(settings.absolute_y * settings.zoom) - settings.height // 2

    zoom_area = pygame.Rect(0, 0, *zoom_size)
    # if self.settings.zoom == previous_zoom:
    #     x, y = wnd_w // 2, wnd_h // 2
    zoom_area.center = (x, y)
    zoom_surf = pygame.Surface(zoom_area.size)
    zoom_surf.blit(screen, (0, 0), zoom_area)
    zoom_surf = pygame.transform.scale(zoom_surf, (wnd_w, wnd_h))
    screen.blit(zoom_surf, (0, 0))