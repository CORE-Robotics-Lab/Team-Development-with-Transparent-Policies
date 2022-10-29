import pygame
import time
import torch

class OptionBox:
    def __init__(self, x, y, w, h, color, highlight_color, font, option_list, selected=0):
        self.color = color
        self.highlight_color = highlight_color
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.option_list = option_list
        self.selected = selected
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1

    def draw(self, surf):
        pygame.draw.rect(surf, self.highlight_color if self.menu_active else self.color, self.rect)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 2)
        msg = self.font.render(self.option_list[self.selected], 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center=self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.option_list):
                rect = self.rect.copy()
                rect.y += (i + 1) * self.rect.height
                pygame.draw.rect(surf, self.highlight_color if i == self.active_option else self.color, rect)
                msg = self.font.render(text, 1, (0, 0, 0))
                surf.blit(msg, msg.get_rect(center=rect.center))
            outer_rect = (
            self.rect.x, self.rect.y + self.rect.height, self.rect.width, self.rect.height * len(self.option_list))
            pygame.draw.rect(surf, (0, 0, 0), outer_rect, 2)

    def update(self, event):
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)

        self.active_option = -1
        for i in range(len(self.option_list)):
            rect = self.rect.copy()
            rect.y += (i + 1) * self.rect.height
            if rect.collidepoint(mpos):
                self.active_option = i
                break

        if not self.menu_active and self.active_option == -1:
            self.draw_menu = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.menu_active:
                self.draw_menu = not self.draw_menu
            elif self.draw_menu and self.active_option >= 0:
                self.selected = self.active_option
                self.draw_menu = False
                return self.active_option
        return -1

class GUITreeNode:
    def __init__(self, surface: pygame.Surface, position: tuple, size: tuple,
                    text: str, top_text=None, font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                    rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.current_text = text
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
        self.previous_text = text
        self.text_color = text_color

        if self.rect_color is None:
            self.rect_color = (255, 255, 255, 255)
        if self.border_color is None:
            self.border_color = (0, 0, 0, 255)

        self.rectangle = pygame.Rect((self.pos_x, self.pos_y, self.size_x, self.size_y))

        if self.transparent:
            self.rect_shape = pygame.Surface(self.rectangle.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rectangle.size)

        self.rect_shape.fill(self.rect_color)
        self.surface.blit(self.rect_shape, self.position)
        if self.border_width > 0:
            pygame.draw.rect(self.surface, self.border_color, self.rectangle, width=self.border_width)

        def draw_text(font, text, y_pos):
            text_rendered = font.render(text, True, pygame.Color(self.text_color))
            self.text_rect = text_rendered.get_rect(center=(self.pos_x + self.size_x // 2, y_pos))
            self.surface.blit(text_rendered, self.text_rect)

        if top_text is None:
            draw_text(self.main_font, text, self.pos_y + self.size_y // 2)
        else:
            draw_text(self.main_font, top_text, self.pos_y + self.size_y // 3)
            draw_text(self.secondary_font, text, self.pos_y + 2 * self.size_y // 3)

        self.cursor = pygame.Rect(self.text_rect.topright, (3, self.text_rect.height + 2))

    def process_event(self, event):
        mouse_position = pygame.mouse.get_pos()
        self.current_text = self.previous_text
        if self.rectangle.collidepoint(mouse_position):
            if event.type == pygame.MOUSEBUTTONUP:
                run = True
                while run:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_BACKSPACE:
                                self.current_text = self.current_text[:-1]
                            elif event.key == pygame.K_RETURN:
                                run = False
                            elif event.type == pygame.QUIT:
                                return False
                            else:
                                self.current_text = self.current_text + event.unicode

                            self.rect_shape.fill((255, 255, 255))
                            self.surface.blit(self.rect_shape, self.position)
                            self.rect_shape.fill(self.rect_color)
                            self.surface.blit(self.rect_shape, self.position)
                            if self.border_width > 0:
                                pygame.draw.rect(self.surface, self.border_color, self.rectangle, width=self.border_width)
                            text_rendered = self.main_font.render(self.current_text, True, pygame.Color(self.text_color))
                            text_rect = text_rendered.get_rect(
                                center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
                            self.surface.blit(text_rendered, text_rect)
                            pygame.display.update()

                    if time.time() % 1 > 0.5:
                        text_rendered = self.main_font.render(self.current_text, True, pygame.Color(self.text_color))
                        text_rect = text_rendered.get_rect(center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
                        self.cursor.midleft = text_rect.midright
                        pygame.draw.rect(self.surface, (0, 0, 0), self.cursor)
                        pygame.display.update()
                self.process_new_text()
        return True

    def process_new_text(self):
        pass

    def process_standby(self):
        mouse_position = pygame.mouse.get_pos()
        if self.rectangle.collidepoint(mouse_position):
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                pass
                # if self.onePress:
                #     self.onclickFunction()
                # elif not self.alreadyPressed:
                #     self.onclickFunction()
                #     self.alreadyPressed = True
            else:
                self.alreadyPressed = False
        else:
            pass
            #self.buttonSurface.fill(self.fillColors['normal'])
        #self.parent_surface.blit(self.buttonSurface, self.buttonRect)


class GUIDecisionNode(GUITreeNode):
    def __init__(self, icct, node_idx: int, env_feat_names: [], surface: pygame.Surface, position: tuple, size: tuple,
                    text: str, font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                    rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.icct = icct
        self.node_idx = node_idx
        self.env_feat_names = env_feat_names
        super(GUIDecisionNode, self).__init__(surface, position, size,
                    text, None, font_size, text_color, transparent,
                    rect_color, border_color, border_width)
        option_color = (150, 150, 150)
        option_highlight_color = (150, 150, 150)
        options_height = 40
        options_width = 160
        options_y_space = 10
        options_x_space = 10
        x, y = position
        self.variables_box = OptionBox(x + options_x_space, y + options_y_space,
                                       options_width, options_height, option_color,
                                       option_highlight_color,
                                       pygame.font.SysFont(None, 30),
                                       env_feat_names)
        # self.variables_box.draw(surface)

    def process_event(self, event_list):
        super(GUIDecisionNode, self).process_event(event_list)
        self.variables_box.update(event_list)

    def parse_text(self, text):
        text_info = {'compare_sign': None, 'comparator_val': None, 'var_name': None}
        if '<' in text:
            text_info['compare_sign'] = '<'
        else:
            text_info['compare_sign'] = '>'
        text_split = text.split(text_info['compare_sign'])
        text_info['var_name'] = text_split[0].strip()
        text_info['comparator_val'] = text_split[1].strip()
        return text_info

    def process_new_text(self):
        print('Processing decision node changes...')
        current_parsed = self.parse_text(self.current_text)
        previous_parsed = self.parse_text(self.previous_text)
        curr_var_idx = self.env_feat_names.index(current_parsed['var_name'])
        prev_var_idx = self.env_feat_names.index(previous_parsed['var_name'])

        if curr_var_idx != prev_var_idx:
            with torch.no_grad():
                weights = torch.abs(self.icct.layers.cpu())
                max_weight = torch.max(weights[self.node_idx])
                self.icct.layers[self.node_idx, curr_var_idx] = 2 * max_weight
                print('New var value!')

        if current_parsed['compare_sign'] != previous_parsed['compare_sign']:
            if curr_var_idx == prev_var_idx:
                with torch.no_grad():
                    self.icct.layers[self.node_idx, curr_var_idx] *= -1
            else:
                # In this case, the user changed the variable, so we need to check for its actual compare sign.
                with torch.no_grad():
                    is_greater_than = (self.icct.alpha.cpu() * self.icct.layers.cpu() > 0)[self.node_idx, curr_var_idx]
                    sign_for_new_var = '>' if is_greater_than else '<'
                    if current_parsed['compare_sign'] != sign_for_new_var:
                        self.icct.layers[self.node_idx, curr_var_idx] *= -1

        if current_parsed['comparator_val'] != previous_parsed['comparator_val']:
            multiplier = float(current_parsed['comparator_val']) / float(previous_parsed['comparator_val'])
            with torch.no_grad():
                self.icct.layers[self.node_idx, curr_var_idx] /= multiplier


class GUIActionNode(GUITreeNode):
    def __init__(self, icct, surface: pygame.Surface, position: tuple, size: tuple, name: str,
                    text: str, font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                    rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.icct = icct
        super(GUIActionNode, self).__init__(surface, position, size,
                    text, name, font_size, text_color, transparent,
                    rect_color, border_color, border_width)

    def process_new_text(self):
        pass
        # self.original_text =

def draw_arrow(surface: pygame.Surface, start: pygame.Vector2, end: pygame.Vector2, color = pygame.Color('black'),
               body_width: int = 5, head_width: int = 15, head_height: int = 10):
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    head_vertices = [pygame.Vector2(0, head_height / 2),
                     pygame.Vector2(head_width / 2, -head_height / 2),
                     pygame.Vector2(-head_width / 2, -head_height / 2)]
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_vertices)):
        head_vertices[i].rotate_ip(-angle)
        head_vertices[i] += translation
        head_vertices[i] += start

    pygame.draw.polygon(surface, color, head_vertices)

    if arrow.length() >= head_height:
        body_verts = [pygame.Vector2(-body_width / 2, body_length / 2),
                      pygame.Vector2(body_width / 2, body_length / 2),
                      pygame.Vector2(body_width / 2, -body_length / 2),
                      pygame.Vector2(-body_width / 2, -body_length / 2)]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pygame.draw.polygon(surface, color, body_verts)