import pygame
import time
import torch


class GUITreeNode:
    def __init__(self, surface: pygame.Surface, position: tuple, size: tuple,
                    text: str, font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                    rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.current_text = text
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
        text_rendered = self.font.render(text, True, pygame.Color(self.text_color))
        text_rect = text_rendered.get_rect(center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
        self.cursor = pygame.Rect(text_rect.topright, (3, text_rect.height + 2))
        self.surface.blit(text_rendered, text_rect)

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
                            text_rendered = self.font.render(self.current_text, True, pygame.Color(self.text_color))
                            text_rect = text_rendered.get_rect(
                                center=(self.pos_x + self.size_x // 2, self.pos_y + self.size_y // 2))
                            self.surface.blit(text_rendered, text_rect)
                            pygame.display.update()

                    if time.time() % 1 > 0.5:
                        text_rendered = self.font.render(self.current_text, True, pygame.Color(self.text_color))
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
                    text, font_size, text_color, transparent,
                    rect_color, border_color, border_width)

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
            with torch.no_grad():
                weights = torch.abs(self.icct.layers.cpu())
                onehot_weights = self.icct.diff_argmax(weights)
                divisors = (weights * onehot_weights).sum(-1).unsqueeze(-1)
                divisors_filler = torch.zeros(divisors.size()).to(divisors.device)
                divisors_filler[divisors == 0] = 1
                divisors = divisors + divisors_filler
                self.icct.comparators[self.node_idx, curr_var_idx] = float(current_parsed['comparator_val']) * divisors[self.node_idx]


class GUIActionNode(GUITreeNode):
    def __init__(self, icct, surface: pygame.Surface, position: tuple, size: tuple,
                    text: str, font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                    rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.icct = icct
        super(GUIActionNode, self).__init__(surface, position, size,
                    text, font_size, text_color, transparent,
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