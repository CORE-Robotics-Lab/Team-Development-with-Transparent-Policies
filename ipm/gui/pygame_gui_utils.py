import pygame
import time
import torch
import random
from ipm.models.idct_helpers import convert_decision_to_leaf, convert_leaf_to_decision

class Legend:
    def __init__(self, surface, x, y, w, h, decision_color, action_color,
                 decision_border_color, action_border_color, highlight_color, font,
                 option_list, selected=-1, transparent=True):
        self.decision_color = decision_color
        self.decision_border_color = decision_border_color
        self.action_color = action_color
        self.action_border_color = action_border_color
        self.position = (x, y)

        self.font = pygame.font.Font('freesansbold.ttf', 24)
        self.surface = surface

        self.rect = pygame.Rect(x, y, w, h)
        self.rect_shape = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self.rect_text = pygame.Rect(x + w + 30, y, w, h)
        self.rect_text_shape = pygame.Surface(self.rect.size)

        y2 = y + h + 10
        self.rect_decision = pygame.Rect(x, y2, w, h)
        self.rect_decision_shape = pygame.Surface(self.rect_decision.size, pygame.SRCALPHA)
        self.rect_decision_text = pygame.Rect(x + w + 30, y2, w, h)
        self.rect_decision_text_shape = pygame.Surface(self.rect_decision.size)

        y3 = y2 + h + 10
        self.position2 = (x, y2)
        self.position3 = (x, y3)
        self.rect_action = pygame.Rect(x, y3, w, h)
        self.rect_action_shape = pygame.Surface(self.rect_action.size, pygame.SRCALPHA)
        self.rect_action_text = pygame.Rect(x + w + 30, y3, w, h)
        self.rect_action_text_shape = pygame.Surface(self.rect_action_text.size)


    def draw(self):
        self.rect_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_shape, self.position)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect, width=2)
        msg = self.font.render(" = Modifiable", 1, (0, 0, 0))
        x, y = self.position
        x += 130
        y += 8
        self.surface.blit(msg, (x, y))

        self.rect_decision_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_decision_shape, self.position2)
        self.rect_decision_shape.fill(self.decision_color)
        self.surface.blit(self.rect_decision_shape, self.position2)
        pygame.draw.rect(self.surface, self.decision_border_color, self.rect_decision, width=2)
        msg = self.font.render(" = Decision Node", 1, (0, 0, 0))
        x, y = self.position2
        x += 130
        y += 8
        self.surface.blit(msg, (x, y))

        self.rect_action_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_action_shape, self.position3)
        self.rect_action_shape.fill(self.action_color)
        self.surface.blit(self.rect_action_shape, self.position3)
        pygame.draw.rect(self.surface, self.action_border_color, self.rect_action, width=2)
        msg = self.font.render(" = Action Node", 1, (0, 0, 0))
        x, y = self.position3
        x += 130
        y += 8
        self.surface.blit(msg, (x, y))

    def draw_children(self):
        pass

    def process_event(self, event):
        return 'continue', None

class OptionBox:
    def __init__(self, surface, x, y, w, h, color, highlight_color, font,
                 option_list, selected=-1, transparent=True):
        self.color = color
        self.highlight_color = highlight_color
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

    def draw(self):
        self.rect_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_shape, self.position)
        self.rect_shape.fill(self.highlight_color if self.menu_active else self.color)
        self.surface.blit(self.rect_shape, self.position)
        pygame.draw.rect(self.surface, (0, 0, 0, 128), self.rect, width=2)
        msg = self.font.render(self.option_list[self.selected], 1, (0, 0, 0))
        self.surface.blit(msg, msg.get_rect(center=self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.option_list):
                rect = self.rect.copy()
                rect.y += (i + 1) * self.rect.height
                if self.transparent:
                    rect_shape = pygame.Surface(rect.size, pygame.SRCALPHA)
                else:
                    rect_shape = pygame.Surface(rect.size)

                rect_shape.fill((255, 255, 255, 255))
                self.surface.blit(rect_shape, (rect.x, rect.y))
                rect_shape.fill(self.highlight_color if i == self.active_option else self.color)
                self.surface.blit(rect_shape, (rect.x, rect.y))
                pygame.draw.rect(self.surface, (0, 0, 0, 128), rect, width=2)

                msg = self.font.render(text, 1, (0, 0, 0))
                self.surface.blit(msg, msg.get_rect(center=rect.center))

    def process_event(self, event):
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)

        for i in range(len(self.option_list)):
            rect = self.rect.copy()
            rect.y += (i + 1) * self.rect.height
            if rect.collidepoint(mpos):
                self.active_option = i
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
        return -1


class TextBox:
    def __init__(self, surface, x, y, w, h, color, highlight_color,
                 font, value='0.0', transparent=True):
        self.color = color
        self.highlight_color = highlight_color
        self.rect = pygame.Rect(x, y, w, h)
        self.transparent = transparent
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

    def draw(self):
        self.rect_shape.fill((255, 255, 255))
        self.surface.blit(self.rect_shape, self.position)
        mpos = pygame.mouse.get_pos()
        self.rect_shape.fill(self.highlight_color if self.rect.collidepoint(mpos) else self.color )
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
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)

        if self.menu_active:
            if event.type == pygame.MOUSEBUTTONUP:
                self.currently_editing = True

        if self.currently_editing:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.value = self.value[:-1]
                elif event.key == pygame.K_RETURN and len(self.value) > 0:
                    self.currently_editing = False
                elif event.type == pygame.QUIT:
                    return False
                else:
                    self.value = self.value + event.unicode


class GUITreeNode:
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

        self.rectangle = pygame.Rect((self.pos_x, self.pos_y, self.size_x, self.size_y))

        if self.transparent:
            self.rect_shape = pygame.Surface(self.rectangle.size, pygame.SRCALPHA)
        else:
            self.rect_shape = pygame.Surface(self.rectangle.size)

        self.child_elements = []

    def draw(self):
        self.rect_shape.fill(self.rect_color)
        self.surface.blit(self.rect_shape, self.position)
        if self.border_width > 0:
            pygame.draw.rect(self.surface, self.border_color, self.rectangle, width=self.border_width)

    def draw_children(self):
        for child in self.child_elements:
            child.draw()

    def process_event(self, event):
        for child in self.child_elements:
           child.process_event(event)
        return True

class Arrow:
    def __init__(self, surface: pygame.Surface, start: pygame.Vector2, end: pygame.Vector2, color=pygame.Color('black'),
                   body_width: int = 5, head_width: int = 15, head_height: int = 10):
        self.surface = surface
        self.start = start
        self.end = end
        self.color = color
        self.body_width = body_width
        self.head_width = head_width
        self.head_height = head_height

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

    def process_event(self, event):
        return 'continue', None

    def draw(self):
        pygame.draw.polygon(self.surface, self.color, self.head_vertices)
        pygame.draw.polygon(self.surface, self.color, self.body_verts)

    def draw_children(self):
        pass


class GUIDecisionNode(GUITreeNode):
    def __init__(self, icct, node_idx: int, env_feat_names: [], surface: pygame.Surface, position: tuple, size: tuple,
                    font_size: int = 12, text_color: str = 'black', transparent: bool = True,
                    variable_idx: int = -1, compare_sign = '<', comparator_value='1.0',
                    rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.icct = icct
        self.node_idx = node_idx
        self.env_feat_names = env_feat_names
        super(GUIDecisionNode, self).__init__(surface=surface, position=position,
                                              size=size, font_size=font_size,
                                              text_color=text_color, transparent=transparent,
                                              rect_color=rect_color, border_color=border_color,
                                              border_width=border_width)

        option_color = (137, 207, 240, 128)
        option_highlight_color = (137, 207, 240, 255)

        x, y = position

        node_options_h = 25
        node_options_w = 180
        node_options_y = 10 + y
        node_options_x = self.pos_x + self.size_x // 2 - node_options_w // 2

        # below assumes that root node will be idx 0
        if node_idx != 0:
            choices = ['Decision Node', 'Action Node']
        else:
            choices = ['Decision Node']

        variable_options_h = 25
        variable_options_w = 180
        variable_options_y = 10 + node_options_y + node_options_h
        variable_options_x = 10 + x

        self.variables_box = OptionBox(surface,
                                  variable_options_x, variable_options_y,
                                  variable_options_w, variable_options_h, option_color,
                                  option_highlight_color,
                                  pygame.font.SysFont(None, 30),
                                  env_feat_names,
                                  selected=variable_idx)
        self.child_elements.append(self.variables_box)

        sign_options_h = 25
        sign_options_w = 40
        sign_options_y = 10 + node_options_y + node_options_h
        sign_options_x = 10 + variable_options_x + variable_options_w

        signs = ['<', '>']

        self.sign_box = OptionBox(surface,
                                  sign_options_x, sign_options_y,
                                  sign_options_w, sign_options_h, option_color,
                                  option_highlight_color,
                                  pygame.font.SysFont(None, 30), signs,
                                  selected=signs.index(compare_sign))
        self.child_elements.append(self.sign_box)

        compare_options_h = 25
        compare_options_w = 70
        compare_options_y = 10 + node_options_y + node_options_h
        compare_options_x = 10 + sign_options_x + sign_options_w

        self.comparator_box = TextBox(surface,
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
                                  node_options_w, node_options_h, option_color,
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
                self.icct.layers[self.node_idx, self.variables_box.selected] = 2 * max_weight
                print('New var value!')
            self.variables_box.previously_selected = self.variables_box.selected
        if self.sign_box.selected != self.sign_box.previously_selected:
            with torch.no_grad():
                is_greater_than = (self.icct.alpha.cpu() * self.icct.layers.cpu() > 0)[self.node_idx, self.variables_box.selected]
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
    def __init__(self, tree, surface: pygame.Surface, position: tuple, size: tuple, name: str,
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
    def __init__(self, tree, surface: pygame.Surface, position: tuple, size: tuple,
                 leaf_idx:int, action_idx: int, actions_list: [], font_size: int = 12,
                 text_color: str = 'black', transparent: bool = True,
                 rect_color: tuple = None, border_color: tuple = None, border_width: int = 0):
        self.tree = tree
        self.leaf_idx = leaf_idx
        super(GUIActionNodeIDCT, self).__init__(surface, position, size,
                    font_size, text_color, transparent,
                    rect_color, border_color, border_width)


        option_color = (240, 128, 101, 128)
        option_highlight_color = (240, 128, 101, 255)

        x, y = position


        node_options_h = 25
        node_options_w = 160
        node_options_y = 10 + y
        node_options_x = self.pos_x + self.size_x // 2 - node_options_w // 2

        choices = ['Decision Node', 'Action Node']


        variable_options_h = 25
        variable_options_w = 160
        variable_options_y = 5 + node_options_y + node_options_h
        variable_options_x = 10 + x

        self.actions_box = OptionBox(surface,
                                     variable_options_x, variable_options_y,
                                     variable_options_w, variable_options_h, option_color,
                                     option_highlight_color,
                                     pygame.font.SysFont(None, 30),
                                     actions_list,
                                     selected=action_idx)
        self.child_elements.append(self.actions_box)

        self.node_box = OptionBox(surface,
                                  node_options_x, node_options_y,
                                  node_options_w, node_options_h, option_color,
                                  option_highlight_color,
                                  pygame.font.SysFont(None, 30),
                                  choices,
                                  selected=1)
        self.child_elements.append(self.node_box)

    def process_event(self, event):
        super(GUIActionNodeIDCT, self).process_event(event)
        if self.actions_box.selected != self.actions_box.previously_selected:
            with torch.no_grad():
                logits = self.tree.leaf_init_information[self.leaf_idx][2]
                for i in range(len(logits)):
                    if i == self.actions_box.selected:
                        logits[i] = 2
                    else:
                        logits[i] = -2
                print('New action value!')
            self.actions_box.previously_selected = self.actions_box.selected
        if self.node_box.selected != self.node_box.previously_selected:
            if self.node_box.selected == 0:
                new_tree = convert_leaf_to_decision(self.tree, self.leaf_idx)
                return 'new_tree', new_tree
        return 'continue', None


