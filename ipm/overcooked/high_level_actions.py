### Note each of these function produce plans and do not do replanning! Playing with human in a shared
# workspace (i.e., not the forced coordination domain) will require the replanning dynamically.


class Behaviors:
    def __init__(self, robot_index):
        self.horizon_env = None
        self.robot_index = robot_index

    def get_onion(self, env, last_pos=None, last_or=None):
        return self.interact_with_obj(env, last_pos, last_or, 'onion')

    def get_tomato(self, env, last_pos=None, last_or=None):
        return self.interact_with_obj(env, last_pos, last_or, 'tomato')

    def get_dish(self, env, last_pos=None, last_or=None):
        return self.interact_with_obj(env, last_pos, last_or, 'dish')

    def serve_dish(self, env, last_pos=None, last_or=None):
        return self.interact_with_obj(env, last_pos, last_or, 'serving')

    def bring_to_pot(self, env, last_pos=None, last_or=None):
        return self.interact_with_obj(env, last_pos, last_or, 'pot')

    def interact_with_obj(self, env, last_pos=None, last_or=None, obj_type='onion'):
        self.horizon_env = env

        counter_objects = self.horizon_env.mdp.get_counter_objects_dict(self.horizon_env.state)

        all_obj_locs = None
        if obj_type == 'onion':
            all_obj_locs = self.horizon_env.mdp.get_onion_dispenser_locations()
        elif obj_type == 'tomato':
            all_obj_locs = self.horizon_env.mdp.get_tomato_dispenser_locations()
        elif obj_type == 'dish':
            all_obj_locs = self.horizon_env.mdp.get_dish_dispenser_locations()
        elif obj_type == 'serving':
            all_obj_locs = self.horizon_env.mdp.get_serving_locations()
        elif obj_type == 'pot':
            all_obj_locs = self.horizon_env.mdp.get_pot_locations()
        elif obj_type == 'counter':
            all_obj_locs = self.horizon_env.mdp.get_counter_locations()
        else:
            raise ValueError('Object type not recognized')


        obj_loc = all_obj_locs + counter_objects[obj_type]

        if last_pos is None:
            _, closest_obj_loc = self.horizon_env.mp.min_cost_to_feature(
                self.horizon_env.state.players[self.robot_index].pos_and_or,
                obj_loc, with_argmin=True)
        else:
            _, closest_obj_loc = self.horizon_env.mp.min_cost_to_feature(
                (last_pos, last_or),
                obj_loc, with_argmin=True)

        # Determine where to stand to pick up
        goto_pos, goto_or = self.horizon_env.mlam._get_ml_actions_for_positions([closest_obj_loc])[0]

        if last_pos is None:
            plan = self.horizon_env.mp._get_position_plan_from_graph(
                self.horizon_env.state.players[self.robot_index].pos_and_or, (goto_pos, goto_or))
            action_list = self.horizon_env.mp.action_plan_from_positions(plan, self.horizon_env.state.players[
                self.robot_index].pos_and_or, (goto_pos, goto_or))
        else:
            plan = self.horizon_env.mp._get_position_plan_from_graph(
                (last_pos, last_or), (goto_pos, goto_or))
            action_list = self.horizon_env.mp.action_plan_from_positions(plan, (last_pos, last_or), (goto_pos, goto_or))[0]


        # save where plan should end up
        self.last_pos = goto_pos
        self.last_or = goto_or
        return action_list