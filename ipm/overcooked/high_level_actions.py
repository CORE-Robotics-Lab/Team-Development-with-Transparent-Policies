### Note each of these function produce plans and do not do replanning! Playing with human in a shared
# workspace (i.e., not the forced coordination domain) will require the replanning dynamically.


class Behaviors:
    def __init__(self, robot_index):
        self.horizon_env = None
        self.robot_index = robot_index

    def get_onion(self, env, last_pos=None, last_or=None):
        self.horizon_env = env

        counter_objects = self.horizon_env.mdp.get_counter_objects_dict(self.horizon_env.state)

        onion_loc = self.horizon_env.mdp.get_onion_dispenser_locations() + counter_objects['onion']

        if last_pos is None:
            _, closest_onion_loc = self.horizon_env.mp.min_cost_to_feature(
                self.horizon_env.state.players[self.robot_index].pos_and_or,
                onion_loc, with_argmin=True)
        else:
            _, closest_onion_loc = self.horizon_env.mp.min_cost_to_feature(
                (last_pos, last_or),
                onion_loc, with_argmin=True)

        # Determine where to stand to pick up
        goto_pos, goto_or = self.horizon_env.mlam._get_ml_actions_for_positions([closest_onion_loc])[0]

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