import numpy as np

class ObservationReducer:
    def __init__(self, layout_name, base_env, cook_time_threshold):
        self.layout_name = layout_name
        self.base_env = base_env
        self.cook_time_threshold = cook_time_threshold
        self.ego_num_ingredients = 0
        self.ego_any_pot_ready = False

    def get_reduced_obs(self, obs, is_ego):
        # assumes 2 pots!
        assert obs.shape[0] == 96

        reduced_obs = []
        # first four features (direction facing)
        # reduced_obs.append(obs[0])
        # reduced_obs.append(obs[1])
        # reduced_obs.append(obs[2])
        # reduced_obs.append(obs[3])

        # next four features (held items)
        reduced_obs.append(obs[4])
        reduced_obs.append(obs[5])
        reduced_obs.append(obs[6])
        if 'two_rooms_narrow' in self.layout_name:
            reduced_obs.append(obs[7])

        # other agent facing direction
        # reduced_obs.append(obs[46])
        # reduced_obs.append(obs[47])
        # reduced_obs.append(obs[48])
        # reduced_obs.append(obs[49])

        # other player holding onion, soup, dish, or tomato
        reduced_obs.append(obs[50])
        reduced_obs.append(obs[51])
        reduced_obs.append(obs[52])
        if 'two_rooms_narrow' in self.layout_name:
            reduced_obs.append(obs[53])

        # # closest soup # onions and # tomatoes
        # reduced_obs.append(obs[16])
        # reduced_obs.append(obs[17])

        # Note: these next 2 features require that the ego plays first (blue)

        onion_on_counter = 0
        for key, obj in self.base_env.state.objects.items():
            if obj.name == 'onion':
                onion_on_counter = 1

        if 'two_rooms_narrow' in self.layout_name:
            tomato_on_counter = 0
            for key, obj in self.base_env.state.objects.items():
                if obj.name == 'tomato':
                    tomato_on_counter = 1

        # either pot needs ingredients
        # let's also consider if the agent is holding an item or if there is an item on the counter!
        if is_ego:
            self.ego_num_ingredients = obs[27] + obs[28] + obs[37] + obs[38] + onion_on_counter
            if 'two_rooms_narrow' in self.layout_name:
                self.ego_num_ingredients += tomato_on_counter
        if 3 * 2 > self.ego_num_ingredients + obs[50] + obs[53] >= 0:
            reduced_obs.append(1.0)
        else:
            reduced_obs.append(0.0)

        # either pot is almost ready
        if is_ego:
            self.ego_any_pot_ready = self.cook_time_threshold >= obs[29] > 0 or obs[26] == 1 or \
                                     self.cook_time_threshold >= obs[39] > 0 or obs[36] == 1
        if self.ego_any_pot_ready:
            reduced_obs.append(1.0)
        else:
            reduced_obs.append(0.0)

        dish_on_counter = 0
        for key, obj in self.base_env.state.objects.items():
            if obj.name == 'dish':
                dish_on_counter = 1
        reduced_obs.append(dish_on_counter)

        # reuse computations from earlier
        reduced_obs.append(onion_on_counter)
        if 'two_rooms_narrow' in self.layout_name:
            reduced_obs.append(tomato_on_counter)

        soup_on_counter = 0
        for key, obj in self.base_env.state.objects.items():
            if obj.name == 'soup':
                soup_on_counter = 1
        reduced_obs.append(soup_on_counter)

        reduced_obs = np.array(reduced_obs)

        return reduced_obs