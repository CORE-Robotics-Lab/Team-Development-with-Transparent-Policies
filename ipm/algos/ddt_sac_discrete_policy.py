# Created by Yaru Niu and Andrew Silva

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as torch
from torch import nn
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, create_sde_features_extractor, register_policy, get_policy_from_name

from ipm.algos.sac_policies import SACPolicy, LOG_STD_MAX, LOG_STD_MIN
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)


from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from ipm.algos.sac_discrete_policies import SACDiscretePolicy


from ipm.models.idct import IDCT
from stable_baselines3.common.type_aliases import Schedule


class DDTDiscreteActor(BasePolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        ddt_kwargs: Dict[str, Any] = None,
    ):
        super(DDTDiscreteActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.log_std = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.ddt_kwargs = ddt_kwargs
        self.action_space = action_space

        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # self.action_dim = get_action_dim(self.action_space)
        self.action_dim = action_space.n

        self.last_layer_dim = features_dim
        self.ddt = IDCT(input_dim=self.features_dim,
                                   output_dim=self.action_dim,
                                   weights=self.ddt_kwargs['weights'],
                                   comparators=self.ddt_kwargs['comparators'],
                                   leaves=self.ddt_kwargs['leaves'],
                                   alpha=self.ddt_kwargs['alpha'],
                                   use_individual_alpha=self.ddt_kwargs['use_individual_alpha'],
                                   device=self.ddt_kwargs['device'],
                                   hard_node=self.ddt_kwargs['hard_node'],
                                   argmax_tau=self.ddt_kwargs['argmax_tau'],
                                   l1_hard_attn=self.ddt_kwargs['l1_hard_attn'],
                                   use_gumbel_softmax=self.ddt_kwargs['use_gumbel_softmax'],
                                   is_value=False,
                                   fixed_idct=self.ddt_kwargs['fixed_idct'],
                                   alg_type=self.ddt_kwargs['alg_type']).to(self.ddt_kwargs['device'])
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        # if self.use_sde:
        #     latent_sde_dim = last_layer_dim
        #     # Separate features extractor for gSDE
        #     if sde_net_arch is not None:
        #         self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
        #             features_dim, sde_net_arch, activation_fn
        #         )
        #
        #     self.action_dist = StateDependentNoiseDistribution(
        #         action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
        #     )
        #     self.mu, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=last_layer_dim, latent_sde_dim=latent_sde_dim, log_std_init=log_std_init
        #     )
        #     # Avoid numerical issues by limiting the mean of the Gaussian
        #     # to be in [-clip_mean, clip_mean]
        #     if clip_mean > 0.0:
        #         self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        # else:
        #     self.action_dist = SquashedDiagGaussianDistribution(action_dim)


    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                sde_net_arch=self.sde_net_arch,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> torch.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        probabilities = self.ddt(features)

        # if self.use_sde:
        #     latent_sde = self.ddt  # Feature extractor goes here
        #     if self.sde_features_extractor is not None:
        #         latent_sde = self.sde_features_extractor(features)
        #     return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        # Original Implementation to cap the standard deviation
        # log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return probabilities

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.ddt(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        distribution = self._get_action_dist_from_latent(obs)
        # Note: the action is squashed
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions
        # return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self._get_action_dist_from_latent(obs)
        # Note: the action is squashed
        actions = distribution.get_actions(deterministic=False)
        log_prob = distribution.log_prob(actions)
        # return action and associated log prob
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, log_prob

    def action_info(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self._get_action_dist_from_latent(obs)
        action_probabilities = distribution.distribution.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-10
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.forward(observation, deterministic)


class DDT_SACDiscretePolicy(SACDiscretePolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            ddt_kwargs: Dict[str, Any] = None,
    ):
        self.ddt_kwargs = ddt_kwargs
        super(DDT_SACDiscretePolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DDTDiscreteActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs['ddt_kwargs'] = self.ddt_kwargs
        return DDTDiscreteActor(**actor_kwargs).to(self.device)

register_policy("DDT_SACDiscretePolicy", DDT_SACDiscretePolicy)
