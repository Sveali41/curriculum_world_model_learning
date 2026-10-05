"""Use the existing MiniGrid P2E policy on generated mini-environments."""

import torch

from domain.minigrid.action_codec import MODEL_ACTION_COUNT
from modelBased.policy_training.ppo.PPO import PPO
from trainer.p2e_baseline import P2E_Ensemble, P2E_Explorer_Policy


class MiniGridP2EExplorer:
    expects_raw_obs = True

    def __init__(self, cfg, world_model, device):
        self.world_model = world_model
        self.ppo = PPO(
            state_dim=int(cfg.attention_model.embed_dim),
            action_dim=MODEL_ACTION_COUNT,
            lr_actor=float(cfg.PPO.lr_actor),
            lr_critic=float(cfg.PPO.lr_critic),
            gamma=0.99,
            K_epochs=4,
            eps_clip=0.2,
            has_continuous_action_space=False,
        )
        self.policy = P2E_Explorer_Policy(self.ppo, world_model, device, "minigrid")
        self.ensemble = P2E_Ensemble(cfg, num_models=int(cfg.p2e.num_models))
        self.reward_scale = float(cfg.p2e.intrinsic_reward_scale)
        self.update_count = 0

    def select_action(self, observation):
        self.world_model.eval()
        return self.policy.select_action(observation)

    def intrinsic_reward(self, observation, action, next_observation):
        with torch.no_grad():
            state = self.policy._prepare_wm_obs(observation)
            feature = self.world_model.encode(state)
            reward = self.ensemble.get_intrinsic_reward(
                feature, self.policy.last_model_action_idx
            )
        return reward * self.reward_scale

    def record_transition(self, *args, **kwargs):
        self.policy.record_transition(*args, **kwargs)

    def finish_rollout(self, dataset):
        count = len(dataset["a"])
        if count == 0 or len(self.ppo.buffer.rewards) != count:
            raise RuntimeError(
                "MiniGrid P2E rollout and PPO buffer have different transition counts: "
                f"dataset={count}, buffer={len(self.ppo.buffer.rewards)}"
            )
        self.ppo.update(collect_metrics=False)
        self.world_model.eval()
        wm_device = next(self.world_model.parameters()).device
        with torch.no_grad():
            # Generated MiniGrid datasets are already NCHW; the policy's NumPy
            # input path assumes raw HWC observations from the environment.
            observations = torch.as_tensor(dataset["a"], device=wm_device)
            next_observations = torch.as_tensor(dataset["b"], device=wm_device)
            feature = self.world_model.encode(self.policy._prepare_wm_obs(observations))
            next_feature = self.world_model.encode(self.policy._prepare_wm_obs(next_observations))
        actions = torch.as_tensor(dataset["c"], dtype=torch.long).reshape(-1)
        ensemble_loss = self.ensemble.train_step(feature, actions, next_feature)
        self.update_count += 1
        return ensemble_loss
