"""Tests for the verifiers <-> tinker conversion layer (no API calls)."""

import pytest

try:
    import verifiers as _verifiers  # noqa: F401

    _has_verifiers = True
except ImportError:
    _has_verifiers = False

pytestmark = pytest.mark.skipif(not _has_verifiers, reason="verifiers not installed")


def _rollout_output(reward: float, with_trajectory: bool = True) -> dict:
    import verifiers as vf

    output = vf.RolloutOutput(
        example_id=0,
        prompt=[{"role": "user", "content": "q"}],
        completion=[{"role": "assistant", "content": "a"}],
        reward=reward,
        timing={},
        is_completed=True,
        is_truncated=False,
        metrics={"correct": reward},
    )
    if with_trajectory:
        output["trajectory"] = [
            {
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "prompt_mask": [0, 0, 0],
                    "completion_ids": [4, 5],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.1, -0.2],
                }
            }
        ]
    return output


class TestConvertOutputsToTrajectoryGroup:
    def test_converts_tokens_rewards_and_metrics(self) -> None:
        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_outputs_to_trajectory_group,
        )

        group = convert_outputs_to_trajectory_group([_rollout_output(1.0), _rollout_output(0.0)])

        assert group.final_rewards_G == [1.0, 0.0]
        assert group.metrics_G == [{"correct": 1.0}, {"correct": 0.0}]
        assert len(group.trajectories_G) == 2
        transition = group.trajectories_G[0].transitions[0]
        assert transition.ob.to_ints() == [1, 2, 3]
        assert transition.ac.tokens == [4, 5]
        assert transition.ac.maybe_logprobs == [-0.1, -0.2]
        assert transition.episode_done

    def test_missing_trajectory_field_raises(self) -> None:
        """`trajectory` is opt-in on RolloutOutput; forgetting state_columns
        would otherwise train on silently empty trajectories."""
        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_outputs_to_trajectory_group,
        )

        with pytest.raises(ValueError, match="state_columns"):
            convert_outputs_to_trajectory_group([_rollout_output(1.0, with_trajectory=False)])


class TestGetRolloutInputs:
    def test_does_not_forward_string_task(self) -> None:
        """verifiers rejects plain-string `task` on a rollout input."""
        from unittest.mock import MagicMock

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersEnvGroupBuilder

        builder = VerifiersEnvGroupBuilder(
            vf_env=MagicMock(),
            prompt=[{"role": "user", "content": "q"}],
            example_id=7,
            task="arithmetic",
            answer="4",
            info={"env_id": "arithmetic"},
        )
        inputs = builder.get_rollout_inputs(3)

        assert len(inputs) == 3
        assert all("task" not in rollout_input for rollout_input in inputs)
        assert inputs[0]["info"] == {"env_id": "arithmetic"}
        assert inputs[0]["example_id"] == 7
        # `task` is still available for logging tags.
        assert builder.logging_tags() == ["arithmetic"]
