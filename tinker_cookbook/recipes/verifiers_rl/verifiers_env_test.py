"""Tests for the verifiers episode -> tinker trajectory conversion (no API calls)."""

import pytest

try:
    import verifiers.v1 as _verifiers  # noqa: F401

    _has_verifiers = True
except ImportError:
    _has_verifiers = False

pytestmark = pytest.mark.skipif(not _has_verifiers, reason="verifiers not installed")


def _user_node(token_ids: list[int], parent: int | None = None):
    import verifiers.v1 as vf

    return vf.MessageNode(
        parent=parent,
        message=vf.UserMessage(content="q"),
        sampled=False,
        token_ids=token_ids,
        mask=[False] * len(token_ids),
    )


def _assistant_node(
    parent: int,
    scaffold: list[int],
    sampled: list[int],
    logprobs: list[float],
):
    import verifiers.v1 as vf

    return vf.MessageNode(
        parent=parent,
        message=vf.AssistantMessage(content="a"),
        sampled=True,
        token_ids=scaffold + sampled,
        mask=[False] * len(scaffold) + [True] * len(sampled),
        logprobs=logprobs,
    )


def _episode(nodes, rewards=None, metrics=None):
    import verifiers.v1 as vf

    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards=rewards or {},
        metrics=metrics or {},
        is_completed=True,
        ok=True,
    )
    episode = vf.Episode(task=trace.task, traces=[trace], ok=True)
    return episode


class TestConvertEpisodesToTrajectoryGroup:
    def test_single_turn_episode(self) -> None:
        import verifiers.v1 as vf

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_episodes_to_trajectory_group,
        )

        episode = _episode(
            nodes=[
                _user_node([1, 2, 3]),
                _assistant_node(parent=0, scaffold=[4, 5], sampled=[6, 7], logprobs=[-0.1, -0.2]),
            ],
            rewards={"correct": vf.Reward(score=1.0)},
            metrics={"accuracy": 1.0, "unscored": None},
        )
        group = convert_episodes_to_trajectory_group([episode])

        assert group.final_rewards_G == [1.0]
        assert group.metrics_G == [{"accuracy": 1.0}]
        [trajectory] = group.trajectories_G
        [transition] = trajectory.transitions
        # The observation is the full context: prompt plus the assistant
        # node's unsampled generation-prompt scaffold.
        assert transition.ob.to_ints() == [1, 2, 3, 4, 5]
        assert transition.ac.tokens == [6, 7]
        assert transition.ac.maybe_logprobs == [-0.1, -0.2]
        assert transition.episode_done
        assert trajectory.final_ob.to_ints() == [1, 2, 3, 4, 5, 6, 7]

    def test_multi_turn_episode_extends_by_prefix(self) -> None:
        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_episodes_to_trajectory_group,
        )

        episode = _episode(
            nodes=[
                _user_node([1, 2]),
                _assistant_node(parent=0, scaffold=[3], sampled=[4, 5], logprobs=[-0.1, -0.2]),
                _user_node([6, 7], parent=1),
                _assistant_node(parent=2, scaffold=[8], sampled=[9], logprobs=[-0.3]),
            ]
        )
        group = convert_episodes_to_trajectory_group([episode])

        [trajectory] = group.trajectories_G
        first, second = trajectory.transitions
        assert first.ob.to_ints() == [1, 2, 3]
        assert first.ac.tokens == [4, 5]
        assert not first.episode_done
        # The second observation extends the first observation+action, so
        # trajectory_to_data can merge the turns into one datum.
        assert second.ob.to_ints() == [1, 2, 3, 4, 5, 6, 7, 8]
        assert second.ac.tokens == [9]
        assert second.ac.maybe_logprobs == [-0.3]
        assert second.episode_done

    def test_weighted_rewards_sum(self) -> None:
        import verifiers.v1 as vf

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_episodes_to_trajectory_group,
        )

        episode = _episode(
            nodes=[
                _user_node([1]),
                _assistant_node(parent=0, scaffold=[], sampled=[2], logprobs=[-0.1]),
            ],
            rewards={
                "correct": vf.Reward(score=1.0),
                "style": vf.Reward(score=0.5, weight=0.2),
                "unscored": None,
            },
        )
        group = convert_episodes_to_trajectory_group([episode])
        assert group.final_rewards_G == [pytest.approx(1.1)]

    def test_failed_episode_stays_in_group(self) -> None:
        """An episode that produced no trace trains on nothing but keeps its
        group slot, so reward centering sees the failure."""
        import verifiers.v1 as vf

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_episodes_to_trajectory_group,
        )

        ok = _episode(
            nodes=[
                _user_node([1]),
                _assistant_node(parent=0, scaffold=[], sampled=[2], logprobs=[-0.1]),
            ],
            rewards={"correct": vf.Reward(score=1.0)},
        )
        failed = vf.Episode(task=ok.task, traces=[], ok=False)
        group = convert_episodes_to_trajectory_group([ok, failed])

        assert group.final_rewards_G == [1.0, 0.0]
        assert group.metrics_G[1] == {"rollout_failed": 1.0}
        [transition] = group.trajectories_G[1].transitions
        assert transition.ac.tokens == []
        assert transition.episode_done

    def test_multi_agent_episode_is_rejected(self) -> None:
        import verifiers.v1 as vf

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_episodes_to_trajectory_group,
        )

        episode = _episode(nodes=[_user_node([1])])
        episode.traces.append(
            vf.Trace(
                task=episode.task,
                agent=vf.AgentInfo(config=vf.AgentConfig(), name="judge"),
            )
        )
        with pytest.raises(ValueError, match="single-agent"):
            convert_episodes_to_trajectory_group([episode])

    def test_branched_trace_is_rejected(self) -> None:
        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            convert_episodes_to_trajectory_group,
        )

        # Two assistant leaves off the same user root: two branches.
        episode = _episode(
            nodes=[
                _user_node([1]),
                _assistant_node(parent=0, scaffold=[], sampled=[2], logprobs=[-0.1]),
                _assistant_node(parent=0, scaffold=[], sampled=[3], logprobs=[-0.2]),
            ]
        )
        with pytest.raises(ValueError, match="branches"):
            convert_episodes_to_trajectory_group([episode])
