estimated_causality_label = {
    "ego": {"shorthand": "E", "full": "ego (E)"},
    "causal": {"shorthand": "C", "full": "causal (C)"},
    "non_causal_ego": {"shorthand": "NC-ego", "full": "non-causal (NC-ego)"},
    "non_causal_all": {"shorthand": "NC-all", "full": "non-causal (NC-all)"},
}


def compute_estimated_causality_label(orca_result, return_shorthand=False, ego_agent_idx=0, causality_threshold=0.0):
    n_agents = len(orca_result["trajectories"])
    ade_causality_effect_on_ego = orca_result["remove_agent_i_ade"][:, ego_agent_idx]
    ade_causality_effect_on_all = orca_result["remove_agent_i_ade"][:, :]

    def _compute_label(agent_idx):
        if agent_idx == ego_agent_idx:
            return estimated_causality_label["ego"]["shorthand" if return_shorthand else "full"]
        elif ade_causality_effect_on_ego[agent_idx] > causality_threshold:
            return estimated_causality_label["causal"]["shorthand" if return_shorthand else "full"]
        elif (
                (ade_causality_effect_on_all[agent_idx, :agent_idx] <= causality_threshold).all()
                and
                (ade_causality_effect_on_all[agent_idx, agent_idx + 1:] <= causality_threshold).all()
        ):
            return estimated_causality_label["non_causal_all"]["shorthand" if return_shorthand else "full"]
        else:
            return estimated_causality_label["non_causal_ego"]["shorthand" if return_shorthand else "full"]

    return [_compute_label(agent_idx) for agent_idx in range(n_agents)]


def compute_agent_types(orca_result, return_shorthand=False, ego_agent_idx=0, causality_threshold=0.0):
    return compute_estimated_causality_label(orca_result, return_shorthand, ego_agent_idx, causality_threshold)
