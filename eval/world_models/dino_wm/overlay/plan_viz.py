# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""plan.py unchanged, plus a record of the executed rollout of the final evaluation, for the planning figures.

  python plan_viz.py --config-name plan_pusht.yaml <the overrides of plan.py> +eval_start=0 +eval_end=4

When the planner's final evaluation runs (filename == "output_final"), the frames and states of the executed trajectories,
the start and goal images and the success flags are written to viz_final.npz in the planning output folder. No metric
changes.
"""
import numpy as np

import planning.evaluator as E

_orig = E.PlanEvaluator.eval_actions


def eval_actions(self, actions, action_len=None, filename="output", save_video=False):
    logs, successes, e_obses, e_states = _orig(self, actions, action_len=action_len, filename=filename, save_video=save_video)
    if filename == "output_final":
        np.savez_compressed("viz_final.npz", visual=np.asarray(e_obses["visual"]), states=np.asarray(e_states), successes=np.asarray(successes),
                            action_len=np.asarray(action_len), state_g=np.asarray(self.state_g), state_0=np.asarray(self.state_0),
                            goal_visual=np.asarray(self.obs_g["visual"]), start_visual=np.asarray(self.obs_0["visual"]))
        print("viz: saved viz_final.npz", np.asarray(e_obses["visual"]).shape, "successes", np.asarray(successes).tolist(), flush=True)
    return logs, successes, e_obses, e_states


E.PlanEvaluator.eval_actions = eval_actions
import hydra  # noqa: E402
import plan  # noqa: E402  (plan.py imports PlanEvaluator from planning.evaluator = the class patched above)


@hydra.main(config_path="conf", config_name="plan")          # declared here so that hydra finds conf/ next to this file
def main(cfg):
    return plan.main.__wrapped__(cfg)                         # the body of the official entry point, unchanged


if __name__ == "__main__":
    main()
