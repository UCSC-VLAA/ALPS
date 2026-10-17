#!/usr/bin/env python
# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The 40 LIBERO instructions of the four suites and their BERT embeddings as the policy computes them.

    cd <LIBERO root> && PYTHONPATH=. python dump_libero_instructions.py --out-dir <dir>
        -> <dir>/libero_instructions.json   {suite: [10 instructions in benchmark order]}
           <dir>/bert_pooler_libero.npz     {suite: [10, 768]}: LIBERO's get_task_embs with task_embedding_format=bert
                                            (bert-base-cased pooler output, max_word_len 25), the embedding the BERT policies use

Run in the LIBERO environment of the policies. embed_instructions.py reads both files: the own-text-tower embeddings are
scaled so that their mean norm over the 40 instructions equals the mean norm of these BERT embeddings (21.533 in the paper).
"""
import argparse
import json
import os

SUITES = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-word-len", type=int, default=25, help="data.max_word_len of the policy config (LIBERO default 25)")
    a = ap.parse_args()
    import numpy as np
    from omegaconf import OmegaConf
    from libero.libero import benchmark as bmklib
    from libero.lifelong.utils import get_task_embs

    os.makedirs(a.out_dir, exist_ok=True)
    texts, embs = {}, {}
    for s in SUITES:
        bmk = bmklib.get_benchmark_dict()[s]()
        texts[s] = [bmk.get_task(i).language for i in range(bmk.n_tasks)]
        cfg = OmegaConf.create({"task_embedding_format": "bert", "task_embedding_one_hot_offset": 1,
                                "data": {"max_word_len": a.max_word_len},
                                "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}})
        embs[s] = get_task_embs(cfg, texts[s]).numpy().astype(np.float32)
        print(s, len(texts[s]), "instructions, BERT embeddings", embs[s].shape, "mean norm %.3f" % np.linalg.norm(embs[s], axis=1).mean())
    json.dump(texts, open(os.path.join(a.out_dir, "libero_instructions.json"), "w"), indent=0)
    np.savez(os.path.join(a.out_dir, "bert_pooler_libero.npz"), **embs)
    allb = np.concatenate([embs[s] for s in SUITES])
    print("mean BERT norm over the 40 instructions: %.6f (paper runs: 21.532940)" % np.linalg.norm(allb, axis=1).mean())


if __name__ == "__main__":
    main()
