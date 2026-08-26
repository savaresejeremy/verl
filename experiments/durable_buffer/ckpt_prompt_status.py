# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Count prompt tags by status in a saved TransferQueue controller checkpoint.

The stock-resume arm of the kill drill re-issues exactly those prompts the
checkpoint recorded as pending/running (trainer_base._reissue_inflight_prompts),
so this count is the drill's precondition: with zero in-flight prompts in the
checkpoint, both arms have nothing to re-issue and the two arms are
indistinguishable by construction.
"""

import collections
import json
import pickle
import sys


def scan(o, prompt_status, depth=0):
    if depth > 6:
        return
    if isinstance(o, dict):
        if "is_prompt" in o or "status" in o:
            key = ("prompt" if o.get("is_prompt", False) else "trajectory") + ":" + str(o.get("status"))
            prompt_status[key] += 1
            return
        for v in o.values():
            scan(v, prompt_status, depth + 1)
    elif isinstance(o, list | tuple | set):
        for v in o:
            scan(v, prompt_status, depth + 1)
    elif hasattr(o, "__dict__"):
        scan(vars(o), prompt_status, depth + 1)


out = {}
for spec in sys.argv[1:]:
    label, _, path = spec.partition("=")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    hist = collections.Counter()
    scan(obj, hist)
    inflight = sum(v for k, v in hist.items() if k.startswith("prompt:") and k.split(":")[1] in ("pending", "running"))
    out[label] = {
        "checkpoint": path,
        "tag_histogram": dict(sorted(hist.items())),
        "prompts_inflight_pending_or_running": inflight,
        "global_index_counter": obj.get("index_manager", {}).get("global_index_counter"),
    }
print(json.dumps(out, indent=1))
