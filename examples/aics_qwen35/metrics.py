# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pure data reduction for host-observed streaming and numerical comparisons."""

from __future__ import annotations

import math


class StreamRecord:
    """Collect delta-ID frames from Engine with stream_output=True.

    Frame arrivals can contain several tokens. They cannot resolve individual
    token latency inside that frame; retain counts instead of inventing times.
    """

    def __init__(self, *, start_s: float):
        self.start_s = start_s
        self.output_ids: list[int] = []
        self.token_logprobs: list[float] = []
        self.frames: list[dict] = []
        self.final_meta: dict = {}
        self.text = ""

    def add(self, frame: dict, *, arrival_s: float) -> None:
        ids = [int(token_id) for token_id in frame["output_ids"]]
        meta = frame["meta_info"]
        entries = meta.get("output_token_logprobs", [])
        if entries and [int(entry[1]) for entry in entries] != ids:
            raise ValueError("Frame logprob token IDs do not match delta output IDs")
        self.output_ids.extend(ids)
        self.token_logprobs.extend(float(entry[0]) for entry in entries)
        if int(meta["completion_tokens"]) != len(self.output_ids):
            raise ValueError(
                "Stream count mismatch: expected delta, not cumulative IDs"
            )
        self.frames.append(
            {
                "arrival_s": arrival_s - self.start_s,
                "new_tokens": len(ids),
                "cumulative_tokens": len(self.output_ids),
            }
        )
        self.final_meta = meta
        self.text = frame.get("text", "")

    def finish(self, *, end_s: float) -> dict:
        token_frames = [frame for frame in self.frames if frame["new_tokens"]]
        first = token_frames[0] if token_frames else None
        last = token_frames[-1] if token_frames else None
        count = len(self.output_ids)
        # TTFT is only an observed bound if the first delivery was coalesced.
        tpot_resolved = first is not None and first["new_tokens"] == 1 and count > 1
        return {
            "output_ids": self.output_ids,
            "actual_output_tokens": count,
            "token_logprobs": self.token_logprobs,
            "text": self.text,
            "finish_reason": self.final_meta.get("finish_reason"),
            "reported_prompt_tokens": self.final_meta.get("prompt_tokens"),
            "cached_tokens": self.final_meta.get("cached_tokens"),
            "request_duration_s": end_s - self.start_s,
            "observed_ttft_s": first["arrival_s"] if first else None,
            "first_delivery_token_count": first["new_tokens"] if first else 0,
            "observed_decode_span_s": (
                last["arrival_s"] - first["arrival_s"] if first else None
            ),
            "observed_tpot_s": (
                (last["arrival_s"] - first["arrival_s"]) / (count - 1)
                if tpot_resolved
                else None
            ),
            "all_deliveries_single_token": all(
                frame["new_tokens"] == 1 for frame in token_frames
            ),
            "frames": self.frames,
        }


def compare_sequences(native: dict, reference: dict, *, logprob_atol: float) -> dict:
    """Compare logprobs only when the conditioning history and scored ID agree."""
    if native["input_ids"] != reference["input_ids"]:
        raise ValueError("Input token IDs differ between native and reference")
    native_ids = native["output_ids"]
    ref_ids = reference["output_ids"]
    if not native_ids or not ref_ids:
        raise ValueError("Cannot validate an empty generated sequence")
    prefix = 0
    for left, right in zip(native_ids, ref_ids):
        if left != right:
            break
        prefix += 1
    forced = reference["trajectory"] == "native_teacher_forced"
    if forced and native_ids != ref_ids:
        raise ValueError("Teacher-forced reference scored a different trajectory")
    score_count = len(native_ids) if forced else prefix
    native_lp = native["token_logprobs"]
    ref_lp = reference["token_logprobs"]
    complete = len(native_lp) == len(native_ids) and len(ref_lp) == len(ref_ids)
    diffs = [
        abs(float(left) - float(right))
        for left, right in zip(native_lp[:score_count], ref_lp[:score_count])
    ]
    finite = all(math.isfinite(float(value)) for value in native_lp + ref_lp)
    greedy_ids = reference["greedy_ids"]
    greedy_matches = sum(left == right for left, right in zip(native_ids, greedy_ids))
    same_length = len(native_ids) == len(ref_ids)
    full_greedy_agreement = same_length and greedy_matches == len(native_ids)
    max_diff = max(diffs) if diffs else None
    return {
        "case_id": native["case_id"],
        "run_id": native["run_id"],
        "reference_trajectory": reference["trajectory"],
        "native_output_tokens": len(native_ids),
        "reference_output_tokens": len(ref_ids),
        "same_output_length": same_length,
        "common_output_prefix_tokens": prefix,
        "first_output_mismatch_step": (
            prefix if prefix < min(len(native_ids), len(ref_ids)) else None
        ),
        "greedy_token_agreement_count": greedy_matches,
        "full_greedy_token_agreement": full_greedy_agreement,
        "comparable_logprob_count": len(diffs),
        "logprobs_complete_and_finite": complete and finite,
        "absolute_logprob_errors": diffs,
        "max_absolute_logprob_error": max_diff,
        "logprob_atol": logprob_atol,
        "all_native_steps_within_logprob_tolerance": (
            complete
            and finite
            and len(diffs) == len(native_ids)
            and max_diff is not None
            and max_diff <= logprob_atol
        ),
        "strict_pass": (
            complete
            and finite
            and full_greedy_agreement
            and len(diffs) == len(native_ids)
            and max_diff is not None
            and max_diff <= logprob_atol
        ),
    }
