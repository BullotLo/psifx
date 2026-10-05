"""Shared runtime constants."""

import os
import re

# Allow overriding the default SAM3 model path without editing source files.
SAM3_PATH = os.environ.get("SAM3_PATH", "facebook/sam3")

# Constants used in masks id correction after tracking.
DEFAULT_MIN_FRAGMENT_FRAMES = 8
DEFAULT_MERGE_THRESHOLD = 0.6
DEFAULT_SIGNATURE_SAMPLES = 5
DEFAULT_POOLED_SAMPLES_PER_MEMBER = 5

# --- zeroth pass: same-time overlap (see module docstring) ---
# real pixel-to-pixel minimum distance between two masks, in px --
# small = the two masks' silhouettes actually touch/nearly touch.
DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD = 15.0
# distance between the two masks' centroids, in px -- the discriminant
# that actually separates same-body splits from different-people
# occlusions. Original calibration (4 hand-checked pairs): positives
# under ~160px, negatives over ~170px. LOWERED to 115.0 after a full
# stress test surfaced a false positive at 130.3px (2 different real
# people) sitting below the old 160px threshold but above the highest
# known true positive (81.3px) -- 115.0 is the midpoint of that gap.
# Re-validated 2026-09-10 on 29 real labeled examples across 5 full
# sessions: safe range now 82-119.9px (margin narrowed to 4.9px on the
# negative side, no violation so far -- see the classifier docstring in
# classifier/train_overlap_classifier.py for the data-driven angle on
# this same threshold).
DEFAULT_OVERLAP_CENTROID_THRESHOLD = 115.0
# minimum number of frames where BOTH ids are simultaneously active,
# for a pair's median to be trusted at all -- same role as
# min_fragment_frames for pass 1/2.
DEFAULT_MIN_OVERLAP_FRAMES = 8

MAX_POOLED_SEARCH_RADIUS = 200
DEFAULT_MASK_THRESHOLD = 127
_IMPOSSIBLE_COST = 10.0

_MASK_FILENAME_RE = re.compile(r"^(\d+)\.mp4$")
