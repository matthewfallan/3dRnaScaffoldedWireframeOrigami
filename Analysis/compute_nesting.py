import pandas as pd
import sys

dot_file = sys.argv[1]
tsv_file = sys.argv[2]

OPEN = "([{<"
CLOSE = ")]}>"
UNPAIRED = "."

with open(dot_file) as f:
    header = f.readline().strip()
    seq = f.readline().strip()
    dot = f.readline().strip()
    depth = list()
    apex_depth = list()
    curr_depth = 0
    curr_apex_depth = -1
    direction = OPEN
    for x in dot:
        if x in OPEN:
            if direction == CLOSE:
                assert curr_apex_depth > 0
                apex_depth.extend([curr_apex_depth] * (len(depth) - len(apex_depth)))
                direction = OPEN
            curr_depth += 1
        elif x in CLOSE:
            if direction == OPEN:
                curr_apex_depth = curr_depth
                direction = CLOSE
            curr_depth -= 1
        elif x != UNPAIRED:
            raise ValueError(x)
        if curr_depth < 0:
            raise ValueError("Mismatched open and close marks")
        depth.append(curr_depth)
    if curr_depth != 0:
        raise ValueError("Mismatched open and close marks")
    apex_depth.extend([curr_apex_depth] * (len(depth) - len(apex_depth)))
nesting = pd.DataFrame.from_dict({
    "depth": pd.Series(depth, index=list(range(1, len(depth) + 1))),
    "apex depth": pd.Series(apex_depth, index=list(range(1, len(depth) + 1))),
})
nesting.to_csv(tsv_file, sep="\t")

