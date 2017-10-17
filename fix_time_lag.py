import sys
import json

TIME_KEY = "time elapsed"
STEP_KEY = "step"
EPISODE_KEY = "episodes"

prev_record = None

shifts = {
    TIME_KEY: 0,
    STEP_KEY: 0,
    EPISODE_KEY: 0,
}

step_shift = 0
time_shift = 0
episode_shift = 0
for line in sys.stdin.readlines():
    record = json.loads(line)
    if "steps" in record:
        record[STEP_KEY] = record["steps"]
        record.pop("steps")

    if TIME_KEY in record and EPISODE_KEY in record:
        if prev_record is None:
            prev_record = record
            continue

        if record[STEP_KEY] + shifts[STEP_KEY] <= prev_record[STEP_KEY]:
            # Need to introduce new shift.
            for k in shifts.keys():
                shifts[k] = prev_record[k]

        for k, v in shifts.items():
            record[k] += v

        prev_record = record

    print(json.dumps(record))
