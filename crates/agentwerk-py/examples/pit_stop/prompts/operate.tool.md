Pick up, put down, or use equipment where you stand.

- Returns `ok` and an updated `observation`.
- Rejected actions also return a `message`.
- NEVER report a rejected action as completed.

Choose the action with `task`:

- `pickup`:
  - Supply an inventory `item`.
  - You MUST arrive with empty hands.
  - Locate stored items by their current `owner`.
    For `slot:bench-2`, move to `storage:bench-2`.
- `drop`:
  - Supply the held `item` and an empty slot as `target`.
  - Stand at that slot first.
  - At `storage:bench-2`, use `target="bench-2"`.
- `use`:
  - Supply the assigned step as `work` and the assigned `target`.
  - Stand at `work:<role>:<target>`.
  - For `adjust`, also supply the requested angle in degrees as `value`.

Equipment rules:

- NEVER carry more than one item.
- You can use a wheel gun on any wheel.
- Use the fresh tire for your assigned wheel position.
- Store tools on benches and tires on tire platforms.
  No other item may occupy the slot.
- Store unsuitable equipment before collecting a replacement.
- Lift with your assigned jack.
  Lifting mounts it and frees your hands.
- Lower the jack empty-handed.
- During cleanup, pick up the lowered jack at its work position.
  Return it to its designated storage.
- Brace, clear, and remove tires with empty hands.
- Stay braced until assigned to clear.
- Removal leaves you holding the old tire.
- Fitting puts your fresh tire on the car.

Example: you hold `wing-key-5` at an empty `storage:bench-2`.
Call `operate(task="drop", item="wing-key-5", target="bench-2")` to store it there.
