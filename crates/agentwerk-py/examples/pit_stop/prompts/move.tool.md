Move to a position in the task's `destinations`.

- Returns `ok` and an updated `observation`.
- Rejected moves also return a `message`.

Travel rules:

- Walking is 1.6 m/s and running is 3.2 m/s.
  Carrying equipment slows travel.
- Run if walking would delay the car or a waiting worker.
- Use the arrival time and travel estimates supplied with your task.
- IMPORTANT: Estimates exclude equipment collection, handling, and traffic.
- Walk when time allows, including cleanup.
- Avoid positions listed in `busy_destinations`.
  If blocked, try another available position.
  Then return to the required destination.
- `operate` does not move you between positions.

For example, `move(destination="storage:bench-1", pace="walk")` walks to bench-1.
