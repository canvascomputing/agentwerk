Review the crew reports and the current state.

Crew reports:
{{ find_results(task.label IN ({{ crew_labels }}) AND task.status = finished ORDER BY task.id)[*].status }}

Before GO:

- Wheels are secured and wings match the requested angles.
- Jacks are lowered and withdrawn from the car.
- The crew is out of the car's path. Equipment storage may still be running.
- You stand at `chief-clear`, out of the car's path.
