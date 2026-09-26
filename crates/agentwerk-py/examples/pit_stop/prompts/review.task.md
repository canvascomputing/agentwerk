Review the crew reports and the current state.

Crew reports:
{{ find_results(task.label IN ({{ crew_labels }}) AND task.status = finished ORDER BY task.id)[*].status }}

Before GO:

- Wheels are secured and wings match the requested angles.
- Jacks are lowered and back in storage. All other equipment is stored.
- Everyone is clear, empty-handed, and still.
- Step aside to `chief-clear` first.
