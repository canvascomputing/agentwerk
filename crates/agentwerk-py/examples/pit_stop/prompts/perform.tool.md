Performs the assigned crew action and returns its validated result.

Checks prerequisites before work starts and returns the completed actor and action. A request rejected by prerequisite checks leaves the car unchanged. A hold during an action stops its remaining phases.

Usage:
- Tool and fresh-tire collection can run on the apron during arrival; car service waits for `car_stopped`
- `action`: the action named in your task, selected from the allowed values
- The tool binds your crew identity; no actor parameter is accepted

# Instructions
- Use the assigned action because the host schedules work only when its prerequisites are satisfied
- Call `finish` with the same action after success, because the result hook schedules the next crew member
- NEVER report a rejected action as finished, because dependent work must remain blocked

Example usage:
<example>
user: "You are gunner-front-left. Perform loosen, then finish."
assistant: perform({"action": "loosen"})
</example>
