import { actionMotion, workProgress } from "./motion.js";

export function wheelStatus(sample, corner) {
  const status = sample.state.wheels[corner];
  if (status === "secured") return "Done";
  const active = (role, action) =>
    sample.actions[`${role}-${corner}`]?.data.action === action;
  if (active("gunner", "tighten")) return "Tightening";
  if (active("wheel-on", "fit")) return "Fitting";
  if (status === "empty") return "Hub clear";
  if (active("wheel-off", "remove")) return "Removing";
  if (active("gunner", "loosen")) return "Loosening";
  if (status !== "old-secured") return "Ready";
  const prepared = ["gunner", "wheel-on"].every((role) =>
    sample.state.crew[`${role}-${corner}`].done.includes("collect"),
  );
  return prepared ? "Ready" : "Fetching";
}

export function updateServiceDiagram(element, side, sample) {
  const { state, actions, time } = sample;
  const heights = {};
  for (const end of ["front", "rear"]) {
    const action = actions[`jack-${end}`];
    const height = action
      ? action.data.action === "lift"
        ? workProgress(action, time)
        : 1 - workProgress(action, time)
      : state.jacks[end] === "up"
        ? 1
        : 0;
    heights[end] = height;
    side.dataset[end] = height;
    const jack = side.querySelector(`[data-end="${end}"]`);
    jack.setAttribute("opacity", height);
    jack.querySelector("line").setAttribute("y2", 48 - height * 12);
  }
  const lift = (heights.front + heights.rear) / 2;
  side.dataset.lifted = String(lift > 0);
  const tilt =
    (Math.atan2((heights.rear - heights.front) * 12, 98) * 180) / Math.PI;
  side
    .querySelector(".side-car")
    .setAttribute(
      "transform",
      `translate(0 ${-12 * heights.rear}) rotate(${tilt} 40 47)`,
    );
  for (const wheel of [
    ...element.querySelectorAll("[data-corner]"),
    ...side.querySelectorAll("[data-corner]"),
  ]) {
    const corner = wheel.dataset.corner;
    const status = state.wheels[corner];
    const working = ["gunner", "wheel-off", "wheel-on"].some((role) => {
      const action = actions[`${role}-${corner}`];
      if (
        !action ||
        ["collect", "return", "stow", "withdraw"].includes(action.data.action)
      )
        return false;
      return (
        !action.data.phases ||
        ["work", "reach", "pull", "align", "seat"].includes(
          actionMotion(action, time).kind,
        )
      );
    });
    wheel.dataset.status = status;
    wheel.dataset.working = String(working);
    const label = `${corner}: ${wheelStatus(sample, corner)}`;
    wheel.querySelector("title").textContent = label;
    wheel.setAttribute("aria-label", label);
    let detached = status === "empty" ? 1 : 0;
    const removing = actions[`wheel-off-${corner}`];
    const fitting = actions[`wheel-on-${corner}`];
    if (removing?.data.action === "remove")
      detached = workProgress(removing, time);
    if (fitting?.data.action === "fit")
      detached = 1 - workProgress(fitting, time);
    const tire = wheel.querySelector(".mini-tire");
    tire.setAttribute("opacity", 1 - detached);
    tire.setAttribute(
      "transform",
      `translate(0 ${wheel.ownerSVGElement === element ? (corner.endsWith("left") ? -4 : 4) * detached : 0})`,
    );
  }
  element.setAttribute(
    "aria-label",
    `Overhead car; ${Object.entries(state.wheels)
      .map(([corner]) => `${corner} ${wheelStatus(sample, corner)}`)
      .join(", ")}`,
  );
  side.setAttribute(
    "aria-label",
    `Side view: front ${Math.round(heights.front * 100)}% lifted, rear ${Math.round(heights.rear * 100)}% lifted`,
  );
}
