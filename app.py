"""
EMERGENCY ROUTE-PLANNING APP (Streamlit Interface)
Student: Goodness Ononogbu | s3573368 | January 2026

This interface exists for one reason: clarity.
When time matters, an intelligent agent should not only work — it should be explainable.

This app lets me:
- choose a scenario (Ambulance routing or Flood evacuation),
- choose a search method (BFS, Dijkstra, A*),
- choose movement style (4-way or 8-way),
- use my student-ID seed for reproducible environments,
- and instantly see both the route and the performance evidence (cost, nodes expanded, runtime).

Academic note:
The algorithms are implemented in s3573368_ononogbu_goodness_ai_route_agent.py; this file focuses on the UI layer only.
"""

import time
import math
import streamlit as st

from s3573368_ononogbu_goodness_ai_route_agent import (
    scenario_ambulance,
    scenario_flood_evac,
    render_grid_emojis,
    generate_maze_grid,
    ensure_start_goal_free,
    run_single,
    bfs_unweighted,
    a_star_search,
)

# ----------------------------
# Streamlit configuration
# ----------------------------
st.set_page_config(page_title="Emergency Route Planner", layout="wide")
st.title("Emergency Route-Planning Agent")
st.caption("BFS vs Dijkstra vs A* — with weighted terrain, reproducible worlds, and an interactive maze view.")


# ----------------------------
# Helper utilities (UI layer)
# ----------------------------
def format_cost(x: float) -> str:
    if not math.isfinite(x):
        return "∞"
    return f"{x:.2f}"


def scenario_icons(scenario_name: str) -> tuple[str, str]:
    """Returns (agent_icon, goal_icon) based on scenario."""
    if scenario_name == "Ambulance-to-Hospital":
        return "🚑", "🏥"
    return "🏃", "🛟"


def summarize_directions(path):
    """Turns a path into a short human-readable directions string."""
    if not path or len(path) < 2:
        return "No route available."

    def move_name(dr, dc):
        return {
            (0, 1): "Right",
            (0, -1): "Left",
            (1, 0): "Down",
            (-1, 0): "Up",
            (1, 1): "Down-Right",
            (1, -1): "Down-Left",
            (-1, 1): "Up-Right",
            (-1, -1): "Up-Left",
        }.get((dr, dc), "Step")

    moves = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        moves.append((r2 - r1, c2 - c1))

    out = []
    count = 1
    for i in range(1, len(moves) + 1):
        if i < len(moves) and moves[i] == moves[i - 1]:
            count += 1
        else:
            dr, dc = moves[i - 1]
            out.append(f"{move_name(dr, dc)} × {count}")
            count = 1

    return " → ".join(out)


def build_start_goal_for_maze(scenario_name: str, rows: int, cols: int):
    """Places start/goal inside the maze corridors."""
    if scenario_name == "Ambulance-to-Hospital":
        return (1, 1), (rows - 2, cols - 2)
    return (1, cols - 2), (rows - 2, 1)


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")

    scenario_name = st.selectbox(
        "Scenario",
        ["Ambulance-to-Hospital", "Flood-Evacuation-to-Shelter"]
    )

    algorithm = st.selectbox(
        "Algorithm",
        ["A*", "Dijkstra", "BFS"]
    )

    diagonal = st.toggle("Allow diagonal movement (8-way)", value=True)

    seed = st.number_input(
        "Seed (use your student ID for reproducibility)",
        value=3573368,
        step=1
    )

    world_type = st.selectbox("World type", ["Maze (interactive)", "Random grid (baseline)"])

    st.divider()
    show_path_fill = st.toggle("Highlight full path", value=True)
    animate = st.toggle("Animate movement", value=True)
    speed = st.slider("Animation speed", 0.01, 0.25, 0.06)

    st.divider()
    regenerate = st.button("Regenerate maze (same seed)")
    run_btn = st.button("Run route planning")


# ----------------------------
# World persistence (maze only)
# ----------------------------
if "maze_grid" not in st.session_state:
    st.session_state.maze_grid = None
if "maze_meta" not in st.session_state:
    st.session_state.maze_meta = {}

def generate_maze_world():
    grid = generate_maze_grid(rows=15, cols=15, heavy_cost_prob=0.30, seed=int(seed))
    rows, cols = len(grid), len(grid[0])
    start, goal = build_start_goal_for_maze(scenario_name, rows, cols)
    ensure_start_goal_free(grid, start, goal)
    st.session_state.maze_grid = grid
    st.session_state.maze_meta = {"start": start, "goal": goal, "rows": rows, "cols": cols}

# Generate maze at startup if user selected Maze mode
if world_type == "Maze (interactive)" and st.session_state.maze_grid is None:
    generate_maze_world()

# Regenerate when requested
if regenerate and world_type == "Maze (interactive)":
    generate_maze_world()


# ----------------------------
# Main run
# ----------------------------
if not run_btn:
    st.info("Choose your settings in the sidebar and click **Run route planning**.")
else:
    agent_icon, goal_icon = scenario_icons(scenario_name)

    # Build scenario object
    scn = scenario_ambulance(seed=int(seed)) if scenario_name == "Ambulance-to-Hospital" else scenario_flood_evac(seed=int(seed))

    # Maze vs Random logic
    if world_type == "Maze (interactive)":
        grid = st.session_state.maze_grid
        start = st.session_state.maze_meta["start"]
        goal = st.session_state.maze_meta["goal"]

        if diagonal:
            st.warning("Maze mode is designed for 4-way movement for cleaner corridors. I will run 4-way for this maze.")
        diag_for_run = False

        if algorithm == "BFS":
            result = bfs_unweighted(grid, start, goal, diagonal=diag_for_run)
        elif algorithm == "Dijkstra":
            result = a_star_search(grid, start, goal, diagonal=diag_for_run, use_heuristic=False)
        else:
            result = a_star_search(grid, start, goal, diagonal=diag_for_run, use_heuristic=True)

        scn.start, scn.goal = start, goal
        movement_label = "4-way"

    else:
        grid, result = run_single(scn, diagonal=diagonal, algorithm=algorithm)
        movement_label = "8-way" if diagonal else "4-way"

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Evidence (Metrics)")
        st.write(f"**Scenario:** {scn.name}")
        st.write(f"**Algorithm:** {result.algorithm}")
        st.write(f"**Movement:** {movement_label}")
        st.write(f"**Path found:** {'Yes' if result.path else 'No'}")
        st.write(f"**Total cost:** {format_cost(result.total_cost)}")
        st.write(f"**Nodes expanded:** {result.nodes_expanded}")
        st.write(f"**Runtime:** {result.runtime_ms:.3f} ms")
        st.write(f"**Path length:** {len(result.path) if result.path else '-'}")

        if result.path:
            st.divider()
            st.subheader("Route Directions (Readable Summary)")
            st.write(summarize_directions(result.path))

        st.divider()
        st.write(
            "Interpretation: BFS is a useful baseline, but it is not cost-aware. "
            "Dijkstra guarantees the lowest-cost route in weighted terrain. "
            "A* aims to find the same optimal route with fewer expansions by using a heuristic."
        )

    with right:
        st.subheader("Map View (Live Agent)")

        if not result.path:
            st.error("No route found. Try regenerating the maze, changing the seed, or switching algorithm.")
        else:
            full_path = result.path if show_path_fill else None
            stage = st.empty()

            if animate:
                for step in result.path:
                    stage.code(
                        render_grid_emojis(
                            grid,
                            scn.start,
                            scn.goal,
                            path=full_path,
                            agent_pos=step,
                            scenario_icon=agent_icon,
                            goal_icon=goal_icon,
                        ),
                        language="text",
                    )
                    time.sleep(speed)
            else:
                stage.code(
                    render_grid_emojis(
                        grid,
                        scn.start,
                        scn.goal,
                        path=full_path,
                        agent_pos=result.path[-1],
                        scenario_icon=agent_icon,
                        goal_icon=goal_icon,
                    ),
                    language="text",
                )
