"""
AI ROUTE-PLANNING EMERGENCY RESPONSE AGENT
Student: Goodness Ononogbu
Student ID: s3573368
Module: Artificial Intelligence Foundations (CIS4049-N)

Purpose of this project:
I built an intelligent agent that can find the best possible route on a map
when every second matters — like an ambulance trying to reach a hospital,
or people trying to evacuate to safety.

This code proves:
- BFS explores without strategy
- Dijkstra finds the cheapest path but without guidance
- A* makes every move count using a smart distance estimate (heuristic)

The map is randomly generated using my student ID as initial the seed,
so the world I test in is reproducible and uniquely mine.
"""

from __future__ import annotations

import time
import math
import heapq
import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple, Set, Iterable

Pos = Tuple[int, int]

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class SearchResult:
    algorithm: str
    path: Optional[List[Pos]]
    total_cost: float
    nodes_expanded: int
    runtime_ms: float


# ----------------------------
# Heuristics
# ----------------------------
def manhattan(a: Pos, b: Pos) -> float:
    """Admissible for 4-direction movement with unit step costs."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def octile(a: Pos, b: Pos) -> float:
    """
    Admissible for 8-direction movement with diagonal cost ~sqrt(2).
    Common heuristic in grid pathfinding.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    F = math.sqrt(2) - 1
    return F * min(dx, dy) + max(dx, dy)


# ----------------------------
# Grid generation & rendering
# ----------------------------
def generate_weighted_grid(
    rows: int,
    cols: int,
    obstacle_prob: float,
    heavy_cost_prob: float,
    seed: int,
) -> List[List[int]]:
    """
    Generates a weighted grid:
    - -1 = obstacle
    -  1 = normal road cost
    -  3 = traffic/rough terrain cost (heavier traversal cost)
    """
    rng = random.Random(seed)
    grid: List[List[int]] = []

    for _ in range(rows):
        row = []
        for _ in range(cols):
            if rng.random() < obstacle_prob:
                row.append(-1)
            else:
                # heavier terrain appears with probability heavy_cost_prob
                row.append(3 if rng.random() < heavy_cost_prob else 1)
        grid.append(row)

    return grid

def generate_maze_grid(
    rows: int,
    cols: int,
    heavy_cost_prob: float,
    seed: int,
) -> List[List[int]]:
    """
    Generates a maze-like grid using randomized DFS carving.

    Representation:
    - -1 = wall/obstacle
    -  1 = normal path
    -  3 = heavy/rough/traffic path (optional overlay)

    Notes:
    - Maze carving works best with odd dimensions.
    - If rows/cols are even, we reduce by 1 to keep odd structure.
    """
    rng = random.Random(seed)

    # Force odd dimensions for clean maze paths
    if rows % 2 == 0:
        rows -= 1
    if cols % 2 == 0:
        cols -= 1

    # Start as all walls
    grid = [[-1 for _ in range(cols)] for _ in range(rows)]

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    # Carve passages by jumping 2 cells at a time
    directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]

    # Pick a starting cell in the maze (must be odd index)
    start_r, start_c = 1, 1
    grid[start_r][start_c] = 1

    stack = [(start_r, start_c)]

    while stack:
        r, c = stack[-1]
        rng.shuffle(directions)

        carved = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid[nr][nc] == -1:
                # Carve the wall between (r,c) and (nr,nc)
                wall_r, wall_c = r + dr // 2, c + dc // 2
                grid[wall_r][wall_c] = 1
                grid[nr][nc] = 1
                stack.append((nr, nc))
                carved = True
                break

        if not carved:
            stack.pop()

    # Overlay heavier terrain on some open cells (keeps maze structure)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and rng.random() < heavy_cost_prob:
                grid[r][c] = 3

    return grid

def ensure_start_goal_free(grid: List[List[int]], start: Pos, goal: Pos) -> None:
    """Guarantee start/goal are not obstacles."""
    sr, sc = start
    gr, gc = goal
    if grid[sr][sc] == -1:
        grid[sr][sc] = 1
    if grid[gr][gc] == -1:
        grid[gr][gc] = 1


def render_grid(
    grid: List[List[int]],
    start: Pos,
    goal: Pos,
    path: Optional[List[Pos]] = None,
) -> str:
    """
    ASCII map:
    S = start
    G = goal
    # = obstacle
    . = normal cost cell (1)
    ~ = heavy cost cell (3)
    * = path
    """
    path_set = set(path) if path else set()
    rows, cols = len(grid), len(grid[0])
    lines: List[str] = []

    for r in range(rows):
        chars: List[str] = []
        for c in range(cols):
            p = (r, c)
            if p == start:
                chars.append("S")
            elif p == goal:
                chars.append("G")
            elif p in path_set:
                chars.append("*")
            else:
                v = grid[r][c]
                if v == -1:
                    chars.append("#")
                elif v == 3:
                    chars.append("~")
                else:
                    chars.append(".")
        lines.append(" ".join(chars))
    return "\n".join(lines)

def render_grid_emojis(
    grid: List[List[int]],
    start: Pos,
    goal: Pos,
    path: Optional[List[Pos]] = None,
    agent_pos: Optional[Pos] = None,
    scenario_icon: str = "🚑",
    goal_icon: str = "🏥",
) -> str:
    """
    Emoji grid for Streamlit display:
    - Walls: ⬛
    - Normal: ⬜
    - Heavy: 🟫
    - Start: 🟩
    - Goal: 🏥 / 🛟 etc.
    - Path: 🟦 (optional)
    - Agent: 🚑 or 🏃
    """
    path_set = set(path) if path else set()
    rows, cols = len(grid), len(grid[0])
    lines: List[str] = []

    for r in range(rows):
        row_chars: List[str] = []
        for c in range(cols):
            p = (r, c)
            v = grid[r][c]

            if agent_pos is not None and p == agent_pos:
                row_chars.append(scenario_icon)
            elif p == start:
                row_chars.append("🟩")
            elif p == goal:
                row_chars.append(goal_icon)
            elif p in path_set:
                row_chars.append("🟦")
            else:
                if v == -1:
                    row_chars.append("⬛")
                elif v == 3:
                    row_chars.append("🟫")
                else:
                    row_chars.append("⬜")

        lines.append("".join(row_chars))

    return "\n".join(lines)


def render_grid_arrows(
    grid: List[List[int]],
    start: Pos,
    goal: Pos,
    path: Optional[List[Pos]] = None,
) -> str:
    """
    Cleaner route rendering:
    - Uses arrows to show direction along the path (easy to understand at first glance).
    """
    arrow_for_step = {
        (0, 1): "→",
        (0, -1): "←",
        (1, 0): "↓",
        (-1, 0): "↑",
        (1, 1): "↘",
        (1, -1): "↙",
        (-1, 1): "↗",
        (-1, -1): "↖",
    }

    arrows: Dict[Pos, str] = {}
    if path and len(path) >= 2:
        for i in range(len(path) - 1):
            (r1, c1) = path[i]
            (r2, c2) = path[i + 1]
            dr, dc = (r2 - r1), (c2 - c1)
            arrows[(r1, c1)] = arrow_for_step.get((dr, dc), "*")

    rows, cols = len(grid), len(grid[0])
    lines: List[str] = []

    for r in range(rows):
        chars: List[str] = []
        for c in range(cols):
            p = (r, c)
            if p == start:
                chars.append("S")
            elif p == goal:
                chars.append("G")
            elif p in arrows:
                chars.append(arrows[p])
            else:
                v = grid[r][c]
                if v == -1:
                    chars.append("#")
                elif v == 3:
                    chars.append("~")
                else:
                    chars.append(".")
        lines.append(" ".join(chars))

    return "\n".join(lines)

# ----------------------------
# Movement & costs
# ----------------------------
def neighbors(pos: Pos, rows: int, cols: int, diagonal: bool) -> Iterable[Tuple[Pos, float]]:
    """
    Returns neighbor positions with step multiplier:
    - cardinal step cost multiplier = 1.0
    - diagonal step cost multiplier = sqrt(2)
    """
    r, c = pos
    steps_4 = [((r + 1, c), 1.0), ((r - 1, c), 1.0), ((r, c + 1), 1.0), ((r, c - 1), 1.0)]

    if not diagonal:
        for p, mult in steps_4:
            rr, cc = p
            if 0 <= rr < rows and 0 <= cc < cols:
                yield (p, mult)
        return

    steps_8 = steps_4 + [
        ((r + 1, c + 1), math.sqrt(2)),
        ((r + 1, c - 1), math.sqrt(2)),
        ((r - 1, c + 1), math.sqrt(2)),
        ((r - 1, c - 1), math.sqrt(2)),
    ]

    for p, mult in steps_8:
        rr, cc = p
        if 0 <= rr < rows and 0 <= cc < cols:
            yield (p, mult)


def reconstruct(came_from: Dict[Pos, Pos], start: Pos, goal: Pos) -> List[Pos]:
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


# ----------------------------
# Search algorithms
# ----------------------------
def a_star_search(
    grid: List[List[int]],
    start: Pos,
    goal: Pos,
    diagonal: bool,
    use_heuristic: bool,
) -> SearchResult:
    """
    Unified search:
    - If use_heuristic=True => A*
    - If use_heuristic=False => Dijkstra (heuristic=0)
    Works with weighted terrain.
    """
    rows, cols = len(grid), len(grid[0])
    t0 = time.perf_counter()

    def h(a: Pos, b: Pos) -> float:
        if not use_heuristic:
            return 0.0
        return octile(a, b) if diagonal else manhattan(a, b)

    open_heap: List[Tuple[float, int, Pos]] = []
    tie = 0

    g_score: Dict[Pos, float] = {start: 0.0}
    came_from: Dict[Pos, Pos] = {}

    heapq.heappush(open_heap, (h(start, goal), tie, start))
    visited: Set[Pos] = set()
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct(came_from, start, goal)
            t1 = time.perf_counter()
            return SearchResult(
                algorithm="A*" if use_heuristic else "Dijkstra",
                path=path,
                total_cost=g_score[current],
                nodes_expanded=nodes_expanded,
                runtime_ms=(t1 - t0) * 1000,
            )

        for nb, step_mult in neighbors(current, rows, cols, diagonal):
            rr, cc = nb
            cell_cost = grid[rr][cc]
            if cell_cost == -1:
                continue

            # cost to move = terrain_cost * step_multiplier
            tentative_g = g_score[current] + (cell_cost * step_mult)

            if nb not in g_score or tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                tie += 1
                f = tentative_g + h(nb, goal)
                heapq.heappush(open_heap, (f, tie, nb))

    t1 = time.perf_counter()
    return SearchResult(
        algorithm="A*" if use_heuristic else "Dijkstra",
        path=None,
        total_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        runtime_ms=(t1 - t0) * 1000,
    )


def bfs_unweighted(
    grid: List[List[int]],
    start: Pos,
    goal: Pos,
    diagonal: bool,
) -> SearchResult:
    """
    BFS baseline: treats all traversable cells as equal cost.
    Note: BFS is not cost-optimal for weighted terrain (this is intentional and valuable to discuss).
    """
    rows, cols = len(grid), len(grid[0])
    t0 = time.perf_counter()

    q = deque([start])
    visited: Set[Pos] = {start}
    came_from: Dict[Pos, Pos] = {}
    nodes_expanded = 0

    while q:
        current = q.popleft()
        nodes_expanded += 1

        if current == goal:
            path = reconstruct(came_from, start, goal)
            t1 = time.perf_counter()
            # approximate “cost” as steps for baseline comparability
            return SearchResult(
                algorithm="BFS",
                path=path,
                total_cost=float(len(path) - 1),
                nodes_expanded=nodes_expanded,
                runtime_ms=(t1 - t0) * 1000,
            )

        for nb, _ in neighbors(current, rows, cols, diagonal):
            rr, cc = nb
            if grid[rr][cc] == -1:
                continue
            if nb in visited:
                continue
            visited.add(nb)
            came_from[nb] = current
            q.append(nb)

    t1 = time.perf_counter()
    return SearchResult(
        algorithm="BFS",
        path=None,
        total_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        runtime_ms=(t1 - t0) * 1000,
    )


# ----------------------------
# Scenarios (world-standard framing)
# ----------------------------
@dataclass
class Scenario:
    name: str
    rows: int
    cols: int
    start: Pos
    goal: Pos
    obstacle_prob: float
    heavy_cost_prob: float
    seed: int


def scenario_ambulance(seed: int) -> Scenario:
    """
    Ambulance routing to hospital: moderate obstacles (road blocks),
    moderate traffic zones (heavier cost).
    """
    return Scenario(
        name="Ambulance-to-Hospital",
        rows=15,
        cols=15,
        start=(0, 0),
        goal=(14, 14),
        obstacle_prob=0.18,
        heavy_cost_prob=0.25,
        seed=seed,
    )


def scenario_flood_evac(seed: int) -> Scenario:
    """
    Flood evacuation to shelter: higher obstacles (blocked streets),
    more rough terrain (higher cost).
    """
    return Scenario(
        name="Flood-Evacuation-to-Shelter",
        rows=15,
        cols=15,
        start=(0, 14),
        goal=(14, 0),
        obstacle_prob=0.22,
        heavy_cost_prob=0.35,
        seed=seed,
    )


# ----------------------------
# Experiment runner (report-friendly)
# ----------------------------
def run_experiment(scn: Scenario, diagonal: bool) -> Tuple[List[List[int]], List[SearchResult]]:
    grid = generate_weighted_grid(
        rows=scn.rows,
        cols=scn.cols,
        obstacle_prob=scn.obstacle_prob,
        heavy_cost_prob=scn.heavy_cost_prob,
        seed=scn.seed,
    )
    ensure_start_goal_free(grid, scn.start, scn.goal)

    results: List[SearchResult] = []
    results.append(bfs_unweighted(grid, scn.start, scn.goal, diagonal=diagonal))
    results.append(a_star_search(grid, scn.start, scn.goal, diagonal=diagonal, use_heuristic=False))  # Dijkstra
    results.append(a_star_search(grid, scn.start, scn.goal, diagonal=diagonal, use_heuristic=True))   # A*

    return grid, results


def print_results_table(scn: Scenario, diagonal: bool, results: List[SearchResult]) -> None:
    move = "8-way (diagonal)" if diagonal else "4-way"
    print(f"\n=== Results: {scn.name} | {move} | seed={scn.seed} ===")
    print(f"{'Algorithm':<10} | {'Path?':<5} | {'Cost':<10} | {'Expanded':<9} | {'Runtime (ms)':<12} | {'Path length':<11}")
    print("-" * 78)
    for r in results:
        has_path = "YES" if r.path else "NO"
        cost_str = f"{r.total_cost:.2f}" if math.isfinite(r.total_cost) else "inf"
        path_len = str(len(r.path)) if r.path else "-"
        print(f"{r.algorithm:<10} | {has_path:<5} | {cost_str:<10} | {r.nodes_expanded:<9} | {r.runtime_ms:<12.3f} | {path_len:<11}")


def choose_best_path_for_visual(results: List[SearchResult]) -> Optional[List[Pos]]:
    """
    Prefer A* path for visualization; fallback to Dijkstra then BFS.
    """
    order = ["A*", "Dijkstra", "BFS"]
    by_name = {r.algorithm: r for r in results}
    for name in order:
        r = by_name.get(name)
        if r and r.path:
            return r.path
    return None


# ----------------------------
# Main: run two scenarios + 2 movement modes + 3 trials
# ----------------------------
def main() -> None:
    print("AI Route-Planning Agent: BFS vs Dijkstra vs A* (Weighted Grid)\n")
    print("Legend: S=start | G=goal | #=obstacle | .=cost1 | ~=cost3 | *=path\n")

    # You can change seeds to make it uniquely yours, while keeping reproducibility.
    # Use your student ID digits as part of the seed for originality (e.g., 1234567).
    seeds = [3573368, 3573369, 3573370]

    scenarios = []
    for i, sd in enumerate(seeds):
        scenarios.append(scenario_ambulance(seed=sd))
        scenarios.append(scenario_flood_evac(seed=sd + 7))  # small offset for variety

    # Choose movement: run both 4-way and 8-way to strengthen evaluation
    for diagonal in (False, True):
        for scn in scenarios:
            grid, results = run_experiment(scn, diagonal=diagonal)

            # Print a table (report-ready evidence)
            print_results_table(scn, diagonal, results)

            # Render one visual per scenario (best available algorithm)
            best_path = choose_best_path_for_visual(results)
            print("\nMap snapshot (use this for screenshot evidence):")
            print(render_grid(grid, scn.start, scn.goal, best_path))
            print("\n" + "=" * 80)

    print("\nDone. Tip: Take screenshots of (1) results table and (2) map snapshot outputs for your report.")


if __name__ == "__main__":
    main()

def run_single(scn: Scenario, diagonal: bool, algorithm: str) -> Tuple[List[List[int]], SearchResult]:
    """
    Runs one scenario with one algorithm and returns (grid, result).
    algorithm: "BFS" | "Dijkstra" | "A*"
    """
    grid = generate_weighted_grid(
        rows=scn.rows,
        cols=scn.cols,
        obstacle_prob=scn.obstacle_prob,
        heavy_cost_prob=scn.heavy_cost_prob,
        seed=scn.seed,
    )
    ensure_start_goal_free(grid, scn.start, scn.goal)

    if algorithm == "BFS":
        res = bfs_unweighted(grid, scn.start, scn.goal, diagonal=diagonal)
    elif algorithm == "Dijkstra":
        res = a_star_search(grid, scn.start, scn.goal, diagonal=diagonal, use_heuristic=False)
    else:
        res = a_star_search(grid, scn.start, scn.goal, diagonal=diagonal, use_heuristic=True)

    return grid, res
