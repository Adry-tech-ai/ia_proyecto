import tkinter as tk
from tkinter import ttk, messagebox
import random
import heapq
import math
from dataclasses import dataclass
from collections import defaultdict

DIRECTIONS = ("Norte", "Sur", "Este", "Oeste")
PHASES = ("NS", "EO")
PHASE_DIRS = {"NS": (0, 1), "EO": (2, 3)}
MIN_GREEN = 6
MAX_GREEN = 24
MAX_QUEUE = 12
DURATIONS = tuple(range(MIN_GREEN, MAX_GREEN + 1))


@dataclass
class Observation:
    queue: int
    oldest_wait: int
    arrivals_last_tick: int


class AccessAgent:
    def __init__(self, direction):
        self.direction = direction
        self.queue = 0
        self.waiting_times = []
        self.completed_wait_times = []
        self.arrivals_last_tick = 0
        self.total_served = 0
        self.total_arrived = 0

    def observe(self):
        oldest = max(self.waiting_times) if self.waiting_times else 0
        return Observation(self.queue, oldest, self.arrivals_last_tick)

    def receive_arrivals(self, count):
        accepted = min(max(count, 0), MAX_QUEUE - self.queue)
        self.arrivals_last_tick = accepted
        self.queue += accepted
        self.total_arrived += accepted
        self.waiting_times.extend([0] * accepted)
        return accepted

    def advance_waiting_time(self):
        self.waiting_times = [w + 1 for w in self.waiting_times]

    def discharge(self, capacity=1):
        served = min(capacity, self.queue)
        for _ in range(served):
            if self.waiting_times:
                self.completed_wait_times.append(self.waiting_times.pop(0))
        self.queue -= served
        self.total_served += served
        return served

    def reset(self):
        self.queue = 0
        self.waiting_times.clear()
        self.completed_wait_times.clear()
        self.arrivals_last_tick = 0
        self.total_served = 0
        self.total_arrived = 0


class BayesianCongestionModel:
    STATES = ("Baja", "Media", "Alta")

    def estimate(self, queue):
        prior = {"Baja": 1/3, "Media": 1/3, "Alta": 1/3}

        likelihood = {
            "Baja":  math.exp(-((queue - 2) ** 2) / (2 * 2.5 ** 2)),
            "Media": math.exp(-((queue - 6) ** 2) / (2 * 2.8 ** 2)),
            "Alta":  math.exp(-((queue - 11) ** 2) / (2 * 2.5 ** 2)),
        }
        unnormalized = {s: prior[s] * likelihood[s] for s in self.STATES}
        total = sum(unnormalized.values()) or 1.0
        posterior = {s: unnormalized[s] / total for s in self.STATES}
        label = max(posterior, key=posterior.get)
        return label, posterior


class TemporalDemandModel:
    TRANSITIONS = {
        "Baja":  {"Baja": 0.70, "Media": 0.25, "Alta": 0.05},
        "Media": {"Baja": 0.20, "Media": 0.60, "Alta": 0.20},
        "Alta":  {"Baja": 0.08, "Media": 0.27, "Alta": 0.65},
    }

    def predict_arrivals(self, current_state):
        probabilities = self.TRANSITIONS[current_state]
        expected = (
            probabilities["Baja"] * 0.15
            + probabilities["Media"] * 0.45
            + probabilities["Alta"] * 0.85
        )
        return expected, probabilities


class TrafficCSP:
    def valid(self, phase, duration):
        if phase not in PHASES:
            return False
        if not (MIN_GREEN <= duration <= MAX_GREEN):
            return False
        return True

    def validate_plan(self, plan):
        return all(self.valid(phase, duration) for phase, duration in plan)


def apply_action(queues, phase, duration, arrival_rate=0.20):
    q = list(queues)
    for i in range(4):
        q[i] += int(round(arrival_rate * duration / 2))

    capacity = duration // 2
    for i in PHASE_DIRS[phase]:
        q[i] = max(0, q[i] - capacity)
    return tuple(q)


def state_cost(queues, duration):
    return duration * sum(queues) + 2 * sum(q * q for q in queues)


def heuristic(queues):
    return sum(queues) * 4


def search_plan(initial_queues, algorithm="astar", depth=3):
    start = (tuple(initial_queues), None, 0)
    frontier = [(0, 0, start, [])]
    counter = 0
    best = {}
    while frontier:
        priority, cost_so_far, state, path = heapq.heappop(frontier)
        queues, previous_phase, level = state

        if level >= depth:
            if path:
                return path[0], cost_so_far

        key = (queues, previous_phase, level)
        if key in best and best[key] <= cost_so_far:
            continue
        best[key] = cost_so_far

        for phase in PHASES:
            if phase == previous_phase:
                continue
            for duration in DURATIONS:
                next_queues = apply_action(queues, phase, duration)
                step_cost = state_cost(next_queues, duration)
                new_cost = cost_so_far + step_cost
                new_state = (next_queues, phase, level + 1)
                new_path = path + [(phase, duration)]
                counter += 1
                extra = heuristic(next_queues) if algorithm == "astar" else 0
                heapq.heappush(
                    frontier,
                    (new_cost + extra, new_cost, new_state, new_path)
                )
    return (("NS", 12), float("inf"))


def evaluate_queues(queues):
    return sum(queues) + 0.12 * sum(q * q for q in queues)


def minimax_value(queues, phase, depth, adversary_turn, alpha, beta):
    if depth == 0:
        return evaluate_queues(queues)

    if adversary_turn:
        value = float("-inf")
        for arrival_rate in (0.35, 0.65):
            next_q = tuple(
                q + int(round(arrival_rate * 8 / 2)) for q in queues
            )
            value = max(
                value,
                minimax_value(next_q, phase, depth - 1, False, alpha, beta)
            )
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value

    value = float("inf")
    for next_phase in PHASES:
        if next_phase == phase:
            continue
        for duration in DURATIONS:
            next_q = apply_action(queues, next_phase, duration, 0.20)
            value = min(
                value,
                minimax_value(next_q, next_phase, depth - 1,
                              True, alpha, beta)
            )
            beta = min(beta, value)
            if beta <= alpha:
                break
    return value


def minimax_choice(queues, current_phase):
    best_action = ("NS", 12)
    best_value = float("inf")
    for phase in PHASES:
        if phase == current_phase:
            continue
        for duration in DURATIONS:
            next_q = apply_action(queues, phase, duration, 0.20)
            value = minimax_value(
                next_q, phase, 2, True, float("-inf"), float("inf")
            )
            if value < best_value:
                best_value = value
                best_action = (phase, duration)
    return best_action, best_value


def evaluate_durations(durations, queues, urgency=(0, 0, 0, 0),
                       predicted=(0, 0, 0, 0)):
    q = [queues[i] + predicted[i] for i in range(4)]
    for phase_index, phase in enumerate(PHASES):
        duration = durations[phase_index]
        q = list(apply_action(tuple(q), phase, duration, 0.20))

    imbalance = abs(sum(q[:2]) - sum(q[2:]))
    starvation = sum(urgency[i] * q[i] for i in range(4))
    return sum(q) * 8 + sum(x*x for x in q) + imbalance * 1.2 + starvation * 0.8


def genetic_optimization(queues, generations=22, population_size=18,
                         urgency=(0, 0, 0, 0), predicted=(0, 0, 0, 0)):
    population = [
        [random.randint(MIN_GREEN, MAX_GREEN),
         random.randint(MIN_GREEN, MAX_GREEN)]
        for _ in range(population_size)
    ]

    for _ in range(generations):
        population.sort(
            key=lambda x: evaluate_durations(x, queues, urgency, predicted)
        )
        survivors = population[:max(2, population_size // 3)]
        new_population = survivors[:]
        while len(new_population) < population_size:
            p1, p2 = random.sample(survivors, 2)
            child = [p1[0], p2[1]]
            if random.random() < 0.35:
                index = random.randrange(2)
                if random.random() < 0.65:
                    child[index] += random.choice((-3, -2, -1, 1, 2, 3))
                else:
                    child[index] = random.randint(MIN_GREEN, MAX_GREEN)
                child[index] = max(MIN_GREEN, min(MAX_GREEN, child[index]))
            new_population.append(child)
        population = new_population

    return min(
        population,
        key=lambda x: evaluate_durations(x, queues, urgency, predicted)
    )


def local_search(durations, queues, urgency=(0, 0, 0, 0),
                 predicted=(0, 0, 0, 0)):
    current = [
        max(MIN_GREEN, min(MAX_GREEN, int(durations[0]))),
        max(MIN_GREEN, min(MAX_GREEN, int(durations[1])))
    ]
    current_cost = evaluate_durations(current, queues, urgency, predicted)
    improved = True
    while improved:
        improved = False
        for index in range(2):
            for delta in (-2, -1, 1, 2):
                neighbor = current[:]
                neighbor[index] = max(
                    MIN_GREEN, min(MAX_GREEN, neighbor[index] + delta)
                )
                cost = evaluate_durations(
                    neighbor, queues, urgency, predicted
                )
                if cost + 1e-9 < current_cost:
                    current, current_cost = neighbor, cost
                    improved = True
    return current, current_cost


class CoordinatorAgent:
    def __init__(self):
        self.bayes = BayesianCongestionModel()
        self.temporal = TemporalDemandModel()
        self.csp = TrafficCSP()
        self.last_decision = "Esperando observaciones..."
        self.last_details = ""

    def decide(self, observations, current_phase):
        queues = tuple(o.queue for o in observations)
        oldest_waits = tuple(o.oldest_wait for o in observations)

        congestion = []
        predictions = []
        predicted_queues = []
        for queue in queues:
            label, posterior = self.bayes.estimate(queue)
            congestion.append(label)
            expected, probabilities = self.temporal.predict_arrivals(label)
            predictions.append(expected)
            predicted_queues.append(expected * 4.0)

        urgency = tuple(min(wait / 12.0, 2.5) for wait in oldest_waits)
        forecast = tuple(
            min(MAX_QUEUE, queues[i] + predicted_queues[i])
            for i in range(4)
        )

        astar_action, astar_cost = search_plan(forecast, "astar", depth=3)
        ucs_action, ucs_cost = search_plan(forecast, "ucs", depth=3)

        robust_action, robust_cost = minimax_choice(forecast, current_phase)

        genetic = genetic_optimization(
            forecast, urgency=urgency, predicted=(0, 0, 0, 0)
        )
        optimized, genetic_cost = local_search(
            genetic, forecast, urgency=urgency
        )

        candidates = [
            astar_action,
            ucs_action,
            robust_action,
            ("NS", optimized[0]),
            ("EO", optimized[1]),
        ]

        def action_score(action):
            phase, duration = action
            next_q = apply_action(forecast, phase, duration, 0.20)
            queue_cost = sum(next_q) * 7 + sum(q*q for q in next_q) * 1.4
            wait_cost = 0.0
            for i in PHASE_DIRS[phase]:
                wait_cost += urgency[i] * max(0, next_q[i]) * 0.35
            for i in PHASE_DIRS["EO" if phase == "NS" else "NS"]:
                wait_cost += urgency[i] * max(0, next_q[i]) * 1.1
            duration_cost = duration * 0.30
            return queue_cost + wait_cost + duration_cost

        candidates = [
            action for action in candidates
            if action[0] != current_phase
        ]
        if not candidates:
            candidates = [("EO" if current_phase == "NS" else "NS", 12)]

        phase, duration = min(candidates, key=action_score)
        duration = max(MIN_GREEN, min(MAX_GREEN, int(duration)))

        plan = [(phase, duration)]
        if not self.csp.validate_plan(plan):
            phase, duration = ("EO" if current_phase == "NS" else "NS"), 12

        self.last_decision = f"Fase {phase} durante {duration} s"
        self.last_details = (
            f"Congestion: {', '.join(congestion)} | "
            f"Espera maxima por via: {list(oldest_waits)} s | "
            f"UCS: {ucs_action[0]} {ucs_action[1]} s ({ucs_cost:.0f}) | "
            f"A*: {astar_action[0]} {astar_action[1]} s ({astar_cost:.0f}) | "
            f"Minimax: {robust_action[0]} {robust_action[1]} s "
            f"({robust_cost:.1f}) | GA+local: {optimized} "
            f"(costo {genetic_cost:.1f}) | "
            f"Prediccion Markov: {[round(x, 2) for x in predictions]}"
        )
        return phase, duration


class TrafficSimulation:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Control semaforico multiagente")
        self.root.geometry("1240x850")
        self.root.minsize(1000, 720)

        self.agents = [AccessAgent(d) for d in DIRECTIONS]
        self.coordinator = CoordinatorAgent()
        self.phase = "NS"
        self.remaining = 8
        self.elapsed = 0
        self.running = False
        self.mode = tk.StringVar(master=self.root, value="Adaptativo")
        self.fixed_duration = 12
        self.total_wait = 0
        self.wait_samples = 0
        self.history_served = 0
        self.after_id = None

        self.build_ui()
        self.refresh()

    def build_ui(self):
        self.root.configure(bg="#eef2f7")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabel", background="#eef2f7", foreground="#172033")
        style.configure("Title.TLabel", font=("Arial", 19, "bold"),
                        foreground="#13233f")
        style.configure("Sub.TLabel", font=("Arial", 10),
                        foreground="#52627a")
        style.configure("TButton", font=("Arial", 10, "bold"), padding=7)

        header = ttk.Frame(self.root, padding=(18, 14, 18, 6))
        header.pack(fill="x")
        ttk.Label(header, text="CENTRO DE CONTROL DE TRAFICO",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Simulacion de una interseccion de cuatro vias · capacidad maxima: 12 vehiculos por via",
            style="Sub.TLabel"
        ).pack(anchor="w", pady=(3, 0))

        controls = ttk.Frame(self.root, padding=(18, 10, 18, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="Modo de control:").pack(side="left")
        mode_box = ttk.Combobox(
            controls, textvariable=self.mode,
            values=("Adaptativo", "Tiempos fijos"),
            state="readonly", width=17
        )
        mode_box.pack(side="left", padx=(7, 16))
        mode_box.bind("<<ComboboxSelected>>", lambda e: self.reset())

        ttk.Button(controls, text="▶  Iniciar", command=self.start).pack(side="left", padx=4)
        ttk.Button(controls, text="Ⅱ  Pausar", command=self.pause).pack(side="left", padx=4)
        ttk.Button(controls, text="↻  Reiniciar", command=self.reset).pack(side="left", padx=4)

        legend = ttk.Frame(self.root, padding=(18, 0, 18, 8))
        legend.pack(fill="x")
        tk.Label(legend, text="  VERDE = VIA HABILITADA  ",
                 bg="#d9f7e5", fg="#17643a",
                 font=("Arial", 9, "bold"), padx=7, pady=4).pack(side="left", padx=(0, 7))
        tk.Label(legend, text="  ROJO = VIA DETENIDA  ",
                 bg="#ffe1e1", fg="#a52c2c",
                 font=("Arial", 9, "bold"), padx=7, pady=4).pack(side="left", padx=7)
        tk.Label(legend, text="  MAXIMO 12 VEHICULOS POR VIA  ",
                 bg="#e1eaff", fg="#254b91",
                 font=("Arial", 9, "bold"), padx=7, pady=4).pack(side="left", padx=7)

        self.canvas = tk.Canvas(
            self.root, bg="#e8edf4", highlightthickness=0,
            height=590
        )
        self.canvas.pack(fill="both", expand=True, padx=14, pady=(0, 8))

        footer = tk.Frame(self.root, bg="#ffffff", padx=16, pady=11,
                          highlightthickness=1, highlightbackground="#d9e1ec")
        footer.pack(fill="x", padx=14, pady=(0, 14))
        self.status = tk.Label(
            footer, text="", bg="#ffffff", fg="#26364d",
            font=("Arial", 10, "bold"), anchor="w"
        )
        self.status.pack(fill="x")
        self.decision = tk.Label(
            footer, text="", bg="#ffffff", fg="#52627a",
            font=("Arial", 9), anchor="w"
        )
        self.decision.pack(fill="x", pady=(5, 0))

    def start(self):
        if not self.running:
            self.running = True
            self.tick()

    def pause(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.refresh()

    def reset(self):
        self.pause()
        for agent in self.agents:
            agent.reset()
        self.phase = "NS"
        self.remaining = self.fixed_duration
        self.elapsed = 0
        self.total_wait = 0
        self.wait_samples = 0
        self.history_served = 0
        self.coordinator = CoordinatorAgent()
        self.refresh()

    def choose_next_phase(self):
        if self.mode.get() == "Tiempos fijos":
            self.phase = "EO" if self.phase == "NS" else "NS"
            self.remaining = self.fixed_duration
            self.coordinator.last_decision = (
                f"Control fijo: fase {self.phase} durante "
                f"{self.fixed_duration} s"
            )
            self.coordinator.last_details = (
                "El controlador no utiliza la demanda para modificar "
                "la duracion."
            )
        else:
            observations = [agent.observe() for agent in self.agents]
            self.phase, self.remaining = self.coordinator.decide(
                observations, self.phase
            )

    def tick(self):
        if not self.running:
            return

        self.elapsed += 1

        arrival_probabilities = (0.42, 0.30, 0.48, 0.25)
        for agent, probability in zip(self.agents, arrival_probabilities):
            arrivals = 1 if random.random() < probability else 0
            if arrivals and random.random() < 0.08:
                arrivals += 1
            agent.advance_waiting_time()
            agent.receive_arrivals(arrivals)

        if self.elapsed % 2 == 0:
            for index in PHASE_DIRS[self.phase]:
                self.history_served += self.agents[index].discharge(1)

        self.total_wait += sum(
            sum(agent.waiting_times) for agent in self.agents
        )
        self.wait_samples += sum(agent.queue for agent in self.agents)

        self.remaining -= 1
        if self.remaining <= 0:
            self.choose_next_phase()

        self.refresh()
        self.after_id = self.root.after(450, self.tick)

    def draw_intersection(self):
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 900)
        h = max(c.winfo_height(), 560)
        cx, cy = w // 2, h // 2
        road_w = 230
        road = "#343d4b"
        asphalt_edge = "#263140"

        c.create_rectangle(0, 0, w, h, fill="#e8edf4", outline="")
        c.create_rectangle(cx-road_w//2-5, 0, cx+road_w//2+5, h,
                           fill=asphalt_edge, outline="")
        c.create_rectangle(0, cy-road_w//2-5, w, cy+road_w//2+5,
                           fill=asphalt_edge, outline="")
        c.create_rectangle(cx-road_w//2, 0, cx+road_w//2, h,
                           fill=road, outline="")
        c.create_rectangle(0, cy-road_w//2, w, cy+road_w//2,
                           fill=road, outline="")

        for y in range(0, h, 42):
            c.create_line(cx, y, cx, y+21, fill="#d9e0e8", width=3)
        for x in range(0, w, 42):
            c.create_line(x, cy, x+21, cy, fill="#d9e0e8", width=3)

        for offset in range(-82, 83, 18):
            c.create_line(cx+offset, cy-137, cx+offset, cy-116,
                          fill="#f8fafc", width=5)
            c.create_line(cx+offset, cy+116, cx+offset, cy+137,
                          fill="#f8fafc", width=5)
            c.create_line(cx-137, cy+offset, cx-116, cy+offset,
                          fill="#f8fafc", width=5)
            c.create_line(cx+116, cy+offset, cx+137, cy+offset,
                          fill="#f8fafc", width=5)

        c.create_rectangle(cx-95, cy-95, cx+95, cy+95,
                           fill="#414b5a", outline="#596779", width=2)
        c.create_text(cx, cy-18, text="INTERSECCION",
                      fill="#f8fafc", font=("Arial", 12, "bold"))
        c.create_text(cx, cy+12, text=f"FASE {self.phase}",
                      fill="#a9f2c5" if self.phase == "NS" else "#ffd28b",
                      font=("Arial", 16, "bold"))
        c.create_text(cx, cy+43, text=f"{self.remaining} s restantes",
                      fill="#ffffff", font=("Arial", 10))

        panels = {
            0: (cx-110, 12, cx+110, 105),
            1: (cx-110, h-105, cx+110, h-12),
            2: (w-220, cy-52, w-12, cy+52),
            3: (12, cy-52, 220, cy+52),
        }
        lane_colors = ("#3b82f6", "#f59e0b", "#8b5cf6", "#10b981")
        light_centers = {
            0: (cx+68, cy-150),
            1: (cx-68, cy+150),
            2: (cx+150, cy+68),
            3: (cx-150, cy-68),
        }

        for i, agent in enumerate(self.agents):
            car_color = lane_colors[i]
            for k in range(min(agent.queue, MAX_QUEUE)):
                if i == 0:
                    x, y = cx-67, cy-143-k*12
                    body = (x-7, y-8, x+7, y+8)
                    roof = (x-5, y-3, x+5, y+4)
                    wheels = [(x-9, y-6, x-7, y-2),
                              (x+7, y-6, x+9, y-2),
                              (x-9, y+2, x-7, y+6),
                              (x+7, y+2, x+9, y+6)]
                elif i == 1:
                    x, y = cx+67, cy+143+k*12
                    body = (x-7, y-8, x+7, y+8)
                    roof = (x-5, y-3, x+5, y+4)
                    wheels = [(x-9, y-6, x-7, y-2),
                              (x+7, y-6, x+9, y-2),
                              (x-9, y+2, x-7, y+6),
                              (x+7, y+2, x+9, y+6)]
                elif i == 2:
                    x, y = cx+143+k*12, cy-67
                    body = (x-8, y-7, x+8, y+7)
                    roof = (x-3, y-5, x+4, y+5)
                    wheels = [(x-6, y-9, x-2, y-7),
                              (x+2, y-9, x+6, y-7),
                              (x-6, y+7, x-2, y+9),
                              (x+2, y+7, x+6, y+9)]
                else:
                    x, y = cx-143-k*12, cy+67
                    body = (x-8, y-7, x+8, y+7)
                    roof = (x-3, y-5, x+4, y+5)
                    wheels = [(x-6, y-9, x-2, y-7),
                              (x+2, y-9, x+6, y-7),
                              (x-6, y+7, x-2, y+9),
                              (x+2, y+7, x+6, y+9)]

                c.create_oval(*body, fill=car_color,
                              outline="#f8fafc", width=1)
                c.create_rectangle(*roof, fill="#dbeafe",
                                   outline="#243247", width=1)
                for wheel in wheels:
                    c.create_rectangle(*wheel, fill="#111827", outline="")

        for i, agent in enumerate(self.agents):
            x1, y1, x2, y2 = panels[i]
            active = self.phase == ("NS" if i in (0, 1) else "EO")
            accent = lane_colors[i]
            panel_bg = "#e8f7ee" if active else "#ffffff"
            c.create_rectangle(x1, y1, x2, y2, fill=panel_bg,
                               outline=accent, width=3)
            c.create_text(x1+12, y1+17, text=DIRECTIONS[i].upper(),
                          anchor="w", fill="#172033",
                          font=("Arial", 11, "bold"))
            c.create_text(x2-12, y1+17,
                          text="N-S" if i in (0, 1) else "E-O",
                          anchor="e", fill=accent,
                          font=("Arial", 9, "bold"))
            c.create_text(x1+12, y1+43,
                          text=f"Cola: {agent.queue}/{MAX_QUEUE}",
                          anchor="w", fill="#26364d",
                          font=("Arial", 10, "bold"))
            c.create_text(x2-12, y1+43,
                          text="HABILITADA" if active else "DETENIDA",
                          anchor="e",
                          fill="#16834a" if active else "#c03939",
                          font=("Arial", 9, "bold"))

            bar_left, bar_right = x1+12, x2-12
            bar_y = y2-19
            usable = bar_right-bar_left
            gap = 2
            seg_w = (usable-gap*(MAX_QUEUE-1))/MAX_QUEUE
            for k in range(MAX_QUEUE):
                sx = bar_left+k*(seg_w+gap)
                fill = accent if k < agent.queue else "#dce3ec"
                c.create_rectangle(sx, bar_y, sx+seg_w, bar_y+9,
                                   fill=fill, outline="")

            lx, ly = light_centers[i]
            c.create_rectangle(lx-12, ly-25, lx+12, ly+25,
                               fill="#17212f", outline="#0f172a", width=2)
            c.create_oval(lx-7, ly-19, lx+7, ly-5,
                          fill="#ef4444" if not active else "#572b32",
                          outline="#111827", width=1)
            c.create_oval(lx-7, ly+5, lx+7, ly+19,
                          fill="#22c55e" if active else "#244b36",
                          outline="#111827", width=1)

        for i, agent in enumerate(self.agents):
            active = self.phase == ("NS" if i in (0, 1) else "EO")
            lx, ly = light_centers[i]
            c.create_rectangle(lx-12, ly-25, lx+12, ly+25,
                               fill="#17212f", outline="#0f172a", width=2)
            c.create_oval(lx-7, ly-19, lx+7, ly-5,
                          fill="#ef4444" if not active else "#572b32",
                          outline="#111827", width=1)
            c.create_oval(lx-7, ly+5, lx+7, ly+19,
                          fill="#22c55e" if active else "#244b36",
                          outline="#111827", width=1)

        c.create_text(cx, cy-111, text="↓", fill="#ffffff",
                      font=("Arial", 22, "bold"))
        c.create_text(cx, cy+111, text="↑", fill="#ffffff",
                      font=("Arial", 22, "bold"))
        c.create_text(cx-111, cy, text="→", fill="#ffffff",
                      font=("Arial", 22, "bold"))
        c.create_text(cx+111, cy, text="←", fill="#ffffff",
                      font=("Arial", 22, "bold"))

    def refresh(self):
        self.draw_intersection()
        total_queue = sum(a.queue for a in self.agents)
        completed_waits = [
            wait
            for agent in self.agents
            for wait in agent.completed_wait_times
        ]
        average_wait = (
            sum(completed_waits) / len(completed_waits)
            if completed_waits else 0.0
        )
        self.status.config(
            text=(
                f"Modo: {self.mode.get()}    |    "
                f"Tiempo simulado: {self.elapsed} s    |    "
                f"Vehiculos en cola: {total_queue}/48    |    "
                f"Vehiculos atendidos: {self.history_served}    |    "
                f"Espera media: {average_wait:.1f} s"
            )
        )
        self.decision.config(
            text=f"Coordinador: {self.coordinator.last_decision}"
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TrafficSimulation()
    app.run()