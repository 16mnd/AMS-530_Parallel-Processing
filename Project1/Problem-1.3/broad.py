import networkx as nx
from collections import deque


def create_graph():

    graph = nx.Graph()

    # Outer 32-node cycle
    for vertex in range(1, 32):
        graph.add_edge(vertex, vertex + 1)

    graph.add_edge(32, 1)

    # Cross-circle edges
    cross_edges = [
        (1, 12),
        (2, 23),
        (3, 19),
        (4, 26),
        (5, 30),
        (6, 13),
        (7, 18),
        (8, 24),
        (9, 31),
        (10, 20),
        (11, 27),
        (14, 21),
        (15, 25),
        (16, 32),
        (17, 28),
        (22, 29)
    ]

    graph.add_edges_from(cross_edges)

    return graph


def validate_graph(graph, log):

    assert graph.number_of_nodes() == 32
    assert graph.number_of_edges() == 48
    assert all(graph.degree(vertex) == 3 for vertex in graph.nodes())
    assert nx.is_connected(graph)
    assert len(list(nx.selfloop_edges(graph))) == 0

    print("GRAPH VALIDATION", file=log)
    print("----------------", file=log)
    print("Vertices:", graph.number_of_nodes(), file=log)
    print("Edges:", graph.number_of_edges(), file=log)
    print("All vertices degree 3: PASS", file=log)
    print("Connected: PASS", file=log)
    print("No self-loops: PASS", file=log)


def generate_transmissions(graph, informed):

    informed = set(informed)
    uninformed = set(graph.nodes()) - informed

    sender_options = []

    # Determine possible receivers for every informed sender
    for sender in sorted(informed):

        receivers = [
            neighbor
            for neighbor in graph.neighbors(sender)
            if neighbor in uninformed
        ]

        if receivers:
            sender_options.append((sender, receivers))

    valid_transmissions = []
    max_transmissions = 0

    # Recursively construct legal simultaneous transmissions
    def search(index, transmissions, used_receivers):

        nonlocal max_transmissions
        nonlocal valid_transmissions

        if index == len(sender_options):

            number_transmissions = len(transmissions)

            if number_transmissions > max_transmissions:
                max_transmissions = number_transmissions
                valid_transmissions = [tuple(transmissions)]

            elif number_transmissions == max_transmissions:
                valid_transmissions.append(tuple(transmissions))

            return

        sender, receivers = sender_options[index]

        # Sender remains idle
        search(index + 1, transmissions, used_receivers)

        # Sender transmits to one uninformed neighbor
        for receiver in receivers:

            if receiver in used_receivers:
                continue

            transmissions.append((sender, receiver))
            used_receivers.add(receiver)

            search(index + 1, transmissions, used_receivers)

            used_receivers.remove(receiver)
            transmissions.pop()

    search(0, [], set())

    valid_transmissions = [
        transmissions
        for transmissions in valid_transmissions
        if len(transmissions) > 0
    ]

    return valid_transmissions


def verify_five_steps(graph, source, log):

    states = {frozenset([source])}

    print("\nFIVE-STEP VERIFICATION", file=log)
    print("----------------------", file=log)
    print("A 5-step broadcast requires perfect doubling:", file=log)
    print("1 -> 2 -> 4 -> 8 -> 16 -> 32", file=log)

    for step in range(1, 6):

        next_states = set()

        print("\nStep:", step, file=log)
        print("Informed vertices required before step:", 2 ** (step - 1), file=log)
        print("Required transmissions:", 2 ** (step - 1), file=log)
        print("States entering step:", len(states), file=log)

        for informed in states:

            required_transmissions = len(informed)

            transmission_options = generate_transmissions(graph, informed)

            for transmissions in transmission_options:

                # Perfect doubling is required
                if len(transmissions) != required_transmissions:
                    continue

                newly_informed = {
                    receiver
                    for sender, receiver in transmissions
                }

                next_state = frozenset(set(informed) | newly_informed)

                next_states.add(next_state)

        print("Valid perfect-doubling states after step:", len(next_states), file=log)

        if not next_states:

            print("\nPerfect doubling becomes impossible at Step {}.".format(step), file=log)
            print("Therefore a 5-step broadcast is IMPOSSIBLE.", file=log)

            return False

        states = next_states

    # If a 32-node state survived, five steps is possible
    for state in states:

        if len(state) == 32:

            print("\nA 5-step broadcast EXISTS.", file=log)

            return True

    print("\nNo 5-step broadcast exists.", file=log)

    return False


def solve_broadcast(graph, source, log):

    all_vertices = frozenset(graph.nodes())
    initial_state = frozenset([source])

    queue = deque([(initial_state, [])])
    visited = {initial_state}

    states_processed = 0
    current_depth = -1

    print("\nBROADCAST SEARCH", file=log)
    print("----------------", file=log)
    print("Source:", source, file=log)

    while queue:

        informed, schedule = queue.popleft()

        states_processed += 1
        depth = len(schedule)

        if depth != current_depth:

            current_depth = depth

            print("\nStep depth:", depth, file=log)
            print("States processed:", states_processed, file=log)
            print("States currently in queue:", len(queue), file=log)
            print("Unique states discovered:", len(visited), file=log)

        if informed == all_vertices:

            print("\nBROADCAST COMPLETE", file=log)
            print("Broadcast steps:", len(schedule), file=log)
            print("States processed:", states_processed, file=log)
            print("Unique states discovered:", len(visited), file=log)

            return schedule

        transmission_options = generate_transmissions(graph, informed)

        for transmissions in transmission_options:

            newly_informed = {
                receiver
                for sender, receiver in transmissions
            }

            next_state = frozenset(set(informed) | newly_informed)

            if next_state in visited:
                continue

            visited.add(next_state)

            next_schedule = schedule + [transmissions]

            queue.append((next_state, next_schedule))

    return None


def print_schedule(schedule, log):

    if schedule is None:

        print("\nNo broadcast schedule found.", file=log)

        return

    print("\nBROADCAST SCHEDULE", file=log)
    print("------------------", file=log)

    total_edge_uses = 0

    for step, transmissions in enumerate(schedule, start=1):

        links = [
            "{}-{}".format(sender, receiver)
            for sender, receiver in transmissions
        ]

        active_edges = len(transmissions)
        total_edge_uses += active_edges

        print(
            "Step {}: {} | Active edges: {}".format(
                step,
                ", ".join(links),
                active_edges
            ),
            file=log
        )

    print("\nRESULT", file=log)
    print("------", file=log)
    print("Total steps:", len(schedule), file=log)
    print("Total edge uses:", total_edge_uses, file=log)


def main():

    graph = create_graph()
    source = 1

    with open("broadcast.log", "w") as log:

        # Validate graph
        validate_graph(graph, log)

        # Determine whether theoretical 5-step minimum is possible
        five_step_exists = verify_five_steps(graph, source, log)

        # Find broadcast schedule
        schedule = solve_broadcast(graph, source, log)

        # Print schedule
        print_schedule(schedule, log)

        # Final conclusion
        print("\nFINAL CONCLUSION", file=log)
        print("----------------", file=log)

        print("The theoretical lower bound is 5 steps.", file=log)

        if five_step_exists:

            print("A 5-step broadcast exists.", file=log)

        elif schedule is not None and len(schedule) == 6:

            print("The exhaustive perfect-doubling test proves that 5 steps is impossible.", file=log)
            print("A valid 6-step broadcast was found.", file=log)
            print("Therefore the minimum broadcast time is 6 steps.", file=log)

        else:

            print("No 5-step solution exists.", file=log)
            print("The minimum broadcast time was not established.", file=log)


if __name__ == "__main__":
    main()
