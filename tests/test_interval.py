from ortools.sat.python import cp_model


def test_interval():
    model = cp_model.CpModel()

    start_var = model.new_int_var(0, 100, "start")
    length_var = model.new_int_var(10, 20, "length")
    end_var = model.new_int_var(0, 100, "end")
    is_present_var = model.new_bool_var("is_present")

    # creating an interval whose length can be influenced by a variable (more expensive)
    flexible_interval = model.new_interval_var(
        start=start_var, size=length_var, end=end_var, name="flexible_interval"
    )

    # creating an interval of fixed length
    fixed_interval = model.new_fixed_size_interval_var(
        start=start_var,
        size=10,  # needs to be a constant
        name="fixed_interval",
    )

    # creating an interval that can be present or not and whose length can be influenced by a variable (most expensive)
    optional_interval = model.new_optional_interval_var(
        start=start_var,
        size=length_var,
        end=end_var,
        is_present=is_present_var,
        name="optional_interval",
    )

    # creating an interval that can be present or not
    optional_fixed_interval = model.new_optional_fixed_size_interval_var(
        start=start_var,
        size=10,  # needs to be a constant
        is_present=is_present_var,
        name="optional_fixed_interval",
    )

    model.add_no_overlap(
        interval_vars=[
            flexible_interval,
            fixed_interval,
            optional_interval,
            optional_fixed_interval,
        ]
    )
    model.add_no_overlap_2d(
        x_intervals=[
            flexible_interval,
            fixed_interval,
            optional_interval,
            optional_fixed_interval,
        ],
        y_intervals=[
            flexible_interval,
            fixed_interval,
            optional_interval,
            optional_fixed_interval,
        ],
    )

    demand_vars = [model.new_int_var(1, 10, f"demand_{i}") for i in range(4)]
    capacity_var = model.new_int_var(1, 100, "capacity")
    model.add_cumulative(
        intervals=[
            flexible_interval,
            fixed_interval,
            optional_interval,
            optional_fixed_interval,
        ],
        demands=demand_vars,
        capacity=capacity_var,
    )


def test_no_overlap():
    from collections import namedtuple

    # Convert time to index and back
    def t_to_idx(hour, minute):
        return (hour - 8) * 12 + minute // 5

    def idx_to_t(timepoint):
        hour = 8 + timepoint // 12
        minute = (timepoint % 12) * 5
        return f"{hour}:{minute:02d}"

    # Define meeting information using namedtuples
    MeetingInfo = namedtuple("MeetingInfo", ["start_times", "duration"])

    # Meeting definitions
    meetings = {
        "meeting_a": MeetingInfo(
            start_times=[
                [t_to_idx(hour=8, minute=0), t_to_idx(hour=12, minute=0)],
                [t_to_idx(hour=16, minute=0), t_to_idx(hour=17, minute=0)],
            ],
            duration=24,  # 2 hours
        ),
        "meeting_b": MeetingInfo(
            start_times=[
                [t_to_idx(hour=10, minute=0), t_to_idx(hour=12, minute=0)],
            ],
            duration=6,  # 30 minutes
        ),
        "meeting_c": MeetingInfo(
            start_times=[
                [t_to_idx(hour=16, minute=0), t_to_idx(hour=17, minute=0)],
            ],
            duration=3,  # 15 minutes
        ),
        "meeting_d": MeetingInfo(
            start_times=[
                [t_to_idx(hour=8, minute=0), t_to_idx(hour=10, minute=0)],
                [t_to_idx(hour=12, minute=0), t_to_idx(hour=14, minute=0)],
            ],
            duration=12,  # 1 hour
        ),
    }

    # Create a new CP-SAT model
    model = cp_model.CpModel()

    # Create start time variables for each meeting
    start_time_vars = {
        meeting_name: model.new_int_var_from_domain(
            cp_model.Domain.from_intervals(meeting_info.start_times),
            f"start_{meeting_name}",
        )
        for meeting_name, meeting_info in meetings.items()
    }

    # Create interval variables for each meeting
    interval_vars = {
        meeting_name: model.new_fixed_size_interval_var(
            start=start_time_vars[meeting_name],
            size=meeting_info.duration,
            name=f"interval_{meeting_name}",
        )
        for meeting_name, meeting_info in meetings.items()
    }

    # Add the no-overlap constraint to the model
    model.add_no_overlap(list(interval_vars.values()))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    # Extract and print the solution
    scheduled_times = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for meeting_name in meetings:
            start_time = solver.value(start_time_vars[meeting_name])
            scheduled_times[meeting_name] = start_time
            print(f"{meeting_name} starts at {idx_to_t(start_time)}")
    else:
        print("No feasible solution found.")
