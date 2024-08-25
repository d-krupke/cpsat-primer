from ortools.sat.python import cp_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Function to convert time to index
    def time_to_index(hour, minute):
        return (hour - 8) * 12 + minute // 5

    # Function to convert index back to time
    def index_to_time(index):
        hour = 8 + index // 12
        minute = (index % 12) * 5
        return hour, minute

    # Possible start times for meetings
    possible_meeting_times = {
        "meeting_a": [
            [time_to_index(hour=8, minute=0), time_to_index(hour=12, minute=0)],
            [time_to_index(hour=16, minute=0), time_to_index(hour=17, minute=0)],
        ],
        "meeting_b": [
            [time_to_index(hour=10, minute=0), time_to_index(hour=12, minute=0)],
        ],
        "meeting_c": [
            [time_to_index(hour=16, minute=0), time_to_index(hour=17, minute=0)],
        ],
        "meeting_d": [
            [time_to_index(hour=8, minute=0), time_to_index(hour=10, minute=0)],
            [time_to_index(hour=12, minute=0), time_to_index(hour=14, minute=0)],
        ],
        # Additional meetings can be added here...
    }

    # Durations of the meetings in 5-minute intervals
    meeting_durations = {
        "meeting_a": 24,  # 2 hours
        "meeting_b": 6,  # 30 minutes
        "meeting_c": 3,  # 15 minutes
        "meeting_d": 12,  # 1 hour
        # Additional meetings can be added here...
    }

    # Create a new CP-SAT model
    model = cp_model.CpModel()

    # Create start time variables for each meeting
    start_time_vars = {
        meeting: model.new_int_var_from_domain(
            cp_model.Domain.from_intervals(times), f"start_{meeting}"
        )
        for meeting, times in possible_meeting_times.items()
    }

    # Create interval variables for each meeting
    interval_vars = {
        meeting: model.new_fixed_size_interval_var(
            start=start_time_vars[meeting], size=duration, name=f"interval_{meeting}"
        )
        for meeting, duration in meeting_durations.items()
    }

    # Add the no-overlap constraint to the model
    model.add_no_overlap(list(interval_vars.values()))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    # Function to convert time index to human-readable format
    def convert_to_timepoint(timepoint):
        hour = 8 + timepoint // 12
        minute = (timepoint % 12) * 5
        return f"{hour}:{minute:02d}"

    # Extract and print the solution
    scheduled_times = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for meeting, interval in interval_vars.items():
            start_time = solver.value(start_time_vars[meeting])
            scheduled_times[meeting] = start_time
            print(f"{meeting} starts at {convert_to_timepoint(start_time)}")
    else:
        print("No feasible solution found.")

    # Plotting the possible and scheduled times for each meeting
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot possible times as shaded areas
    for meeting, times in possible_meeting_times.items():
        for time_range in times:
            start, end = time_range
            ax.plot(
                [start, end],
                [meeting, meeting],
                color="red",
                linestyle="-",
                marker="o",
                label=(
                    "Possible Times"
                    if meeting == "meeting_a" and time_range == times[0]
                    else ""
                ),
            )

    # Plot scheduled times as blue bars
    for meeting, start_time in scheduled_times.items():
        duration = meeting_durations[meeting]
        ax.barh(
            meeting,
            duration,
            left=start_time,
            color="blue",
            edgecolor="black",
            label="Scheduled Time" if meeting == "meeting_a" else "",
        )

    # Customizing the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Meetings")
    ax.set_title("Meeting Schedule")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    # Convert x-axis to show actual time labels
    def time_ticks(x, pos):
        hour, minute = index_to_time(int(x))
        return f"{hour}:{minute:02d}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(time_ticks))
    ax.xaxis.set_major_locator(plt.MultipleLocator(12))  # Every hour
    ax.xaxis.set_minor_locator(plt.MultipleLocator(3))  # Every 15 minutes

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("meeting_schedule.png")
# plt.show()
