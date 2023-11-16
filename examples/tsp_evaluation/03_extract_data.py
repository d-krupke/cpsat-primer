from algbench import read_as_pandas
from _conf import EXPERIMENT_DATA, SIMPLIFIED_RESULTS, TIME_LIMIT, STRATEGIES


def is_valid(entry):
    return (
        entry["parameters"]["args"]["time_limit"] == TIME_LIMIT
        and entry["parameters"]["args"]["strategy"] in STRATEGIES
    )


if __name__ == "__main__":
    t = read_as_pandas(
        EXPERIMENT_DATA,
        lambda entry: {
            "instance_name": entry["parameters"]["args"]["instance_name"],
            "num_nodes": entry["result"]["num_nodes"],
            "time_limit": entry["parameters"]["args"]["time_limit"],
            "strategy": entry["parameters"]["args"]["strategy"],
            "opt_tol": entry["parameters"]["args"]["opt_tol"],
            "runtime": entry["runtime"],
            "objective": entry["result"]["objective"],
            "lower_bound": entry["result"]["lower_bound"],
        }
        if is_valid(entry)
        else None,
    )
    t.drop_duplicates(subset=["instance_name", "strategy", "opt_tol"], inplace=True)
    t.to_json(SIMPLIFIED_RESULTS)
