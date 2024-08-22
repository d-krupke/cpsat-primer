from ortools.sat.python import cp_model


def test_truck_example():
    model = cp_model.CpModel()
    # A value representing the load that needs to be transported
    load_value = model.new_int_var(0, 100, "load_value")

    # ... some logic to determine the load value ...

    # A variable to decide which truck to rent
    truck_a = model.new_bool_var("truck_a")
    truck_b = model.new_bool_var("truck_b")
    truck_c = model.new_bool_var("truck_c")

    # only rent one truck
    model.add_at_most_one([truck_a, truck_b, truck_c])

    # Depending on which truck is rented, the load value is limited
    model.add(load_value <= 50).only_enforce_if(truck_a)
    model.add(load_value <= 80).only_enforce_if(truck_b)
    model.add(load_value <= 100).only_enforce_if(truck_c)

    # Some additional logic
    driver_has_big_truck_license = model.new_bool_var("driver_has_big_truck_license")
    driver_has_special_license = model.new_bool_var("driver_has_special_license")
    # Only drivers with a big truck license or a special license can rent truck c
    model.add_bool_or(
        driver_has_big_truck_license, driver_has_special_license
    ).only_enforce_if(truck_c)

    # Minimize the rent cost
    model.minimize(30 * truck_a + 40 * truck_b + 80 * truck_c)


def test_only_enforce_if_list_with_negation():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    z = model.new_int_var(0, 10, "z")

    b2 = model.new_bool_var("b2")
    b3 = model.new_bool_var("b3")

    model.add(x + z == 10).only_enforce_if([b2, ~b3])  # only enforce if b2 AND NOT b3
