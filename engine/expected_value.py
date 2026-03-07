def calc_expected_value(prob_dict, odds_dict):

    ev = {}

    for combo, p in prob_dict.items():
        if combo not in odds_dict:
            continue
        ev[combo] = p * odds_dict[combo]

    return ev
