def allocate_kelly(ev_dict, prob_dict, odds_dict, bankroll=10000):

    bet = {}

    for combo in ev_dict:

        p = prob_dict.get(combo)
        o = odds_dict.get(combo)

        if not p or not o:
            continue

        b = o - 1
        q = 1 - p

        f = (b * p - q) / b

        if f <= 0:
            continue

        bet[combo] = min(bankroll * f, bankroll * 0.25)

    return bet
