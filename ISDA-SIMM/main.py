import pandas as pd
from src.agg_margins import SIMM

if __name__ == "__main__":

    path = "ISDA-SIMM/CRIF/"
    crif = pd.read_csv(path+"crif.csv", header=0)
    portfolio1 = SIMM(crif=crif, calculation_currency="USD", exchange_rate=1)

    # Total SIMM
    print(portfolio1.simm)

    # SIMM breakdown
    print(portfolio1.simm_break_down)