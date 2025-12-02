import gurobipy as gp
from gurobipy import GRB
import pandas as pd

csv_file_path = 'Data\Fantasy Baseball Stats.csv'
df = pd.read_csv(csv_file_path)
print(df.head())

rounds = range(1, 17)  
positions = range(1, 9) 

pos_names = {
    1: "1B",
    2: "2B",
    3: "3B",
    4: "SS",
    5: "C",
    6: "OF",
    7: "SP",
    8: "RP"
}


P = {(i, j): 0.0 for i in rounds for j in positions}


m = gp.Model("Fantasy_Baseball_Draft_Optimization")

X = m.addVars(rounds, positions, vtype=GRB.BINARY, name="X")

m.setObjective(
    gp.quicksum(P[i, j] * X[i, j] for i in rounds for j in positions),
    GRB.MAXIMIZE
)

# -----------------------------
# Constraints
# -----------------------------

for i in rounds:
    m.addConstr(gp.quicksum(X[i, j] for j in positions) == 1,
                name=f"OnePickPerRound_{i}")

for j in range(1, 6):
    m.addConstr(
        gp.quicksum(X[i, j] for i in rounds) >= 1,
        name=f"MinPos_{j}"
    )
    m.addConstr(
        gp.quicksum(X[i, j] for i in rounds) <= 3,
        name=f"MaxPos_{j}"
    )

j_of = 6
m.addConstr(
    gp.quicksum(X[i, j_of] for i in rounds) >= 3,
    name="OF_Min"
)
m.addConstr(
    gp.quicksum(X[i, j_of] for i in rounds) <= 5,
    name="OF_Max"
)

m.addConstr(
    gp.quicksum(X[i, j] for i in rounds for j in range(1, 7)) == 10,
    name="TotalHitters"
)

j_sp = 7
m.addConstr(
    gp.quicksum(X[i, j_sp] for i in rounds) >= 2,
    name="SP_Min"
)
m.addConstr(
    gp.quicksum(X[i, j_sp] for i in rounds) <= 6,
    name="SP_Max"
)

j_rp = 8
m.addConstr(
    gp.quicksum(X[i, j_rp] for i in rounds) >= 2,
    name="RP_Min"
)
m.addConstr(
    gp.quicksum(X[i, j_rp] for i in rounds) <= 6,
    name="RP_Max"
)

m.addConstr(
    gp.quicksum(X[i, j_sp] for i in rounds)
    + gp.quicksum(X[i, j_rp] for i in rounds)
    == 8,
    name="TotalPitchers"
)
#------------------
#End Constraints
#------------------

m.optimize()

if m.status == GRB.OPTIMAL:
    print(f"\nOptimal objective value (total expected points): {m.objVal:.2f}\n")
    print("Draft plan (round -> position):")
    for i in rounds:
        for j in positions:
            if X[i, j].X > 0.5:  
                print(f"  Round {i}: {pos_names[j]} (X[{i},{j}] = 1)")
else:
    print("No optimal solution found. Model status:", m.status)
