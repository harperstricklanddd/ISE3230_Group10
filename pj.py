import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

csv_file_path = 'Data\\PositionRoundAverages.csv'
df = pd.read_csv(csv_file_path)
print(df)
# print(df.head)

pos_names = {
    1: "1B",
    2: "2B",
    3: "3B",
    4: "SS",
    5: "C",
    6: "OF",
    7: "DH",
    8: "SP",
    9: "RP"
}

value_cols = df.columns[1:]

values = df[value_cols].apply(pd.to_numeric, errors='coerce')

bad_mask = ~np.isfinite(values.to_numpy())
if bad_mask.any():
    print("Found NaN or Inf in your CSV at these (row, col) positions:")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            if bad_mask[r, c]:
                print(f"  row {r+1} (in df), column '{value_cols[c]}' -> {values.iloc[r,c]}")

n_rounds = values.shape[0]        
n_positions = values.shape[1]    

rounds = range(1, n_rounds - 1) #+1
positions = range(1, n_positions - 1) #+2
print(rounds)
print(positions)

P = {}
for i in rounds:
    for j in positions:
        P[(i, j)] = float(values.iloc[j+1, i+1])
        print(i)
        print(j)


m = gp.Model("Fantasy_Baseball_Draft_Optimization")

X = m.addVars(rounds, positions, vtype=GRB.BINARY, name="X")

m.setObjective(
    gp.quicksum(P[(i, j)] * X[i, j] for i in rounds for j in positions),
    GRB.MAXIMIZE
)

# -----------------------------
# Constraints
# -----------------------------

for i in rounds:
    m.addConstr(gp.quicksum(X[i, j] for j in positions) == 1,
                name=f"OnePickPerRound_{i}")

for j in range(1, 6):
    m.addConstr(gp.quicksum(X[i, j] for i in rounds) >= 1,
                name=f"MinPos_{j}")
    m.addConstr(gp.quicksum(X[i, j] for i in rounds) <= 3,
                name=f"MaxPos_{j}")

m.addConstr(gp.quicksum(X[i, 6] for i in rounds) >= 3,
            name="OF_Min")
m.addConstr(gp.quicksum(X[i, 6] for i in rounds) <= 5, 
            name="OF_Max")

m.addConstr(
    gp.quicksum(X[i, j] for i in rounds for j in range(1, 7)) == 10,
    name="TotalHitters"
)

m.addConstr(gp.quicksum(X[i, 7] for i in rounds) >= 2, 
            name="SP_Min")
m.addConstr(gp.quicksum(X[i, 7] for i in rounds) <= 6, 
            name="SP_Max")

m.addConstr(gp.quicksum(X[i, 8] for i in rounds) >= 2, 
            name="RP_Min")
m.addConstr(gp.quicksum(X[i, 8] for i in rounds) <= 6, 
            name="RP_Max")

m.addConstr(
    gp.quicksum(X[i, 7] for i in rounds) +
    gp.quicksum(X[i, 8] for i in rounds) == 8,
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
