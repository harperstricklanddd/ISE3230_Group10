import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

csv_file_path = 'Data\\PositionRoundAverages.csv'
df = pd.read_csv(csv_file_path)

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

meta_cols = ['Position', 'Round']

all_value_cols = [c for c in df.columns if c not in meta_cols]

value_cols = all_value_cols[:9]

values = df[value_cols].apply(pd.to_numeric, errors='coerce')

value_cols = df.columns[1:]

values = values.iloc[:18, :9]
if not np.isfinite(values.to_numpy()).all():
    raise ValueError("Numeric data still has NaN/Inf; please clean the CSV.")

n_rounds = values.shape[0]     
n_positions = values.shape[1]  

rounds = range(1, n_rounds + 1)         
positions = range(1, n_positions + 1)

P = {}
for i in rounds:            
    for j in positions:     
        P[(i, j)] = float(values.iloc[i - 1, j - 1])


m = gp.Model("Fantasy_Baseball_Draft_Optimization")

X = m.addVars(rounds, positions, vtype=GRB.BINARY, name="X")

m.setObjective(
    gp.quicksum(P[(i, j)] * X[i, j] for i in rounds for j in positions),
    GRB.MAXIMIZE
)


# -----------------------------
# Constraints
# -----------------------------

# Draft constraint
for i in rounds:
    m.addConstr(gp.quicksum(X[i, j] for j in positions) == 1)

# Infield and catcher
for j in range(1, 5):
    m.addConstr(gp.quicksum(X[i, j] for i in rounds) >= 1)
    m.addConstr(gp.quicksum(X[i, j] for i in rounds) <= 3)

# Outfield
m.addConstr(gp.quicksum(X[i, 6] for i in rounds) >= 3)
m.addConstr(gp.quicksum(X[i, 6] for i in rounds) <= 5)

# Designated hitter
m.addConstr(gp.quicksum(X[i, 7] for i in rounds) >= 0)
m.addConstr(gp.quicksum(X[i, 7] for i in rounds) <= 2)

# Total position players
m.addConstr(
    gp.quicksum(X[i, 1] for i in rounds) +
    gp.quicksum(X[i, 2] for i in rounds) +
    gp.quicksum(X[i, 3] for i in rounds) +
    gp.quicksum(X[i, 4] for i in rounds) +
    gp.quicksum(X[i, 5] for i in rounds) +
    gp.quicksum(X[i, 6] for i in rounds) +
    gp.quicksum(X[i, 7] for i in rounds) 
    == 10
)

# Starting pitchers
m.addConstr(gp.quicksum(X[i, 8] for i in rounds) >= 2)
m.addConstr(gp.quicksum(X[i, 8] for i in rounds) <= 6)

# Bullpen pitchers
m.addConstr(gp.quicksum(X[i, 9] for i in rounds) >= 2)
m.addConstr(gp.quicksum(X[i, 9] for i in rounds) <= 6)

# Total pitchers
m.addConstr(
    gp.quicksum(X[i, 8] for i in rounds) +
    gp.quicksum(X[i, 9] for i in rounds) 
    == 8
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
