# Was pasted into main project file

print("\n--- SENSITIVITY ANALYSIS (LP Relaxation) ---\n")

# Dual values for constraints
for c in lp.getConstrs():
    print(f"{c.ConstrName:45s}  Dual (Pi) = {c.Pi:8.3f},  Slack = {c.Slack:5.1f}")

# Reduced costs for variables
print("\nReduced Costs for non-selected X[i,j]:")
for i in rounds:
    for j in positions:
        if X[i,j].X < 0.5:        # not chosen in integer solution
             v = lp.getVarByName(f"X[{i},{j}]")
             print(f"  X[{i},{j}] ({pos_names[j]}): RC = {v.RC:7.3f}")
