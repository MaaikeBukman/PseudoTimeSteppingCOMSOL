import mph
import numpy as np
import pandas as pd
import time
from RunNetwork import run_network


client = mph.start()
model = client.load("file backstep_gen + path") # Paste here the file + path
network_file = "" # Network_full + path
outfile = 'NetworkOutcomes_python_final.xlsx'
sheet = "Sheet1"

limit = 6.45E-9     #convergence criterion
max_iter = 100
 
uin_list  = [0.001, 0.004, 0.007, 0.01, 0.012, 0.015]         # inflow velocities
mesh_list = np.linspace(0.0106, 0.0256, int((0.0256 - 0.0106) / 0.005) + 1)

num_cases = len(uin_list)*len(mesh_list)
print("Number of cases: ", num_cases)

OUTCOMES = []
OUTCOMES2 = []

count = 0
start_time = time.time()

for i, uin in enumerate(uin_list):
    for j,mesh_size in enumerate(mesh_list):
        print(f'Running case {count}: uin = {uin:.4f}, mesh = {mesh_size:.4f}')

        # parameters
        model.parameter("uin", f"{uin}[m/s]")
        model.java.mesh("mesh1").feature("size").set("hmax", f'{mesh_size}')

        # Run solvers 
        model.java.sol('sol2').runAll()
        model.java.sol('sol6').runAll()
        model.java.sol('sol5').runAll()
        model.java.sol('sol3').runAll()
        model.java.sol('sol5').runAll()
       
        # CFL_e table
        try:
            model.java.sol('sol4').runAll()
            tbl2 = model.java.result().table('tbl2')
            data = np.array(tbl2.getReal())  
            CONV = data[:, -2] 
            patience = CONV[-1]
            l1 = len(CONV)
        except Exception as e:
            print("Error reading table:", e)
            CONV = np.zeros(200)
            l1 = 150
            patience = limit

        print(f'CFL_e length = {len(CONV)}')
        print('CFL_e first 5:', CONV[:5], 'last 5:', CONV[-5:])

        # CFL_iter table

        try:
            model.java.sol('sol7').runAll()
            tbl7 = model.java.result().table('tbl2')
            data2 = np.array(tbl2.getReal())
            CONV2 = data2[:, 1]  
            patience2 = CONV2[-1]
            l2 = len(CONV2)
        except Exception as e:
            print("Error reading table:", e)
            CONV2 = np.zeros(200)
            l2 = 150
            patience2 = limit

        print(f'CFL_iter length = {len(CONV2)}')
        print('CFL_iter first 5:', CONV2[:5], 'last 5:', CONV2[-5:])

        # convergence criterion
        if l1 < 150 and l2 < 150:
            patience = max(patience, patience2, limit)
        elif l1 == 150 and l2 == 150:
            if patience < 1e-6 or patience2 < 1e-6:
                patience = max(min(patience, patience2), limit)
            else:
                patience = 1e-7
        else:
            patience = max(min(patience, patience2), limit)

        # NN
        CONV3 = run_network(model, network_file, iter, max_iter, patience)
        print(f"NN output (CONV1): length = {len(CONV3)}")

        # Store results
        if CONV3[-1] > 10:
            NN_iter = 200
        else:
            NN_iter = len(CONV3)

        OUTCOMES.append([
            uin,               # inflow velocity
            mesh_size,              # mesh size
            NN_iter,           # NumIter NN
            len(CONV),         # NumIter CFL_e
            len(CONV2)         # NumIter CFL_iter
        ])

        OUTCOMES2.append([
            uin,
            mesh_size,
            CONV3[-1],         # final NN error
            np.min(CONV),      # min CFL_e residual
            np.min(CONV2)      # min CFL_iter residual
        ])


        elapsed = time.time() - start_time
        h, m = divmod(elapsed // 60, 60)
        s = elapsed % 60
        print(f"{int(h):02}:{int(m):02}:{int(s):02} Case {count}: "
              f"NN = {len(CONV3)}, CFL_e = {len(CONV)}, CFL_iter = {len(CONV2)}")
        
        count += 1


OUTCOMES_df = pd.DataFrame(OUTCOMES, columns=[
    'uin', 'mesh', 'NN_NumIter', 'CFL_e_NumIter', 'CFL_iter_NumIter'
])

OUTCOMES2_df = pd.DataFrame(OUTCOMES2, columns=[
    'uin', 'mesh', 'NN_FinalError', 'CFL_e_MinResidual', 'CFL_iter_MinResidual'
])

with pd.ExcelWriter(outfile, engine='openpyxl', mode='w') as writer:
    OUTCOMES_df.to_excel(writer, sheet_name=sheet, index=False)
    OUTCOMES2_df.to_excel(writer, sheet_name=sheet, startcol=7, index=False)

print(f"✅ All simulations complete. Results written to {outfile}")