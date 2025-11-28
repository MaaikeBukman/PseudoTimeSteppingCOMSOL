import numpy as np
import torch
import mph
from GenerateData import gen_data


def make_predictions(model, network, sheet, iter):
    model_nn = torch.load(network, weights_only=False)
    model_nn.eval()

    sol = "sol5" if iter==1 else "sol2"

    inputs, _ = gen_data(model,iter,sol,1)
    in_put = torch.tensor(inputs, dtype = torch.float32)

    CFL = model_nn(in_put[0:124, :].T).detach().cpu().numpy().squeeze()
    Cx = in_put[124, :].detach().cpu().numpy()
    Cy = in_put[125, :].detach().cpu().numpy()

    np.savetxt("datagrid.txt", np.vstack([Cx, Cy, CFL]).T, fmt="%.6f %.6f %.6f")

    if  iter == 1:
        model.java.sol('sol2').runAll()
        model.java.sol('sol6').runAll()
        model.java.sol('sol5').runAll()
        model.java.sol('sol3').runAll()
        model.java.sol('sol4').runAll()

        model.java.sol('sol5').feature('s1').feature('fc1').set('niter','1')
        model.java.sol('sol5').feature('s1').feature('aDef').set('storeresidual', 'solvingandoutput')
        model.java.sol('sol5').runAll()
        
        model.java.study('std1').feature('stat').set('initmethod', 'sol')
        model.java.study('std1').feature('stat').set('initstudy','std1')
        model.java.sol('sol2').feature('s1').set('linpmethod', 'sol')
        model.java.sol('sol4').runAll()

        model.java.func('int1').active(True)
        model.java.component('comp1').func('int1').refresh()
        
        model.java.study('std1').feature('stat').set('initmethod', 'init')
        model.java.study('std1').feature('stat').set('initstudy','std2')
        model.java.sol('sol2').runAll()
    
        model.java.func('int1').active(True)
        model.java.component('comp1').func('int1').refresh()
        
        model.java.result('pg35').feature('surf1').set('expr', 'int1(x,y)')
        model.java.result('pg35').feature('surf1').set('rangecoloractive', 'on')
        
        model.java.result('pg33').feature('surf1').set('expr', 'spf.U')
        model.java.result('pg33').feature('surf1').set('rangecoloractive', 'on')


        tabl = model.java.result().table('tbl2')
        tabl_data = np.array(tabl.getReal(), dtype=float)
        CONV = float(tabl_data.flatten(order='F')[-1])

        model.java.study('std1').feature('stat').set('initmethod', 'sol')
        model.java.study('std1').feature('stat').set('initstudy','std1')
        model.java.sol('sol2').feature('s1').set('linpmethod', 'sol')
        outcome = model.evaluate(['spf.U'], dataset="Study 1//Solution 2")        
    else:
        model.java.func('int1').active(True)
        model.java.component('comp1').func('int1').refresh()
        model.java.sol('sol2').runAll()
        
        tabl = model.java.result().table('tbl2')
        tabl_data = np.array(tabl.getReal(), dtype=float)
        CONV = float(tabl_data.flatten(order='F')[-1])

        outcome = model.evaluate(['spf.U'], dataset="Study 1//Solution 2") 
    return CONV, outcome, CFL

