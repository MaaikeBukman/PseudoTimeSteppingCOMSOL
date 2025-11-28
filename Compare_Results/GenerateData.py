import numpy as np
import pandas as pd
from mph import Model
from BoundaryElements import boundary_elements  
from scipy.spatial import cKDTree

def gen_data(model, iter, sol, SheetNormParams):
    xmi = model.java.sol(sol).xmeshInfo('main')  

    nodes = xmi.nodes()
    coords = np.array(nodes.coords())            
    n_nodes = coords.shape[1]

    elems = xmi.elements('tri')
    tri_nodes = np.array(elems.nodes())           
    n_elems = tri_nodes.shape[1]

    mesh_info = np.zeros((7, n_elems))
    mesh_info[0, :] = np.arange(1, n_elems + 1)      
    mesh_info[1:4, :] = tri_nodes + 1                
    
    for i in range(n_elems):
        n1, n2, n3 = (tri_nodes[:, i]).astype(int)   
        p1 = coords[:, n1]
        p2 = coords[:, n2]
        p3 = coords[:, n3]
        mesh_info[4, i] = np.linalg.norm(p1 - p2)
        mesh_info[5, i] = np.linalg.norm(p1 - p3)
        mesh_info[6, i] = np.linalg.norm(p3 - p2)

    Cx = np.mean(coords[0, tri_nodes.astype(int)], axis=0)
    Cy = np.mean(coords[1, tri_nodes.astype(int)], axis=0)

    if sol == "sol5":
        data = model.evaluate(
            ["u2", "v2", "p2", "residual(u2)", "residual(v2)", "residual(p2)",
             "spf2.cellRe", "spf2.res_u", "spf2.res_v", "spf2.res_p"],
            dataset="Study 2//Solution 5"
        )
        model.java.sol("sol3").feature("s1").feature("aDef").set("storeresidual", "solvingandoutput")
        model.java.sol("sol3").runAll()
        data2 = model.evaluate(
            ["u2", "v2", "p2", "residual(u2)", "residual(v2)", "residual(p2)",
             "spf2.cellRe", "spf2.res_u", "spf2.res_v", "spf2.res_p"],
            dataset="Study 3//Solution 3"
        )
    else:
        data = model.evaluate(
            ["u", "v", "p", "residual(u)", "residual(v)", "residual(p)",
             "spf.cellRe", "spf.res_u", "spf.res_v", "spf.res_p"],
            dataset="Study 1//Solution 2"
        )
        model.java.sol("sol6").feature("s1").feature("aDef").set("storeresidual", "solvingandoutput")
        model.java.sol("sol6").runAll()
        data2 = model.evaluate(
            ["u", "v", "p", "residual(u)", "residual(v)", "residual(p)",
             "spf.cellRe", "spf.res_u", "spf.res_v", "spf.res_p"],
            dataset="Study 5//Solution 6"
        )

    u = np.array(data[0])
    v = np.array(data[1])
    p = np.array(data[2])
    res_u = np.array(data2[3])
    res_v = np.array(data2[4])
    res_p = np.array(data2[5])
    Re = np.array(data[6])
    resu = np.array(data[7])
    resv = np.array(data[8])
    resp = np.array(data[9])

    perm_mat = np.zeros(u.shape[0], dtype=int)
    for j in range(u.shape[0]):
        xi, yi = coords[0, j], coords[1, j]
        diffs = (coords[0, :] - xi) ** 2 + (coords[1, :] - yi) ** 2
        perm_mat[j] = np.argmin(diffs)

    u = u[perm_mat]
    v = v[perm_mat]
    p = p[perm_mat]
    res_u = res_u[perm_mat]
    res_v = res_v[perm_mat]
    res_p = res_p[perm_mat]
    Re = Re[perm_mat]
    resu = resu[perm_mat]
    resv = resv[perm_mat]
    resp = resp[perm_mat]

    u1 = u[tri_nodes[0, :].astype(int)]
    u2 = u[tri_nodes[1, :].astype(int)]
    u3 = u[tri_nodes[2, :].astype(int)]
    v1 = v[tri_nodes[0, :].astype(int)]
    v2 = v[tri_nodes[1, :].astype(int)]
    v3 = v[tri_nodes[2, :].astype(int)]
    p1 = p[tri_nodes[0, :].astype(int)]
    p2 = p[tri_nodes[1, :].astype(int)]
    p3 = p[tri_nodes[2, :].astype(int)]

    res_u1 = res_u[tri_nodes[0, :].astype(int)]
    res_u2 = res_u[tri_nodes[1, :].astype(int)]
    res_u3 = res_u[tri_nodes[2, :].astype(int)]
    res_v1 = res_v[tri_nodes[0, :].astype(int)]
    res_v2 = res_v[tri_nodes[1, :].astype(int)]
    res_v3 = res_v[tri_nodes[2, :].astype(int)]
    res_p1 = res_p[tri_nodes[0, :].astype(int)]
    res_p2 = res_p[tri_nodes[1, :].astype(int)]
    res_p3 = res_p[tri_nodes[2, :].astype(int)]

    Re_gem = np.mean(Re[tri_nodes.astype(int)], axis=0)

    eps = 1e-16
    resu1 = np.log(np.abs(resu[tri_nodes[0, :].astype(int)]) + eps)
    resu2 = np.log(np.abs(resu[tri_nodes[1, :].astype(int)]) + eps)
    resu3 = np.log(np.abs(resu[tri_nodes[2, :].astype(int)]) + eps)
    resv1 = np.log(np.abs(resv[tri_nodes[0, :].astype(int)]) + eps)
    resv2 = np.log(np.abs(resv[tri_nodes[1, :].astype(int)]) + eps)
    resv3 = np.log(np.abs(resv[tri_nodes[2, :].astype(int)]) + eps)
    resp1 = np.log(np.abs(resp[tri_nodes[0, :].astype(int)]) + eps)
    resp2 = np.log(np.abs(resp[tri_nodes[1, :].astype(int)]) + eps)
    resp3 = np.log(np.abs(resp[tri_nodes[2, :].astype(int)]) + eps)

    dat_tab = np.column_stack([
        mesh_info[4, :], mesh_info[5, :], mesh_info[6, :],   
        u1, u2, u3, v1, v2, v3, p1, p2, p3,                  
        res_u1, res_u2, res_u3, res_v1, res_v2, res_v3,      
        res_p1, res_p2, res_p3,                             
        resu1, resu2, resu3, resv1, resv2, resv3,            
        resp1, resp2, resp3,                                 
        Re_gem, Cx, Cy                                       
    ])

    information = pd.DataFrame(dat_tab)  

    X_coords = np.zeros((n_elems, 3))
    Y_coords = np.zeros((n_elems, 3))
    for i in range(n_elems):
        X_coords[i, 0] = coords[0, tri_nodes[0, i].astype(int)]
        X_coords[i, 1] = coords[0, tri_nodes[1, i].astype(int)]
        X_coords[i, 2] = coords[0, tri_nodes[2, i].astype(int)]
        Y_coords[i, 0] = coords[1, tri_nodes[0, i].astype(int)]
        Y_coords[i, 1] = coords[1, tri_nodes[1, i].astype(int)]
        Y_coords[i, 2] = coords[1, tri_nodes[2, i].astype(int)]

    n = X_coords.shape[0]

    mesh_inf = np.zeros((n, 4))
    coords_vertex = np.zeros((n, 4))
    coords_vertex2 = np.zeros((n, 4))

    centroids = np.column_stack((X_coords.mean(axis=1), Y_coords.mean(axis=1)))
    tree = cKDTree(centroids)

    radius = np.max(np.ptp(X_coords, axis=1)) * 1.01

    for i in range(n):
        mesh_iter = np.zeros((3, 4))
        jiter = 0
        
        candidates = tree.query_ball_point(centroids[i], r=radius)
        for j in candidates:
            if i == j:
                continue
            
            match_matrix = np.isclose(X_coords[i][:, None], X_coords[j][None, :]) & \
                        np.isclose(Y_coords[i][:, None], Y_coords[j][None, :])
            h = np.sum(match_matrix)
            
            if h >= 2 and jiter < 3:
                k_idx, l_idx = np.argwhere(match_matrix)[0]
                mesh_iter[jiter, 0] = i
                mesh_iter[jiter, 1] = j
                mesh_iter[jiter, 2] = k_idx
                mesh_iter[jiter, 3] = l_idx
                jiter += 1
        
        mesh_inf[i, 0] = i
        mesh_inf[i, 1:4] = mesh_iter[:3, 1]

        if mesh_iter[2, 1] == 0 and mesh_iter[1, 1] != 0:
            countnum = 0
            neigh1 = int(mesh_iter[0, 1])
            neigh2 = int(mesh_iter[1, 1])
            
            for k in range(3):
                countiter = 0
                k_iter, l_iter = -1, -1
                
                match1 = np.isclose(X_coords[i, k], X_coords[neigh1]) & np.isclose(Y_coords[i, k], Y_coords[neigh1])
                if np.any(match1):
                    countiter += 1
                    l_iter = np.argmax(match1)
                    k_iter = k
                    
                match2 = np.isclose(X_coords[i, k], X_coords[neigh2]) & np.isclose(Y_coords[i, k], Y_coords[neigh2])
                if np.any(match2):
                    countiter += 1
                    l_iter = np.argmax(match2)
                    k_iter = k
                
                if countiter == 1:
                    coords_vertex[i, countnum] = X_coords[i, k]
                    coords_vertex[i, countnum + 1] = Y_coords[i, k]
                    coords_vertex2[i, countnum] = k_iter
                    coords_vertex2[i, countnum + 1] = l_iter
                    countnum += 2

    mesh_information = mesh_inf.copy()

    dat_tab = boundary_elements(coords_vertex2, coords_vertex, dat_tab)

    tbl = np.zeros((mesh_information.shape[0], 126))
    information_array = information.to_numpy()

    for i in range(mesh_information.shape[0]):
        neigh1 = int(mesh_information[i, 1])
        neigh2 = int(mesh_information[i, 2])
        neigh3 = int(mesh_information[i, 3])

        if neigh3 != 0:  #
            tbl[i, :] = np.concatenate([
                information_array[i, :31],                   
                information_array[neigh1, :31],             
                information_array[neigh2, :31],               
                information_array[neigh3, :31],               
                information_array[i, 31:33]                   
            ])
        elif neigh2 != 0:  
            tbl[i, :] = np.concatenate([
                information_array[i, :31],
                information_array[neigh1, :31],
                information_array[neigh2, :31],
                dat_tab[i, :31],          
                information_array[i, 31:33]
            ])
        else:
            tbl[i, :] = np.zeros((126,))

    tbl_df = pd.DataFrame(tbl)
    tbl_df.to_excel("PatchDataStep.xlsx", index=False)
    tbl_df = pd.read_excel("PatchDataStep.xlsx", sheet_name=0)

    NormParams = pd.read_excel("NormalizationParams", sheet_name=SheetNormParams) # Add here correct name NormalizationParams excel + path

    # C and S (centers and scales)
    C = NormParams.iloc[0, :124].astype(float).values
    S = NormParams.iloc[1, :124].astype(float).values

    S[S == 0] = 1.0

    # Normalize first 124 columns
    tbl_df.iloc[:, :124] = (tbl_df.iloc[:, :124] - C) / S

    inputs = tbl_df.to_numpy().T
    input_size = tbl_df.shape[0]

    return inputs, input_size
