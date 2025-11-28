import numpy as np
from mph import Model

def boundary_elements(coords_vertex2, coords_vertex, dat_tab):
    edge_length = np.linalg.norm(coords_vertex[:, 0:2] - coords_vertex[:, 2:4], axis=1)

    n = len(coords_vertex)
    dat_out = np.zeros((n, 38))  

    for i in range(n):
        cv = coords_vertex[i, :].astype(int)
        cv2 = coords_vertex2[i, :].astype(int)

        u1 = dat_tab[i, 3 + cv2[0]]
        u2 = dat_tab[i, 3 + cv2[1]]
        u3 = 0
        v1 = dat_tab[i, 6 + cv2[0]]
        v2 = dat_tab[i, 6 + cv2[1]]
        v3 = 0
        p1 = dat_tab[i, 9 + cv2[0]]
        p2 = dat_tab[i, 9 + cv2[1]]
        p3 = 0
        res_u1 = dat_tab[i, 12 + cv2[0]]
        res_u2 = dat_tab[i, 12 + cv2[1]]
        res_u3 = 0
        res_v1 = dat_tab[i, 15 + cv2[0]]
        res_v2 = dat_tab[i, 15 + cv2[1]]
        res_v3 = 0
        res_p1 = dat_tab[i, 18 + cv2[0]]
        res_p2 = dat_tab[i, 18 + cv2[1]]
        res_p3 = 0
        resu1 = dat_tab[i, 21 + cv2[0]]
        resu2 = dat_tab[i, 21 + cv2[1]]
        resu3 = 0
        resv1 = dat_tab[i, 24 + cv2[0]]
        resv2 = dat_tab[i, 24 + cv2[1]]
        resv3 = 0
        resp1 = dat_tab[i, 27 + cv2[0]]
        resp2 = dat_tab[i, 27 + cv2[1]]
        resp3 = 0

        Re_gem = 0
        itercount = dat_tab[i, 32]
        CFL = 0
        Cx = 0
        Cy = 0
        u_new = 0
        v_new = 0
        p_new = 0

        dat_out[i, :] = [
            edge_length[i], 0, 0,
            u1, u2, u3, v1, v2, v3, p1, p2, p3,
            res_u1, res_u2, res_u3, res_v1, res_v2, res_v3,
            res_p1, res_p2, res_p3, resu1, resu2, resu3,
            resv1, resv2, resv3, resp1, resp2, resp3,
            Re_gem, itercount, CFL, Cx, Cy, u_new, v_new, p_new
        ]

    return dat_out
