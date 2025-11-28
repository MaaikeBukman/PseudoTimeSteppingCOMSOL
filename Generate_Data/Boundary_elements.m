function dat_tab = Boundary_elements(coords_vertex, coords_vertex2, sheet, filename)

table = readtable(filename, 'TextType', 'String', 'Sheet', sheet);
table = table2array(table);
nRows = size(coords_vertex,1);
nCols = size(table,2);

edge_length = vecnorm([coords_vertex(:,1) - coords_vertex(:,3), ...
                       coords_vertex(:,2) - coords_vertex(:,4)], 2, 2);

u1 = zeros(nRows,1); u2 = zeros(nRows,1); u3 = zeros(nRows,1);
v1 = u1; v2 = u2; v3 = u3; p1 = u1; p2 = u2; p3 = u3;
res_u1 = u1; res_u2 = u2; res_u3 = u3; res_v1 = u1; res_v2 = u2; res_v3 = u3;
res_p1 = u1; res_p2 = u2; res_p3 = u3; resu1 = u1; resu2 = u2; resu3 = u3;
resv1 = u1; resv2 = u2; resv3 = u3; resp1 = u1; resp2 = u2; resp3 = u3;
Re_gem = zeros(nRows,1); CFL = Re_gem; Cx = Re_gem; Cy = Re_gem;
u_new = Re_gem; v_new = Re_gem; p_new = Re_gem; itercount = Re_gem;

for i = 1:nRows
    idx2 = coords_vertex(i,2);
    idx4 = coords_vertex(i,4);
    if idx2 < 1 || idx2 > 3 || isnan(idx2), idx2 = 1; end
    if idx4 < 1 || idx4 > 3 || isnan(idx4), idx4 = 2; end

    if i > size(table,1), break; end
    if (27+idx2) > nCols
        warning('Row %d skipped: invalid column index', i);
        continue;
    end

    u1(i) = table(i,3+idx2);
    u2(i) = table(i,3+idx4);
    v1(i) = table(i,6+idx2);
    v2(i) = table(i,6+idx4);
    p1(i) = table(i,9+idx2);
    p2(i) = table(i,9+idx4);
    res_u1(i) = table(i,12+idx2);
    res_u2(i) = table(i,12+idx4);
    res_v1(i) = table(i,15+idx2);
    res_v2(i) = table(i,15+idx4);
    res_p1(i) = table(i,18+idx2);
    res_p2(i) = table(i,18+idx4);
    resu1(i) = table(i,21+idx2);
    resu2(i) = table(i,21+idx4);
    resv1(i) = table(i,24+idx2);
    resv2(i) = table(i,24+idx4);
    resp1(i) = table(i,27+idx2);
    resp2(i) = table(i,27+idx4);
    itercount(i) = table(i,32);
end

dat_tab = [edge_length, zeros(nRows,2), u1, u2, u3, v1, v2, v3, ...
           p1, p2, p3, res_u1, res_u2, res_u3, res_v1, res_v2, res_v3, ...
           res_p1, res_p2, res_p3, resu1, resu2, resu3, resv1, resv2, resv3, ...
           resp1, resp2, resp3, Re_gem, itercount, CFL, Cx, Cy, u_new, v_new, p_new];
end


