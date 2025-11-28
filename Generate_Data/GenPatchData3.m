% -------------------------------------------------------------------------
% GenPatchData3_safe.m  — Robust version to avoid index-out-of-bounds
% -------------------------------------------------------------------------

clearvars -except
filename     =  "C:\Users\maaike\.comsol\v63\llmatlab\DataSet32_final.xlsx";     % File with single element data from Gen_data3.m
new_filename = 'DataSet32Patch3_final.xlsx';% New filename to write patches
tablength    = 2;                     % Expected number of input sheets to process

% Load COMSOL model (same as you had)
model = mphload("\\nl-filer1\users$\maaike\Desktop\backstepopt4c2.mph");


% Obtain mesh information of simulation
info = mphxmeshinfo(model);
mesh_info = zeros(4, size(info.elements.tri.nodes,2));
coords = info.nodes.coords;

mesh_info(1,:) = 1:size(info.elements.tri.nodes,2);
mesh_info(2,:) = info.elements.tri.nodes(1,:)+1;
mesh_info(3,:) = info.elements.tri.nodes(2,:)+1;
mesh_info(4,:) = info.elements.tri.nodes(3,:)+1;

% Centroids
Cx = (coords(1,mesh_info(2,:)) + coords(1,mesh_info(3,:)) + coords(1,mesh_info(4,:))) / 3;
Cy = (coords(2,mesh_info(2,:)) + coords(2,mesh_info(3,:)) + coords(2,mesh_info(4,:))) / 3;

% Build X_coords, Y_coords with correct per-element assignment
nElem = length(Cx);
X_coords = zeros(nElem,3);
Y_coords = zeros(nElem,3);
for i = 1:nElem
    X_coords(i,1) = coords(1,mesh_info(2,i));
    X_coords(i,2) = coords(1,mesh_info(3,i));
    X_coords(i,3) = coords(1,mesh_info(4,i));
    Y_coords(i,1) = coords(2,mesh_info(2,i));
    Y_coords(i,2) = coords(2,mesh_info(3,i));
    Y_coords(i,3) = coords(2,mesh_info(4,i));
end

% Initialize helper arrays
coords_vertex = zeros(nElem,4);   % coordinates of unique vertex (boundary)
coords_vertex2 = zeros(nElem,4);  % vertex indices mapping for boundary handling
mesh_iter_all = zeros(nElem,3,4); % for debugging if needed
mesh_inf = zeros(nElem,4);        % central + up to 3 neighbors

% --- find neighbors (same logic you had, but safer) ---
for i = 1:nElem
    mesh_iter = zeros(3,4); jiter = 1;
    for j = 1:nElem
        if i == j, continue; end
        common = 0;
        idxs = zeros(2,1); % not used beyond detection
        for k = 1:3
            for l = 1:3
                if X_coords(i,k) == X_coords(j,l) && Y_coords(i,k) == Y_coords(j,l)
                    common = common + 1;
                    if common <= 2
                        mesh_iter(common,:) = [i, j, k, l];
                    end
                end
            end
        end
        if common >= 2
            jiter = jiter + 1; % marker, mesh_iter already set
        end
    end
    mesh_inf(i,1) = i;
    % mesh_iter might have zeros for missing neighbors
    mesh_inf(i,2) = mesh_iter(1,2);
    mesh_inf(i,3) = mesh_iter(2,2);
    mesh_inf(i,4) = mesh_iter(3,2);
    mesh_iter_all(i,:,:) = mesh_iter; %#ok<AGROW>
    
    % boundary-case assembly for coords_vertex / coords_vertex2 (as before)
    if mesh_iter(3,2) == 0 && mesh_iter(2,2) ~= 0
        countnum = 1;
        for k = 1:3
            countiter = 0;
            for l = 1:3
                if X_coords(i,k) == X_coords(mesh_iter(1,2),l) && Y_coords(i,k) == Y_coords(mesh_iter(1,2),l)
                    countiter = countiter + 1; k_iter = k; l_iter = l;
                end
            end
            for l = 1:3
                if X_coords(i,k) == X_coords(mesh_iter(2,2),l) && Y_coords(i,k) == Y_coords(mesh_iter(2,2),l)
                    countiter = countiter + 1; k_iter = k; l_iter = l;
                end
            end
            if countiter == 1
                coords_vertex(i,countnum)   = X_coords(i,k);
                coords_vertex(i,countnum+1) = Y_coords(i,k);
                coords_vertex2(i,countnum)   = k_iter;
                coords_vertex2(i,countnum+1) = l_iter;
                countnum = countnum + 2;
            end
        end
    end
end

% Keep a copy for logic checking (mesh_information used for neighbor counts)
mesh_information = mesh_inf;

% Determine which sheet contains the long 'information' table we need.
% Often your combined table is on the last sheet (tablength+1) but not always.
% We'll search for a sheet whose number of rows matches nElem (or is largest).
sheetCandidate = [];
maxRows = 0;
for s = 1:max(1,tablength+1)
    try
        tmp = readtable(filename,'TextType','String','Sheet',s);
        r = size(tmp,1);
        if r >= nElem
            sheetCandidate = s;
            break
        end
        if r > maxRows
            maxRows = r; sheetCandidate = s;
        end
    catch
        % skip invalid sheets
    end
end

if isempty(sheetCandidate)
    error('Could not find a suitable sheet in %s. Check file.', filename);
end

fprintf('Using sheet %d as source of "information" (rows=%d). mesh elements = %d\n', ...
    sheetCandidate, maxRows, nElem);

% Now loop over the (original) sheets we want to convert to patches.
final_table = [];
totalSkipped = 0;
totalCreated = 0;

for j = 1:tablength
    % Read data for this sheet (this is the per-element table you generated earlier)
    try
        table = readtable(filename,'TextType','String','Sheet',j);
    catch ME
        error('Cannot read sheet %d: %s', j, ME.message);
    end
    information = table2array(table);
    infoRows = size(information,1);
    infoCols = size(information,2);

    % dat_tab = boundary approximations (ghost elements)
    dat_tab = Boundary_elements(coords_vertex2, coords_vertex, j, filename);
    dat_tab = double(dat_tab); % ensure numeric

    new_table = zeros(nElem, 131);

    % For each element: assemble a patch but validate neighbor indices.
    for i = 1:nElem
        % neighbor indices from mesh_inf
        idx1 = mesh_inf(i,1);
        idx2 = mesh_inf(i,2);
        idx3 = mesh_inf(i,3);
        idx4 = mesh_inf(i,4);

        % helper that returns a 1x31 block given an index (or fallback)
        b1 = getBlock(idx1, information, infoRows, infoCols, dat_tab, i);
        b2 = getBlock(idx2, information, infoRows, infoCols, dat_tab, i);
        b3 = getBlock(idx3, information, infoRows, infoCols, dat_tab, i);
        b4 = getBlock(idx4, information, infoRows, infoCols, dat_tab, i);


        % Assemble depending on neighbor availability (use getBlock to be safe)
        try
            if idx4 ~= 0 % 3 neighbours expected
                b1 = getBlock(idx1);
                b2 = getBlock(idx2);
                b3 = getBlock(idx3);
                b4 = getBlock(idx4);
                tail = zeros(1,7);
                if idx1 >=1 && idx1 <= infoRows && infoCols >= 38
                    tail = information(idx1,32:38); % use central element tail if available
                end
                new_row = [b1, b2, b3, b4, tail];
                totalCreated = totalCreated + 1;
            elseif idx3 ~= 0 % 2 neighbours
                b1 = getBlock(idx1);
                b2 = getBlock(idx2);
                b3 = getBlock(idx3);
                b4 = getBlock(0); % dat_tab fallback via getBlock
                tail = zeros(1,7);
                if idx1 >=1 && idx1 <= infoRows && infoCols >= 38
                    tail = information(idx1,32:38);
                end
                new_row = [b1, b2, b3, b4, tail];
                totalCreated = totalCreated + 1;
            else % 1 or 0 neighbours (rare)
                b1 = getBlock(idx1);
                b2 = getBlock(idx2);
                b3 = getBlock(idx3);
                b4 = getBlock(idx4);
                tail = zeros(1,7);
                if idx1 >=1 && idx1 <= infoRows && infoCols >= 38
                    tail = information(idx1,32:38);
                end
                new_row = [b1, b2, b3, b4, tail];
                totalCreated = totalCreated + 1;
            end
            new_table(i,:) = double(new_row);
        catch ME
            % If assembly fails, insert a zero-row and increment skipped
            new_table(i,:) = zeros(1,131);
            totalSkipped = totalSkipped + 1;
            % no crash
        end
    end

    % Append to final_table
    final_table = [final_table; new_table]; %#ok<AGROW>

    % write intermediate sheet
    writetable(array2table(new_table), new_filename, 'Sheet', j);
    fprintf('Written sheet %d (created %d rows, skipped %d rows) \n', j, totalCreated, totalSkipped);
end

% write merged sheet
writetable(array2table(final_table), new_filename, 'Sheet', tablength+1);

fprintf('All done. totalCreated=%d, totalSkipped=%d. Output: %s\n', totalCreated, totalSkipped, new_filename);
 
function block = getBlock(idx, information, infoRows, infoCols, dat_tab, i)
    if idx >= 1 && idx <= infoRows && infoCols >= 31
        block = information(idx, 1:31);
    else
        if i <= size(dat_tab,1)
            block = dat_tab(i, 1:31);
        else
            block = zeros(1,31);
        end
    end
end
