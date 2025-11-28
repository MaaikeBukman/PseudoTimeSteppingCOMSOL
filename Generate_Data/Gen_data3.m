%Here, data from one single element are gathered. After that,
%GenPatchData3.m can be run in order to obtain patches.


clear all 
%model = mphload('backstepopt4c.mph'); %load COMSOL model
%model = mphload('cavityopt4c.mph');
model = mphload("\\nl-filer1\users$\maaike\Desktop\backstepopt4c2.mph");

filename = 'DataSet32_final.xlsx'; %Filename for dataset

steps = [1,10,2,10,3,10]; %Which nonlinear steps  to run


%For loop over the steps to generate data
for j = 1:length(steps)
    numiter = num2str(j);
    disp(numiter)

    sol_mat = 1;% this is only adapted in special cases

    %For each step in steps, determine inflow velocity and max. element
    %size 
    if j <= 2
        model.param.set('uin', '0.05[m/s]');
        model.study('std4').feature('stat').set('plistarr', '0.01 0.05');%obtain "exact" solution
        model.mesh('mesh1').feature('size').set('hmax', '0.00156');
        model.sol('sol4').runAll;
        sol_mat = 1;
        %model.study('std1').feature('stat').set('listsolnum', 1); %If you
        %are using an auxiliary sweep in COMSOL
    elseif j<=4
        model.param.set('uin', '0.1[m/s]');
        model.study('std4').feature('stat').set('plistarr', '0.01 0.05 0.1');
        model.mesh('mesh1').feature('size').set('hmax', '0.00186');
        model.sol('sol4').runAll;
        sol_mat = 1;
        %model.study('std1').feature('stat').set('listsolnum', 2); %If you
        %are using an auxiliary sweep in COMSOL
    elseif j <= 6
        model.param.set('uin', '0.13[m/s]');
        model.study('std4').feature('stat').set('plistarr', '0.01 0.05 0.1 0.13');
        model.mesh('mesh1').feature('size').set('hmax', '0.00126');
        model.sol('sol4').runAll;
        %model.study('std1').feature('stat').set('listsolnum', 3); %If you
        %are using an auxiliary sweep in COMSOL
    end
    
    step = steps(j);
    model.sol('sol1').feature('s1').feature('fc1').set('niter', num2str(step));%Set the number of iterations
    model.sol('sol1').feature('s1').feature('aDef').set('storeresidual', 'solvingandoutput');%Store residual

    try
        model.sol('sol1').runAll;
    catch ME
        warning('Optimization failed at j=%d: %s', j, ME.message);
        continue
    end


    %model.sol('sol1').runAll; %Run COMSOL model 

    %Start optimization study
    disp('Beginning to optimize')
    try
        model.sol('sol6').runAll;
    catch ME
        warning('Optimization failed at j=%d: %s', j, ME.message);
        continue
    end    
    %model.sol('sol6').runAll;
    disp('Optimization done')
    
    %Obtain mesh information of simulation
    info = mphxmeshinfo(model); 
    mesh_info = zeros(4,size(info.elements.tri.nodes,2));
    coords = info.nodes.coords; 
    
    %Obtain node labels
    mesh_info(1,:) = (1:size(info.elements.tri.nodes,2));
    mesh_info(2,:) = info.elements.tri.nodes(1,:)+1;
    mesh_info(3,:) = info.elements.tri.nodes(2,:)+1;
    mesh_info(4,:) = info.elements.tri.nodes(3,:)+1;
    
    %Compute edge length
    for i = 1:size(info.elements.tri.nodes,2)
        mesh_info(5,i) = norm( [coords(1,mesh_info(2,i)) coords(2,mesh_info(2,i))]-[coords(1,mesh_info(3,i)) coords(2,mesh_info(3,i))] );
        mesh_info(6,i) = norm( [coords(1,mesh_info(2,i)) coords(2,mesh_info(2,i))]-[coords(1,mesh_info(4,i)) coords(2,mesh_info(4,i))] );
        mesh_info(7,i) = norm( [coords(1,mesh_info(4,i)) coords(2,mesh_info(4,i))]-[coords(1,mesh_info(3,i)) coords(2,mesh_info(3,i))] );
    end
    
    %Compute centroids
    Cx = 1/3*(coords(1,mesh_info(2,:))+coords(1,mesh_info(3,:))+coords(1,mesh_info(4,:)));
    Cy = 1/3*(coords(2,mesh_info(2,:))+coords(2,mesh_info(3,:))+coords(2,mesh_info(4,:)));
    
    %Create permutation matrix, because the different COMSOL information
    %functions do not match out coordinate-wise
    outcome = mpheval(model,{'spf.res_u'},'refine',1);
    perm_mat = zeros(1,length(outcome.p(1,:)));
    for k = 1:length(outcome.p(1,:))
        for i = 1:length(outcome.p(1,:))
            if outcome.p(1,k) == info.nodes.coords(1,i) && outcome.p(2,k) == info.nodes.coords(2,i)
                perm_mat(k) = i;
                break
            end
        end
    end


    % Obtain input data information
    data = mpheval(model,{'u2','v2','p2','residual(u2)','residual(v2)','residual(p2)','spf2.cellRe',...
        'spf2.res_u','spf2.res_v','spf2.res_p'},'refine',1,'dataset','dset1');
    
    % Apply permutation matrix to obtain correct solution order
    u = data.d1(sol_mat,perm_mat);
    v = data.d2(sol_mat,perm_mat);
    p = data.d3(sol_mat,perm_mat);
    Re = data.d7(sol_mat,perm_mat);

    u1 = u(mesh_info(2,:));
    u2 = u(mesh_info(3,:));
    u3 = u(mesh_info(4,:));
    v1 = v(mesh_info(2,:));
    v2 = v(mesh_info(3,:));
    v3 = v(mesh_info(4,:));
    p1 = p(mesh_info(2,:));
    p2 = p(mesh_info(3,:));
    p3 = p(mesh_info(4,:));
    Re_gem = (Re(mesh_info(2,:))+Re(mesh_info(3,:))+Re(mesh_info(4,:)))/3;
    u_new = griddata(data.p(1,:),data.p(2,:),data.d1(sol_mat,:),Cx,Cy);%Not really used
    v_new = griddata(data.p(1,:),data.p(2,:),data.d2(sol_mat,:),Cx,Cy);%Not really used
    p_new = griddata(data.p(1,:),data.p(2,:),data.d3(sol_mat,:),Cx,Cy);%Not really used

    itercount = steps(j)*ones(size(u_new,1),size(u_new,2));%Used in table iteration count


    data = mpheval(model,'cfl2','refine',1,'dataset','dset8');%Obtain optimised CFL
    CFL = griddata(data.p(1,:),data.p(2,:),data.d1,Cx,Cy);

    
    % Run COMSOL again, because residuals are from previous iteration
    model.sol('sol1').feature('s1').feature('fc1').set('niter', num2str(step+1));
    model.sol('sol1').runAll;
    data = mpheval(model,{'u2','v2','p2','residual(u2)','residual(v2)','residual(p2)','spf2.cellRe',...
        'spf2.res_u','spf2.res_v','spf2.res_p'},'refine',1,'dataset','dset1');
    % Obtain correct residuals
    res_u = data.d4(sol_mat,perm_mat);
    res_v = data.d5(sol_mat,perm_mat);
    res_p = data.d6(sol_mat,perm_mat);
    resu = data.d8(sol_mat,perm_mat);
    resv = data.d9(sol_mat,perm_mat);
    resp = data.d10(sol_mat,perm_mat);

    % All residuals on all nodes
    resu1 = resu(mesh_info(2,:));
    resu2 = resu(mesh_info(3,:));
    resu3 = resu(mesh_info(4,:));
    resv1 = resv(mesh_info(2,:));
    resv2 = resv(mesh_info(3,:));
    resv3 = resv(mesh_info(4,:));
    resp1 = resp(mesh_info(2,:));
    resp2 = resp(mesh_info(3,:));
    resp3 = resp(mesh_info(4,:));
    res_u1 = res_u(mesh_info(2,:));
    res_u2 = res_u(mesh_info(3,:));
    res_u3 = res_u(mesh_info(4,:));
    res_v1 = res_v(mesh_info(2,:));
    res_v2 = res_v(mesh_info(3,:));
    res_v3 = res_v(mesh_info(4,:));
    res_p1 = res_p(mesh_info(2,:));
    res_p2 = res_p(mesh_info(3,:));
    res_p3 = res_p(mesh_info(4,:));


    %Write data of single element to table
        dat_tab = [mesh_info(5,:)', mesh_info(6,:)',mesh_info(7,:)', u1', u2', u3', v1', v2', v3',p1', p2', p3',...
            res_u1',res_u2',res_u3',res_v1',res_v2',res_v3',res_p1',res_p2',res_p3',resu1',resu2',...
            resu3',resv1',resv2',resv3',resp1',resp2',resp3',Re_gem',itercount',CFL', Cx',Cy',u_new',v_new',p_new']; %32 inputs & 6 outputs 
    
       
    dat_tab = array2table(dat_tab);    
    try
        disp('Writing to table...')
        writetable(dat_tab, filename, 'Sheet', j);
        disp('Table done')
    catch ME
        warning('Failed at j = %d: %s. Writing empty sheet.', j, ME.message);
        
        % Write an empty sheet so sheet numbering remains consistent
        empty_table = array2table([]);
        writetable(empty_table, filename, 'Sheet', j);
        
    end
end


table = readtable(filename,'TextType','String','Sheet',j);
table = table2array(table);
new_table = table;


for j = 2:length(steps)
    table = readtable(filename,'TextType','String','Sheet',j);
    table = table2array(table);
    new_table = [new_table; table];
end
sheet = length(steps)+1;
new_table = array2table(new_table);
writetable(new_table,filename,'Sheet',sheet);
