function [best_fitness,graph] = bso2(ProblemStruct)
% fun = fitness_function
% n_p; population size
% n_d; number of dimension
% n_c: number of clusters
% rang_l; left boundary of the dynamic range
% rang_r; right boundary of the dynamic range

n_p=ProblemStruct.PopSize;
n_d=ProblemStruct.Dim;
n_c=ProblemStruct.clusters;
rang_l=ProblemStruct.rang_l;
rang_r=ProblemStruct.rang_r;
max_iteration=ProblemStruct.max_iteration;
max_EFs=ProblemStruct.max_EFs;
objfun=ProblemStruct.FunctionName;
func_num=ProblemStruct.Func_num;

prob_one_cluster = 0.8; % probability for select one cluster to form new individual;
stepSize = ones(1,n_d); % effecting the step size of generating new individuals by adding random values
popu = rang_l + (rang_r - rang_l) * rand(n_p,n_d); % initialize the population of individuals
popu_copy= rang_l + (rang_r - rang_l) * rand(n_p,n_d); % initialize the population of individuals
popu_sorted  = rang_l + (rang_r - rang_l) * rand(n_p,n_d); % initialize the  population of individuals sorted according to clusters
n_iteration = 1; % current iteration number
% initialize cluster probability to be zeros
prob = zeros(n_c,1);
best = zeros(n_c,1);  % index of best individual in each cluster
centers = rang_l + (rang_r - rang_l) * rand(n_c,n_d);  % initialize best individual in each cluster
centers_copy = rang_l + (rang_r - rang_l) * rand(n_c,n_d);  % initialize best individual-COPY in each cluster FOR the purpose of introduce random best
best_fitness = 1000000*ones(max_iteration,1);
fitness_popu = 1000000*ones(n_p,1);  % 保存每个个体的适应度
fitness_popu_sorted = 1000000*ones(n_p,1);  % store  fitness value for each sorted individual
indi_temp = zeros(1,n_d);  % store temperary individual
EFs=0;
graph=[];
count=0;
select_stagnated_counter=zeros(n_p,1);%
stagnated_gbest=0;
%**************************************************************************
%**************************************************************************

% calculate fitness for each individual in the initialized population
ObjVal=feval(objfun,popu',func_num);
fitness_popu=ObjVal';
graph = [graph;0,min(fitness_popu)];
n_iteration=n_iteration+1;
% for idx = 1:n_p
%     fitness_popu(idx,1) = fun(popu(idx,:));
% end
while 1&&EFs < max_EFs
    OL_best_time=0;
    cluster = kmeans(popu, n_c,'Distance','cityblock','Start',centers,'EmptyAction','singleton'); % k-mean cluster
    % clustering
    fit_values = abs(max(fitness_popu)*10*ones(n_c,1));  % assign a initial big fitness value  as best fitness for each cluster in minimization problems
    number_in_cluster = zeros(n_c,1);  % initialize 0 individual in each cluster
    for idx = 1:n_p
        number_in_cluster(cluster(idx,1),1)= number_in_cluster(cluster(idx,1),1) + 1;
        % find the best individual in each cluster
        if fit_values(cluster(idx,1),1) > fitness_popu(idx,1)  % minimization
            fit_values(cluster(idx,1),1) = fitness_popu(idx,1);
            best(cluster(idx,1),1) = idx;
        end
    end
    % form population sorted according to clusters
    counter_cluster = zeros(n_c,1);  % initialize cluster counter to be 0
    acculate_num_cluster = zeros(n_c,1);  % initialize accumulated number of individuals in previous clusters
    for idx =2:n_c
        acculate_num_cluster(idx,1) = acculate_num_cluster((idx-1),1) + number_in_cluster((idx-1),1);
    end
    %start form sorted population
    for idx = 1:n_p
        counter_cluster(cluster(idx,1),1) = counter_cluster(cluster(idx,1),1) + 1 ;
        temIdx = acculate_num_cluster(cluster(idx,1),1) +  counter_cluster(cluster(idx,1),1);
        popu_sorted(temIdx,:) = popu(idx,:);
        fitness_popu_sorted(temIdx,1) = fitness_popu(idx,1);
    end
    % record the best individual in each cluster
    for idx = 1:n_c
        centers(idx,:) = popu(best(idx,1),:);
    end
    centers_copy = centers;  % make a copy
    
    if (rand() < 0.2) %  select one cluster center to be replaced by a randomly generated center
        cenIdx = ceil(rand()*n_c);
        centers(cenIdx,:) = rang_l + (rang_r - rang_l) * rand(1,n_d);
    end
    % calculate cluster probabilities based on number of individuals in
    % each cluster
    for idx = 1:n_c
        prob(idx,1) = number_in_cluster(idx,1)/n_p;
        if idx > 1
            prob(idx,1) = prob(idx,1) + prob(idx-1,1);
        end
    end
    index=find(fitness_popu==(min(fitness_popu(:,1))));
    gbest(1,:) = popu(index(1,:),:); %找到全局最优
    %best_value=fun(gbest)
    best_value=feval(objfun,gbest',func_num);
    % generate n_p new individuals by adding Gaussian random values
    for idx = 1:n_p
        stagnated_gbest_step=0;
        r_1 = rand();  % probability for select one cluster to form new individual
        if r_1 < prob_one_cluster % select one cluster
            idx_clu = cluster(idx,1);
            if rand() < 0.7  % use the center
                indi_temp(1,:) = centers(idx_clu,:);
            else % use one randomly selected  cluster
                indi_1 = acculate_num_cluster(idx_clu,1) + ceil(rand() * number_in_cluster(idx_clu,1));
                indi_temp(1,:) = popu_sorted(indi_1,:);
            end
            choose_select=1;
        else % select two clusters
            % pick two clusters
            cluster_1 = ceil(rand() * n_c);
            indi_1 = acculate_num_cluster(cluster_1,1) + ceil(rand() * number_in_cluster(cluster_1,1));
            cluster_2 = ceil(rand() * n_c);
            indi_2 = acculate_num_cluster(cluster_2,1) + ceil(rand() * number_in_cluster(cluster_2,1));
            tem = rand();
            if rand() < 0.7 %use center
                indi_temp(1,:) = tem * centers(cluster_1,:) + (1-tem) * centers(cluster_2,:);
            else   % use randomly selected individuals from each cluster
                indi_temp(1,:) = tem * popu_sorted(indi_1,:) + (1-tem) * popu_sorted(indi_2,:);
            end
            choose_select=2;
        end
        stepSize = logsig(((0.5*max_iteration - n_iteration)/20)) * rand(1,n_d);
        step_value= normrnd(0,1,1,n_d);
        origin=feval(objfun,indi_temp',func_num);
        if select_stagnated_counter(idx,1)>15&&OL_best_time<1&&rand()<(1-EFs/max_EFs)
            OL_best_time=OL_best_time+1
            choose_t=rand();
            if choose_t<=0.25
                t(1,:)=indi_temp(1,:)+rand()*(gbest(1,:)-indi_temp(1,:));%Convergence
            elseif choose_t>0.25&&choose_t<=0.5
                t(1,:)=rand()*(rang_r+rang_l)-indi_temp(1,:);   %Divergence
            elseif  choose_t>0.5&&choose_t<=0.75
                choose_centers= ceil(rand() * n_c);
                t(1,:)=indi_temp(1,:)+rand()*(centers(choose_centers,:)-indi_temp(1,:));  %Exploitation
            else
                popu_copy(find(ismember(popu_copy,indi_temp,'rows')),:)=[];%Exploration
                row_popu_copy=size(popu_copy,1);
                random_select=round(rand(1,1)*row_popu_copy);
                if random_select==0
                    random_select=1;
                end
                t(1,:)=popu_copy(random_select,:);
            end
            value_space=[indi_temp(1,:)',t(1,:)'];
            [indi_temp(1,:),EFs,count,graph]=OL_select_operation(n_d,value_space,objfun,func_num,EFs,max_EFs,count,graph,n_iteration,best_fitness);%调用对Xselect正交的函数
            indi_temp(1,:) = indi_temp(1,:) + stepSize .* step_value;
            if feval(objfun,indi_temp',func_num)<fitness_popu(idx,1)
                select_stagnated_counter(idx,1)=0;
            end
        else
            indi_temp(1,:) = indi_temp(1,:) + stepSize .* step_value;
        end
        if stagnated_gbest>4&&rand()<(1-EFs/max_EFs)
            if idx==index(1,:)
                if choose_select==1
                    one=ceil(rand()*n_p);
                    while 1
                        two=ceil(rand()*n_p);
                        if one~=two
                            break;
                        end
                    end
                    indi_one=popu(one,:);
                    indi_two=popu(two,:);
                    
                    if rand()<=0.5
                        r_step=stepSize;
                    else
                        r_step=rand(1,n_d).*(indi_one(1,:)-indi_two(1,:));
                    end
                    while stagnated_gbest_step<3
                        value_temp_new=[1,-1];
                        value_space_new=repmat(value_temp_new,n_d,1);
                        [indi_temp,EFs,r_step,count,graph]=OL_gbest_local_search(n_d,value_space_new,indi_temp,objfun,func_num,EFs,max_EFs,r_step,count,graph,n_iteration,best_fitness);%全局最优局部搜索
                        %    fun(indi_temp)
                        fv=feval(objfun,indi_temp',func_num);
                        if fv>best_value
                            stagnated_gbest_step=stagnated_gbest_step+1;
                            r_step=r_step.*rand(1,n_d);
                        else
                            stagnated_gbest=0;
                            break;
                        end
                    end
                elseif rand()<0.01
                    t(1,:)=indi_temp(1,:)+rand()*(gbest(1,:)-indi_temp(1,:));%Convergence
                    value_space=[indi_temp(1,:)',t(1,:)'];
                    [indi_temp(1,:),EFs,count,graph]=OL_select_operation(n_d,value_space,objfun,func_num,EFs,max_EFs,count,graph,n_iteration,best_fitness);%调用对Xselect正交的函数
                    indi_temp(1,:) = indi_temp(1,:) + stepSize .* step_value;
                end
                if feval(objfun,indi_temp',func_num)<best_value
                    stagnated_gbest=0;
                end
            end
        end
        % if better than the previous one, replace it
        fv=feval(objfun,indi_temp',func_num);
        EFs = EFs+1;
        if EFs > max_EFs
            return;
        end
        count=count+1;
        if count==50
            graph = [graph;EFs,best_fitness(n_iteration)];
            count=0;
        end
        if fv < fitness_popu(idx,1)  % better than the previous one, replace
            fitness_popu(idx,1) = fv;
            popu(idx,:) = indi_temp(1,:);
            select_stagnated_counter(idx,1)=0;
        else
            select_stagnated_counter(idx,1)=select_stagnated_counter(idx,1)+1;
        end
    end
    if feval(objfun,gbest',func_num)== min(fitness_popu(:,1))
        stagnated_gbest=stagnated_gbest+1;
    else
        stagnated_gbest=0;
    end
    for idx = 1:n_c
        popu(best(idx,1),:) = centers_copy(idx,:);
        fitness_popu(best(idx,1),1) = fit_values(idx,1);
    end
    n_iteration = n_iteration +1;
    %重出始化
    if stagnated_gbest>1/20*max_EFs
        for pi=1:n_p
            if select_stagnated_counter(pi,1)>=10
                if rand()<0.5
                    while 1
                        first=ceil(rand() * n_p);
                        second=ceil(rand() * n_p);
                        third=ceil(rand() * n_p);
                        if first~=second~=third
                            break;
                        end
                    end
                    indi_first=popu(first,:);
                    indi_second=popu(second,:);
                    indi_third=popu(third,:);
                    popu(pi,:)=indi_first+0.5*(indi_second-indi_third);
                else
                    popu(pi,:)=rang_l+rand(1,n_d)*(rang_r-rang_l);
                end
            end
        end
    end
    % record the best fitness in each iteration
    best_fitness(n_iteration, 1) = min(fit_values);
end
save (['Data\Rank_Sum\OLBSO_',ProblemStruct.funStr,'_',num2str(ProblemStruct.Dim),'_run',num2str(ProblemStruct.exp_num),'.mat']);


