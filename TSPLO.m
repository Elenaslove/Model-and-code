clc;clear;
close all
function [Bestp,Bests,Ccurve]=TSPLO(N,MaxFEs,lb,ub,dim,Maiter,fobj)
FEs = 0;
it = 1;
fitness=inf*ones(N,1);
fitnew=inf*ones(N,1);
X=initon1(N,dim,ub,lb);
V=ones(N,dim);
X_new=zeros(N,dim);
Ccurve=zeros(1,MaxFEs/N);
for i=1:N
    fitness(i)=fobj(X(i,:));
    FEs=FEs+1;
end
[fitness, SortOrder]=sort(fitness);
X=X(SortOrder,:);
Bestpos=X(1,:);
Bests=fitness(1);
Ccurve(it)=Bests;
    while it < Maiter
        X_sum=sum(X,1);
        X_mean=X_sum/N;
        w1=tansig((FEs/MaxFEs)^4);
        w2=exp(-(2*FEs/MaxFEs)^3); 
        ik=it/Maiter;
        for i=1:N
            a=rand()/2+1;
            V(i,:)=1*exp((1-a)/100*FEs);
            LS=V(i,:);
            GS=chaos(ik)*Levy(dim).*(X_mean-X(i,:))+lb/2+rand(1,dim).*(ub-lb)/2;
            X_new(i,:)=X(i,:)+(w1*LS+w2*GS).*rand(1,dim);
        end
        for i=1:N
            E =sqrt(FEs/MaxFEs);
            for j=1:dim
                if (rand < 0.05) && (rand < E) 
                    A_neighbour = randperm(N, 2); 
                    X_new(i,j) = X(i,j) + sin(rand*pi) * (X(i,j) - X(A_neighbour(1),j)) + cos(rand*pi) * (X(i,j) - X(A_neighbour(2),j));
                end
            end
            Flag4ub = X_new(i,:) > ub;
            Flag4lb = X_new(i,:) < lb;
            X_new(i,:) = (X_new(i,:) .* ~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb;
            fitnew(i) = fobj(X_new(i,:));
            FEs = FEs + 1;
            if fitnew(i) < fitness(i)
                X(i,:) = X_new(i,:);
                fitness(i) = fitnew(i);
            end
       end
        [fitness, SortOrder]=sort(fitness);
        X=X(SortOrder,:);
        if fitness(1)<Bests
            Bestpos=X(1,:);
            Bests=fitness(1);
        end
        it = it + 1;
        Ccurve(it)=Bests;
        Bestp=Bestpos;
    end
end
function chaos_num = chaos(ik)
    r = 2-ik;  
    x = rand();  
    if x < 1/r
        x = r * x;
    else
        x = (1 - x)/(1-1/r);
    end
    chaos_num = x;
end

function Positions =initon1 (SearchAgents, dim, ub, lb)
    sobol_seq = sobolset(dim);
    sobol_seq = scramble(sobol_seq,'MatousekAffineOwen'); 
    sobol_points = net(sobol_seq, SearchAgents); 
    Positions = lb + (ub - lb) .* sobol_points;
end
function o=Levy(d)
    beta=1.5;
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,d)*sigma;v=randn(1,d);
    step=u./abs(v).^(1/beta);
    o=step;
end

