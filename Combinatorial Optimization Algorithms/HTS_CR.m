clear all
load('results_1_final.mat')

nEvCu       = [];
bestEvCu    = [];
bestEvCuScr = [];
bestEvCuReg = [];

%--------------------------------------------------------------------------
%create all couples

cps         = [];
cps_p       = [];
cps_string  = [];

for cp1 = 0:numSources-1
    for cp2 = 0:numSources-1
        if cp1 ~= cp2
            cps         = [cps; {[cp1, cp2]}];
            cps_string  = [cps_string; {cv_string(cps(end))}];
            
            cps_p = [cps_p ; find(strcmp(TAGS_Reg, {cv_string(cps(end))} ))];
        end
    end
end

[cps_p_sorted, cps_p_index] = sort(cps_p);

cps = cps(cps_p_index);

%--------------------------------------------------------------------------

couple_scores = zeros(numSources,2);

[r c] = size(cps);

for t = 1 : r
    score = t;
    
    currentTag = cell2mat(cps(t));%select one tag at a time and convert it to numeric array
    
    couple_scores(currentTag(1) + 1, 1)     = couple_scores(currentTag(1) + 1, 1) + score;
    couple_scores(currentTag(end) + 1, 2)   = couple_scores(currentTag(end) + 1, 2) + score;
    
end

sources = 0:numSources;

heads_scores = couple_scores(:,1);
[h_s_s, h_i] = sort(heads_scores);

tails_scores = couple_scores(:,2);
[t_s_s, t_i] = sort(tails_scores);

sources_h = sources(h_i).';
sources_t = sources(t_i).';

%------------------------ HYPERPARAMETERS ---------------------------------
h_fraction = 1;
t_fraction = 1;
%--------------------------------------------------------------------------

nEvCu       = [nEvCu; size(cps,1)];
bestEvCu    = [bestEvCu; cps(1)];
bestEvCuScr = [bestEvCuScr; cps_p_sorted(1)];
bestEvCuReg = [bestEvCuReg; Reg(bestEvCuScr(end))];

candidate = [];

for hyper = 1:2*(numSources - 1)
       
    if mod(hyper,2) == 1
        h_fraction = h_fraction + 1;
    else
        t_fraction = t_fraction + 1;
    end
    
    heads = sources_h(1:h_fraction);
    tails = sources_t(1:t_fraction);

    for l = 3:currMaxLen
        for cmb_h = 1:h_fraction
            hd = heads(cmb_h);

            for cmb_t = 1:t_fraction
                tl = tails(cmb_t);

                if hd ~= tl
                    left_overs = unique([heads; tails]);
                    left_overs(find(left_overs == hd)) = [];
                    left_overs(find(left_overs == tl)) = [];

                    if (size(left_overs,1) ~= 0) && (size(left_overs,1) ~= 1)
                        left_overs_combinations = permsNK(left_overs, l-2);

                        left_overs_combinations = left_overs;
                    end
                    

                    for elem = 1:size(left_overs_combinations)
                        cv = [hd, left_overs_combinations(elem, :) , tl];
                        candidate = [candidate; {cv_string({cv})}];
                    end
                end
            end
        end
        
        candidate = [cps_string; candidate];

        candidate           = unique(candidate);
        candidate_scores    = [];

        for sc = 1:size(candidate)
            candidate_scores    = [candidate_scores; find(strcmp(TAGS_Reg, candidate(sc)))];
        end

        [cs_sorted, cs_index] = sort(candidate_scores);

        candidate = candidate(cs_index);

        nEvCu       = [nEvCu; size(candidate,1)];
        bestEvCu    = [bestEvCu; {candidate(1)}];
        bestEvCuScr = [bestEvCuScr; cs_sorted(1)];    
        bestEvCuReg = [bestEvCuReg; Reg(bestEvCuScr(end))];
        
    end
    
end

plot(nEvCu,-bestEvCuScr)

toSave = [nEvCu, bestEvCuScr, bestEvCuReg];