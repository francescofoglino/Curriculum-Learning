%This script executes the HTS_CR algorithm. 
%This code cannot be used for running the algorithhm online but it is meant 
%to work offline after having collected all the curricula performance in a 
%dataset.
%
%In our case the dataset is a .mat file named 'results.mat' and its
%elements are as following:
%   - numSources:   number of source tasks available in the set of source tasks
%   - currMaxLen:   the maximum number of source tasks for building a
%                   curricuum
%   - TAGS_cr:      cell-array composed by the string ID of all the
%                   curricula considered in the expriment. The first
%                   element is the curriculum with the best cumulative return
%                   performance and the last is the curriculum with the
%                   worst
%   - CR:           vector whose elements are sorted accordingly to TAGS_cr
%                   reporting the relative cumulative return performance for 
%                   each curriculum in TAGS_cr


%--------------------------------------------------------------------------
%General initialization

clear all
load('results.mat')

nEvCu       = [];
bestEvCu    = [];
bestEvCuScr = [];
bestEvCuCR = [];

%--------------------------------------------------------------------------
%Generate the ID of all the possible source tasks pairs and sort them
%in crescent order. The sorthed vector of curricula IDs is the vector cps.

cps         = [];
cps_p       = [];
cps_string  = [];

for cp1 = 0:numSources-1
    for cp2 = 0:numSources-1
        if cp1 ~= cp2
            cps         = [cps; {[cp1, cp2]}];
            cps_string  = [cps_string; {cv_string(cps(end))}];
            
            cps_p = [cps_p ; find(strcmp(TAGS_cr, {cv_string(cps(end))} ))];
        end
    end
end

[cps_p_sorted, cps_p_index] = sort(cps_p);

cps = cps(cps_p_index);

%--------------------------------------------------------------------------
%All the previously generated pairs (cps) are here used for grading each
%source task in being the first element of the pair and the second. 
%The idea is that the first source task in a good graded pair is likely to
%be a good first task (or head) for a curriculum as well. The same goes for
%the second task of a pair (or tail).

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

%vectors of source tasks sorted from the best 'head' to the worst and from
%the best 'tail' to he worst
sources_h = sources(h_i).';
sources_t = sources(t_i).';

%--------------------------------------------------------------------------
%Main lopp algorithm initialization

h_fraction = 1;
t_fraction = 1;

nEvCu       = [nEvCu; size(cps,1)];
bestEvCu    = [bestEvCu; cps(1)];
bestEvCuScr = [bestEvCuScr; cps_p_sorted(1)];
bestEvCuCR = [bestEvCuCR; CR(bestEvCuScr(end))];

candidate = [];

%The first FOR incrementally selects an higher portion of the source tasks
%in sources_h and sources_t which will compose curricula for this iteration
for hyper = 1:2*(numSources - 1)
       
    if mod(hyper,2) == 1
        h_fraction = h_fraction + 1;
    else
        t_fraction = t_fraction + 1;
    end
    
    heads = sources_h(1:h_fraction);
    tails = sources_t(1:t_fraction);
    
    %The second FOR select the length of the curricula to be generated at
    %this iteration. It starts from 3 since length 1 is disregarded and
    %length 2 has already been evaluated during the pairs generation
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
                        %combine all the elements in thee left_overs vector
                        %in all the possible permutations for generating
                        %all the possible curricula body for the current h
                        %and t
                        left_overs_combinations = permutationsNK(left_overs, l-2);
                    else
                        left_overs_combinations = left_overs;
                    end
                    
                    %actually build each single possible curriculum for
                    %this iteration
                    for elem = 1:size(left_overs_combinations)
                        cv = [hd, left_overs_combinations(elem, :) , tl];
                        candidate = [candidate; {cv_string({cv})}];
                    end
                end
            end
        end
        
        %In the end of the second FOR loop, we calculate/extract thee
        %performance for the just created curricula
        
        candidate = [cps_string; candidate];

        candidate           = unique(candidate);
        candidate_scores    = [];

        for sc = 1:size(candidate)
            candidate_scores    = [candidate_scores; find(strcmp(TAGS_cr, candidate(sc)))];
        end

        [cs_sorted, cs_index] = sort(candidate_scores);

        candidate = candidate(cs_index);

        nEvCu       = [nEvCu; size(candidate,1)];
        bestEvCu    = [bestEvCu; {candidate(1)}];
        bestEvCuScr = [bestEvCuScr; cs_sorted(1)];    
        bestEvCuCR  = [bestEvCuCR; CR(bestEvCuScr(end))];
        
    end
end

%Plotting for performance analysis
plot(nEvCu,-bestEvCuScr)
