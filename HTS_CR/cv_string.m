%This function is meant to convert a curriculum ID into its string form so
%to be able to query the dataset. 
%
%A curriculum ID in this context is a vector within a cell-array. Each ith
%element of the vector is an ID corresponding to the source task at ith
%position of the curriculum.
%
%Example -> INPUT: {[0, 1, 7]} ; OUTPUT: '[0, 1, 7]'

function output = cv_string(ids)
    output = '[';
    
    for i = 1:length(ids{1})
        s = ids{1};%select the actual vector of sources ids
        
        if ~isnumeric(s)
            s = s{i};
        end
        
        if length(s)~=1
            s = s(i);
        end    
        
        output = [output mat2str(s)];
        
        if i ~= length(ids{1})
            output = [output ',' blanks(1)];
        end    
    end    
    
    output = [output ']'];
    
end
