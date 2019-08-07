%This function simply calls the python script permutations.py for computing
%the possible permutations of the elments in v in k position.

function out = permutationsNK(v,k)
    out = [];
    
    vStr = sprintf('%d,', v);

    input = strcat(vStr(1:end-1)," ", num2str(k));

    commandStr = ''; %insert here the string for the command to run the python script permutations.py
    
    [status, commandOut] = system(commandStr);
    
    commandOut = commandOut(1:end-1);
    
    if status==0
        commandOut = erase(commandOut, {'[', ']', ''''});
        g = strsplit(commandOut,"), ");

        for i = 1:size(g,2)
            g(i) = erase(g(i), {'(', ')', });
            out = [out; str2num(cell2mat(g(i)))];
        end
    else
        e = 'ERROR'
    end
end
