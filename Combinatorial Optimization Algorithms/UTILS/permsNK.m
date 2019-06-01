function out = permsNK(v,k)
    out = [];
    
    vStr = sprintf('%d,', v);

    input = strcat(vStr(1:end-1)," ", num2str(k));

    commandStr = strcat('C:/Users/fogli/Anaconda3/python.exe C:/Users/fogli/Documents/Matlab/"Regret Minimization"/permutations.py', " ", input);

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