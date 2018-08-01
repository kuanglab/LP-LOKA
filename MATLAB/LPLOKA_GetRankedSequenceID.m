function SeqIds = LPLOKA_GetRankedSequenceID(Fhat, H)
    % Fhat: (m x n) is the output from LPLOKA
    % H: (m x 1) m sequence IDs
    % SeqIds: (m x n) is the list of sequences, sorted in the same order as
    % in Fhat

    SeqIds = cell(size(Fhat));
    for i=1:size(Fhat,2)
        [~,idx] = sort(Fhat(:,i),'descend');
        SeqIds(:,i) = H(idx);
    end

end
