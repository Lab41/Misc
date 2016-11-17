% checkval = 2;

% Display associated metadata
words( metadata{checkval} )

% Create normalized word vectors
Vw = wordvecs ./ sqrt( repmat( sum(wordvecs.^2), [200, 1] ) );

% Normalize hash vector (unnecessary if word vectors are normalized)
vq = Vh(:,checkval);

% Compute dot product and then sort all results
[dval idx] = sort(vq'*Vw, 'descend');

% Display most relevant words
words( idx(1:30) )
