outpath='/p/lscratchf/ni4/w2imvec.words/vwiki/singapore/outputs/';
inpath='/p/lscratchf/ni4/w2imvec.words/vwiki/singapore/inputs/';

imfeats = [inpath 'test101'];
wordprefix = [outpath 'words101'];
vectorprefix = [outpath 'vectors101'];
Wmatname = 'W101.txt';

% load features
alexvecs = load(imfeats);

% vocabulary vectors
fid=fopen([wordprefix '.vocab.vecs'], 'rb');
wordvecs = fread(fid, 'float');
fclose(fid);

% vocabulary words
fid=fopen([wordprefix '.vocab.text'], 'rb');
words = textscan(fid, '%s'); words = words{1};
fclose(fid);

% vocabulary frequencies
fid=fopen([wordprefix '.vocab.freqs'], 'rb');
wordcounts=fread(fid,'float');
fclose(fid);

% make everything the correct size
N = size(words,1);
wordvecs=reshape(wordvecs, [200 N]);

% Load the matrix after optimization
Wmat = load([outpath Wmatname]);

% The metadata indices, read line by line
fid = fopen([wordprefix '.index.meta'], 'rb');
i = 0;
while ~feof(fid)
    i=i+1;
    metadata{i} = str2num(fgetl(fid))+1;
end
fclose(fid);

