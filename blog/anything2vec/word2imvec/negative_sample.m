function index = negative_sample(wordcdfmap, maxval, negsamp)

  randindx = ceil(rand(negsamp,1)*maxval);
  index = wordcdfmap(randindx);
