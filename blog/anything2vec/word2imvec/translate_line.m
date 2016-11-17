function translation = translate_line( vocabulary, the_line )

   for i = 1:length(the_line)
       % disp( vocabulary{the_line(i)} );
       translation{i} = vocabulary{the_line(i)};
   end
   
  