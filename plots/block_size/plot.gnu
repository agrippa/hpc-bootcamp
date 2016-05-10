set terminal pdf enhanced
set output 'naive_vecadd.pdf'
set datafile separator ','

# set logscale x
set key top left
set xlabel 'Block Size'
set ylabel 'Execution Time (microseconds)'
set xtics rotate

plot 'data' using 1:2 with lines title col, \
         '' using 1:3 with lines title col
