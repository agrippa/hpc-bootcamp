set terminal pdf enhanced
set output 'radius.pdf'
set datafile separator ','

# set logscale x
set key top left
set xlabel 'Radius'
set ylabel 'Performance (Mcells/s/step)'
set xtics rotate

plot 'data' using 1:2 with lines title col, \
         '' using 1:3 with lines title col
