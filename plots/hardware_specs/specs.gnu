set terminal pdf enhanced size 5,4
set output 'specs.pdf'
set multiplot
set size 1,0.3

set style data histogram
set style histogram gap 1
set boxwidth 0.8
set datafile separator ','

set ylabel 'Memory Bandwidth (GB/s)'
set key right top
set style fill solid border rgb "black"
set auto x
set yrange [0:*]
set ytics out
unset xtics

set origin 0.0,0.65
plot 'bandwidth.dat' using 2 title col, \
         '' using 3 title col, \
         '' using 4 title col, \
         '' using 5 title col, \
         '' using 6 title col

unset key
set ylabel '64-bit GFLOPS'
set origin 0.0,0.30
set yrange [0:9000]
set ytics 0,2000,9000
plot 'tflops.dat' using 2 title col, \
         '' using 3 title col, \
         '' using 4 title col, \
         '' using 5 title col, \
         '' using 6 title col

unset key
set ylabel 'Power (Watts)'
set origin 0.0,0.0
set yrange [0:320]
plot 'power.dat' using 2 title col, \
         '' using 3 title col, \
         '' using 4 title col, \
         '' using 5 title col, \
         '' using 6 title col
