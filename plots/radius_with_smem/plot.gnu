set terminal pdf enhanced
set output 'gld_transactions.pdf'
set datafile separator ','

# set logscale x
set key top left
set xlabel 'Radius'
set ylabel '# Global Load Transactions'
set xtics rotate

plot 'gld_transactions' using 1:2 with lines title col, \
         '' using 1:3 with lines title col
