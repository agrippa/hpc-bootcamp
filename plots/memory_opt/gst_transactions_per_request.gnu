set terminal pdf enhanced noenhanced
set output 'gst_transactions_per_request.pdf'
set datafile separator ','

set style data histogram

set key top left

set style fill solid border rgb "black"
set auto x
set yrange [0:*]
plot 'gst_transactions_per_request' using 2:xtic(1) title col, \
         '' using 3:xtic(1) title col, \
         '' using 4:xtic(1) title col, \
         '' using 5:xtic(1) title col, \
         '' using 6:xtic(1) title col
