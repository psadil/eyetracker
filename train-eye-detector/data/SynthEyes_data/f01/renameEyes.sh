find -name '*.png' | # find jpegs
gawk 'BEGIN{ a=1 }{ printf "mv %s %04d.png\n", $0, a++ }' | # build mv command
bash # run that command