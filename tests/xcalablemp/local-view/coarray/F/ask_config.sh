arg1=$1
awk -F '=' '$1=="'$arg1'"  { print $2 }' ../../../../../config.log
