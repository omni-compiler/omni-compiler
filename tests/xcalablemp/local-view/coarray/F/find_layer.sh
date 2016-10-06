awk '
  $1 ~ /GASNET=.0./        { print "GASNET" }
  $1 ~ /FJRDMA=.0./        { print "FJRDMA" }
  $1 ~ /MPI3_ONESIDED=.0./ { print "MPI3" }
' ../../../../../config.log
