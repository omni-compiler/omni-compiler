awk '
  $1 ~ /OM_EXEC_F_COMPILER=.mpifrtpx./  { print "mpifrtpx" }                               
  $1 ~ /OM_EXEC_F_COMPILER=.mpif90./    { print "mpif90" }                                 
' ../../../../../config.log
