#!/bin/sh
#
ndx=$2
ndy=$3
ndz=$4
#
case "$1" in
    ssmall | XS )
       mx0=33
       my0=33
       mz0=65 ;;
    small | S )
       mx0=65
       my0=65
       mz0=129 ;;
    midium | M )
       mx0=129
       my0=129
       mz0=257 ;;
    large| L )
       mx0=257
       my0=257
       mz0=513 ;;
    elarge| XL )
       mx0=513
       my0=513
       mz0=1025 ;;
    * )
       echo ' Invalid argument'
       echo ' Usage:: % program <Grid size> <ID> <JD> <KD>'
       echo '         Grid size= XS (32x32x64)'
       echo '                    S  (64x64x128)'
       echo '                    M  (128x128x256)'
       echo '                    L  (256x256x512)'
       echo '                    XL (512x512x1024)'
       echo ' '
       echo ' <ID> <JD> <KD> is partition size'
       echo '        <ID> is the number of partition for I-dimensional'
       echo '        <JD> is the number of partition for J-dimensional'
       echo '        <KD> is the number of partition for K-dimensional'
       echo ' '
       echo ' The number of PE is fixed partition size'
       echo '        Number of PE= <ID> x <JD> x <KD>'
       exit ;;
esac
#
if [ -f param.h ]
then
  rm param.h
fi
#
echo '/*' >> param.h
echo ' *' >> param.h
echo ' */' >> param.h
echo '#define MX0     '$mx0 >> param.h
echo '#define MY0     '$my0 >> param.h
echo '#define MZ0     '$mz0 >> param.h
#
if [ $ndx -eq 1 ]
then
    itmp=$mx0
elif [ $ndx -ne 1 ]
then
    iib=`expr $mx0 / $ndx`
    itmp=`expr $iib + 3`
fi
#
if [ $ndy -eq 1 ]
then
    jtmp=$my0
elif [ $ndy -ne 1 ]
then
    iib=`expr $my0 / $ndy`
    jtmp=`expr $iib + 3`
fi
#
if [ $ndz -eq 1 ]
then
    ktmp=$mz0
elif [ $ndz -ne 1 ]
then
    iib=`expr $mz0 / $ndz`
    ktmp=`expr $iib + 3`
fi

echo '#define MIMAX     '$itmp >> param.h
echo '#define MJMAX     '$jtmp >> param.h
echo '#define MKMAX     '$ktmp >> param.h
echo '#define NDX0      '$ndx >> param.h
echo '#define NDY0      '$ndy >> param.h
echo '#define NDZ0      '$ndz >> param.h
unset mx0
unset my0
unset mz0
unset itmp
unset jtmp
unset ktmp
unset nxd
unset nyd
unset nzd
unset iib
