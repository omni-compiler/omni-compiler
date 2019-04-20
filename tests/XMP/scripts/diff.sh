NEW_XCC=/home/mnakao/work/xmp-trunk/bin/xmpcc
OLD_XCC=~/work/xmp-trunk.old/bin/xmpcc

#__omni_tmp__117.c

for file in `cat list`
do
    BASENAME=`basename $file`
    TMP_NAME="__omni_tmp__"$BASENAME
    $NEW_XCC -c --tmp $file
    mv $TMP_NAME /tmp/
    $OLD_XCC -c --tmp $file
    diff $TMP_NAME /tmp/$TMP_NAME
    rm -f $TMP_NAME /tmp/$TMP_NAME *.o
done
