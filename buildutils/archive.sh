#!/bin/sh

# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
if [ ! -e configure.ac ]; then
	pwd
	echo execute this script in top directory
	exit 1
fi

if [ -e "./config.log" ]; then
	echo execute this script before configure
	exit 1
fi

if [ "$*" = "" ]; then
	echo specify argument 'c' or 'f' or 'all' or 'release'
	exit 1
fi

for arg in $*;
do
	case $arg in
	c)
		ENABLE_C=1;;
	f)
		ENABLE_F=1;;
	all)
		ENABLE_C=1
		ENABLE_F=1;;
        release)
		ENABLE_C=1
		ENABLE_F=1
                ENABLE_RELEASE=1;;
	esac
done

if [ "$ENABLE_C" = "1" -a "$CONFIGURE_OPT" = "" ]; then
    echo set environment CONFIGURE_OPT as configure options
    exit 1
fi

if [ "$ENABLE_C" != 1 -a "$ENABLE_F" != 1 ]; then
	echo specify argument 'c' or 'f' or 'all'
	exit 1
fi

if [ "$ENABLE_RELEASE" = 1 ]; then
        if [ ! -e "release/releasename.txt" ]; then
             echo need release/releasename.txt
             exit 1
        else
             RELEASE="`line < release/releasename.txt`"
        fi

        if [ ! -e "release/copyright.txt" ]; then
             echo need release/copyright.txt
             exit 1
        fi
fi

PACKAGE_NAME=omni-3
WORK_DIR=./dist
PACKAGE_DIR=$WORK_DIR/$PACKAGE_NAME
DOC_DIR=$PACKAGE_DIR/doc

echo autogen.sh
./autogen.sh

echo delete dist dir
rm -rf $WORK_DIR
mkdir $WORK_DIR
echo copy whole files
cp -r . $PACKAGE_DIR
rmdir $PACKAGE_DIR/dist

if [ "$ENABLE_C" != 1 ]; then
	echo delete c2c project files \(exlucde C-BackEnd\)
	rm -rf $PACKAGE_DIR/C-FrontEnd
	rm -rf $PACKAGE_DIR/libxmp
fi

if [ "$ENABLE_F" != 1 ]; then
	echo delete f2f project files
	rm -rf $PACKAGE_DIR/F-FrontEnd
	rm -rf $PACKAGE_DIR/F-BackEnd
fi

echo delete .svn dirs
find $PACKAGE_DIR -name \.svn -exec rm -rf \{\} \; 2> /dev/null

if [ "$ENABLE_RELEASE" = "1" ]; then
    echo embed header.
    ./release/addrcsheader -ignore release/ignore.txt $PACKAGE_DIR
    echo expand copyright, release, rcsId.
    ./release/expand_tsukuba_keyword \
        -release $RELEASE \
        -copyright release/copyright.txt \
        -ignore release/ignore.txt  $PACKAGE_DIR
fi

echo configure
./configure $CONFIGURE_OPT > /dev/null
mkdir -p $DOC_DIR

# common project documents
echo build documents : C-BackEnd
cd C-BackEnd
make html > /dev/null
cd ..
mv C-BackEnd/docs $DOC_DIR/C-BackEnd

mkdir -p $DOC_DIR
echo build documents : Driver
cd Driver/src
make html > /dev/null
cd ../..
mv Driver/doc $DOC_DIR/Driver


# c2c project documents
if [ "$ENABLE_C" = 1 ]; then
	echo build documents : C-FrontEnd
	cd C-FrontEnd/src
	make html > /dev/null
	cd ../..
	mv C-FrontEnd/doc $DOC_DIR/C-FrontEnd

	mkdir -p $DOC_DIR
	echo build documents : libxmp
	cd libxmp/src
	make html > /dev/null
	cd ../..
	mv libxmp/doc $DOC_DIR/libxmp
fi

# f2f project documents
if [ "$ENABLE_F" = 1 ]; then
	echo build documents : F-FrontEnd
	cd F-FrontEnd/src
	make html > /dev/null
	cd ../..
	mv F-FrontEnd/doc $DOC_DIR/F-FrontEnd

	echo build documents : F-BackEnd
	cd F-BackEnd
	make html > /dev/null
	cd ..
	mv F-BackEnd/docs $DOC_DIR/F-BackEnd
fi

echo touch Makefile.in
find $PACKAGE_DIR -name Makefile.in -exec touch \{\} \;
echo touch configure
touch $PACKAGE_DIR/configure

echo archiving
cd $WORK_DIR
tar czf $PACKAGE_NAME.tar.gz $PACKAGE_NAME

echo completed

