#!/bin/bash
#
# V3.2
#

#VERBOSE=yes
NM=nm
MODE=1         # generate subroutines respectively for all groups
#MODE=2          # generate one subroutine for all proceures of all groups

USAGE='usage: '$0' <option> ... <input_file> ...
  <option>
    --help               this help
    --verbose            verbose mode
    --F | --C            which language (necessary)
    --nm                 nm command
    --prefix <prefix>    prefix characters of the target procecdure names, such as xmpf and xmpc
    --platform <OpenCL|CUDA> platform for OpenACC
    -o <output_file>     name of the output file (*.f90 or *.c file is expected)
  <input_file>           input file name (.o and .a files are expected)
'

#--------------------------------------------------------------
# option analysis
#--------------------------------------------------------------
INFILES=()
PLATFORM="CUDA" # default is CUDA
while [ -n "$1" ]; do
    case "$1" in
    --help)     HELP=yes;;
    --verbose)  VERBOSE=yes;;
    --F)        LANG=F;;
    --C)        LANG=C;;
    --prefix)   shift; PREFIX="$1";;
    --nm)       shift; NM="$1";;
    --nm_opt)   shift; NM_OPT="$1";;
    --platform) shift; PLATFORM="$1";;
    -o)         shift; OUTFILE="$1";;
    *.o)        INFILES+=("$1");;
    *.a)        INFILES+=("$1");;
    *)          INFILES+=("$1"); echo "$0: Warning: illegal filename $1" 1>&2;;
    esac
    shift
done

if [ "$HELP" = "yes" -o "$LANG" = "" ]; then
   echo "$USAGE" 1>&2
   exit 1
fi

if [ "$VERBOSE" = "yes" ]; then
   echo ---------------------------
   echo NM=$NM $NM_OPT
   echo HELP=$HELP
   echo VERBOSE=$VERBOSE
   echo LANG=$LANG
   echo PREFIX=$PREFIX
   echo PLATFORM=$PLATFORM
   echo INFILES=${INFILES[@]}
   echo OUTFILE=$OUTFILE
   echo ---------------------------
fi

#--------------------------------------------------------------
# find symbol names in the input files:
#  1) defNames: symbols defined in the files
#  2) usedNames: symbols used in the files
#  3) undefNames: symbols used but not defined in the files
#--------------------------------------------------------------

#-- 1) get defNames
defNames=`$NM $NM_OPT "${INFILES[@]}" | \
        awk 'NF >= 2 && $(NF-1) ~ "^[DdTt]$" { print $NF }'`

#-- 2) get usedNames
usedNames=`$NM $NM_OPT "${INFILES[@]}" | \
        awk 'NF >= 2 && $(NF-1) ~ "^[U]$" { print $NF }'`

#-- 3) get undefNames
undefNames=""
for uname in $usedNames; do
    match="no"
    for dname in $defNames; do
	if [ $uname == $dname ]; then
	    match="yes"
	    break
	fi
    done
    if [ $match == "no" ]; then
	undefNames+=" $uname"
    fi
done

if [ "$VERBOSE" = "yes" ]; then
    echo
    echo "[symbols defined in the input files]"
    echo $defNames
    echo "[symbols used in the input files]"
    echo $usedNames
    echo "[symbols used but not defined in the input files]"
    echo $undefNames
fi

#--------------------------------------------------------------
# get names of the traversers and target procedures
#  1) traversers: will be defined by this tool to call all target
#     procedures, and must be called from the user program.
#  2) target procedures: are previously defined by the user, and 
#     will be called from the traverser.
#--------------------------------------------------------------

#-- 1) get traversers
traversers=()
for name in $undefNames; do
    name=${name#_}                # re-mangling for MacOS
    case $name in
        ${PREFIX}_traverse_?*)
        case "$LANG" in 
            F) traversers+=( ${name%_} );;    # omit the last '_'
            C) traversers+=( ${name#.} );;    # omit the first '.' on the HITACHI SR
        esac;;
    esac
done

#-- 2) get target procedures
procedures=()
for name in $defNames; do
    name=${name#_}                # re-mangling for MacOS
    case $name in
        ${PREFIX}_traverse_*_?* )
            case "$LANG" in
                F) procedures+=( ${name%_} );;    # omit the last '_'
                C) procedures+=( ${name#.} );;    # omit the first '.' on the HITACHI SR
            esac;;
        ${PREFIX}_traverse_* )
            echo found unacceptable name of traverse procedure: \"${name}\"
            error=yes;;
    esac
done
if [ "${error}" = "yes" ]; then
    exit 1
fi

if [ "$VERBOSE" = "yes" ]; then
    echo
    echo "[traversers]"
    echo "  ${traversers[@]}"
    echo "[target procedures]"
    echo "  ${procedures[@]}"
fi


#--------------------------------------------------------------
# action
#--------------------------------------------------------------

#--- classify the chosen names by keywords
if [ $MODE == 1 ]; then
    groupnames=( ${traversers[@]} )
else
  groupnames=()
  for name in "${procedures[@]}"; do
    # separate procedure name "prefix_travserse_keywd_reststring"
    # into three parts, groupname, '_' and basename, where
    #  groupname is "prefix_travserse_keywd" and
    #  basename is any string possibly including '_'
    basename=${name#${PREFIX}_*_*_}
    groupname=${name%_${basename}}

    # append part1 to keystrings if not appended yet
    case "${groupnames[*]}" in
        $groupname)  ;;
        *)           groupnames+=( $groupname );;
    esac
  done
fi

if [ "$VERBOSE" = "yes" ]; then
    echo
    echo "[group names (=traversers if mode==1)]"
    echo "  ${groupnames[@]}"
    echo
fi

#--------------------------------------------------------------
# file output
#--------------------------------------------------------------

fortran_output_MODE1() {
    for groupname in "${groupnames[@]}"; do
        echo "subroutine ${groupname}"
        for name in "${procedures[@]}"; do
            if [[ ${name} =~ ${groupname}_* ]]; then
                echo "  call ${name}"
            fi
        done
        echo "end subroutine"
        echo
    done
}

fortran_output_MODE2() {
    echo "subroutine ${PREFIX}_traverse"
        for name in "${procedures[@]}"; do
            echo "  call ${name}"
        done
    echo "end subroutine"
}

c_output_MODE1() {
    for groupname in "${groupnames[@]}"; do
        echo "extern void ${groupname}(void);"
        for name in "${procedures[@]}"; do
            if [[ ${name} =~ ${groupname}_* ]]; then
                echo "extern void ${name}(void);"
            fi
        done
    done
    echo

    for groupname in "${groupnames[@]}"; do
        echo "void ${groupname}() {"
        for name in "${procedures[@]}"; do
            if [[ ${name} =~ ${groupname}_* ]]; then
                echo "  ${name}();"
            fi
        done
        echo "}"
        echo
    done
}

c_output_MODE2() {
    for groupname in "${groupnames[@]}"; do
        echo "extern void ${groupname}(void);"
    done
    echo

    echo "void ${PREFIX}_traverse()"
    echo "{"
        for name in "${procedures[@]}"; do
            echo "  ${name}();"
        done
    echo "}"
}

c_output_MODE_ACC() {
    echo "#include <string.h> // ACC mode ..."
    for groupname in "${groupnames[@]}"; do
        echo "extern void ${groupname}(void);"
        for name in "${procedures[@]}"; do
            if [[ ${name} =~ ${groupname}_* ]]; then
                echo "extern void ${name}(void);"
            fi
        done
    done
    echo

    for groupname in "${groupnames[@]}"; do
	if [ "$groupname" = "acc_traverse_init" ]; then
            for name in "${procedures[@]}"; do
		if [[ ${name} =~ ${groupname}_* ]]; then
                    FNAME=${name:23};
		    echo "char *_binary___omni_tmp___"$FNAME"_cl_start;"
		    echo "char *_binary___omni_tmp___"$FNAME"_cl_end;"
		    echo "extern char _cl_prog_"$FNAME"[];"
		fi
            done
	fi
    done
    echo
    
    for groupname in "${groupnames[@]}"; do
#	echo "groupname=" ${groupname}
        echo "void ${groupname}() {"
	if [ "$groupname" = "acc_traverse_init" ]; then
           for name in "${procedures[@]}"; do
               if [[ ${name} =~ ${groupname}_* ]]; then
                   FNAME=${name:23};
		   echo "_binary___omni_tmp___"$FNAME"_cl_start=_cl_prog_"$FNAME";"
		   echo "_binary___omni_tmp___"$FNAME"_cl_end=_cl_prog_"$FNAME"+strlen(_cl_prog_"$FNAME");"
               fi
           done
	fi
        for name in "${procedures[@]}"; do
            if [[ ${name} =~ ${groupname}_* ]]; then
                echo "  ${name}();"
            fi
        done
        echo "}"
        echo
    done
}

file_output() {
    case $MODE in
    1)  case "$LANG" in
        F) fortran_output_MODE1;;
        C) if [ "$PREFIX" = "acc" ] && [ "$PLATFORM" = "OpenCL" ]; then
	       c_output_MODE_ACC
	   else
	       c_output_MODE1
	   fi;;
        esac;;
    2)  case "$LANG" in
        F) fortran_output_MODE2;;
        C) c_output_MODE2;;
        esac;;
    esac
}    

if [ "$VERBOSE" = "yes" ]; then
    echo ---------------------------------------------- output $OUTFILE
    file_output
    echo ------------------------------------------ end output $OUTFILE
fi

if [ -z "$OUTFILE" ]; then
    file_output
else
    file_output > $OUTFILE
fi

exit 0
