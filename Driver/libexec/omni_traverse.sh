#!/bin/bash
#
# V3
#

#VERBOSE=yes
MODE=1         # generate subroutines respectively for all groups
#MODE=2          # generate one subroutine for all proceures of all groups

USAGE='usage: '$0' <option> ... <input_file> ...
  <option>
    --help               this help
    --verbose            verbose mode
    --F | --C            which language (necessary)
    --nm                 nm command
    --sr                 Option for HITACHI SR series
    --prefix <prefix>    prefix characters of the target procecdure names, such as xmpf and xmpc
    -o <output_file>     name of the output file (*.f90 or *.c file is expected)
  <input_file>           input file name (.o and .a files are expected)
'

#--------------------------------------------------------------
# option analysis
#--------------------------------------------------------------
INFILES=()
USE_SR=no
while [ -n "$1" ]; do
    case "$1" in
    --help)     HELP=yes;;
    --verbose)  VERBOSE=yes;;
    --F)        LANG=F;;
    --C)        LANG=C;;
    --prefix)   shift; PREFIX="$1";;
    --nm)       shift; NM="$1";;
    --nm_opt)   shift; NM_OPT="$1";;
    --sr)       USE_SR=yes;;
    -o)         shift; OUTFILE="$1";;
    *.o)        INFILES+=("$1");;
    *.a)        INFILES+=("$1");;
    *)          INFILES+=("$1"); echo "$0: Warning: illegal filename $1" 1>&2;;
    esac
    shift
done

#--------------------------------------------------------------
# set PREFIX for mangling 
#--------------------------------------------------------------
if [ $USE_SR = no ]; then
    M_PREFIX="$PREFIX"
else
    M_PREFIX=."$PREFIX"
fi

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
   echo INFILES=${INFILES[@]}
   echo OUTFILE=$OUTFILE
   echo ---------------------------
fi


#--------------------------------------------------------------
# find names ${PREFIX}_traverse_* 
#--------------------------------------------------------------

#--- get traverse procedures
trav_cands=`$NM $NM_OPT "${INFILES[@]}" | \
    awk 'NF >= 2 && $(NF-1) ~ "^[U]$" { print $NF }'`
traversers=()

if test "$LANG" = "C"; then
    traversers+=(xmpc_traverse_init)
    traversers+=(xmpc_traverse_finalize)
fi
for name in $trav_cands; do
    name=${name#_}                # re-mangling for MacOS
    case $name in
        ${M_PREFIX}_traverse_?*)
        case "$LANG" in 
            F) traversers+=( ${name%_} );;    # omit the last '_'
            C) traversers+=( ${name#.} );;    # omit the first '.' on the HITACHI SR
        esac;;
    esac
done

#--- get procedures to be traversed
proc_cands=`$NM $NM_OPT "${INFILES[@]}" | \
    awk 'NF >= 2 && $(NF-1) ~ "^[DdTt]$" { print $NF }'`
procedures=()
for name in $proc_cands; do
    name=${name#_}                # re-mangling for MacOS
    case $name in
        ${M_PREFIX}_traverse_*_?* )
            case "$LANG" in
                F) procedures+=( ${name%_} );;    # omit the last '_'
                C) procedures+=( ${name#.} );;    # omit the first '.' on the HITACHI SR
            esac;;
        ${M_PREFIX}_traverse_* )
            echo found unacceptable name of traverse procedure: \"${name}\"
            error=yes;;
    esac
done

if [ "${error}" = "yes" ]; then
    exit 1
fi

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
    echo "[candidate names for tarverse procedures]"
    echo "  "$trav_cands
    echo "[generated traverse procedures]"
    echo "  ${traversers[@]}"
    echo "[group names]"
    echo "  ${groupnames[@]}"
    echo "[candidate names for called procedures]"
    echo "  "$proc_cands
    echo "[procedures called in traverse procedures]"
    echo "  ${procedures[@]}"
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
        echo "extern void ${groupname}_(void);"
        for name in "${procedures[@]}"; do
            if [[ ${name} =~ ${groupname}_* ]]; then
                echo "extern void ${name}(void);"
            fi
        done
    done
    echo

    for groupname in "${groupnames[@]}"; do
        echo "void ${groupname}_() {"
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

file_output() {
    case $MODE in
    1)  case "$LANG" in
        F) fortran_output_MODE1;;
        C) c_output_MODE1;;
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
