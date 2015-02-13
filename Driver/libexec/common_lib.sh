function xmp_error_exit()
{
    echo "$0: error: $1"
    echo "compilation terminated."
    exit 1
}

function xmp_print_version()
{
    VERSION_FILE="${OMNI_HOME}/etc/version"
    if [ -f $VERSION_FILE ]; then
	cat $VERSION_FILE
	echo ""
    else
	xmp_error_exit "$VERSION_FILE not exist."
    fi
}

function xmp_exec()
{
    if [ $VERBOSE = true ] || [ $DRY_RUN = true ]; then
	echo $@
    fi

    if [ $DRY_RUN = false ]; then
	eval $@
    fi

    [ $? -ne 0 ] && { xmp_exec rm -rf $TEMP_DIR; exit 1; }
}

