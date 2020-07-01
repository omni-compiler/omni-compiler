grep OM_EXEC_F_COMPILER= ../../../../../config.log | awk -F= '{print $2}' | xargs basename
