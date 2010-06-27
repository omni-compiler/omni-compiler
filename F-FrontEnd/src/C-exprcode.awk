# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
BEGIN{
        printf("/* generated from C-exprcode.def */\n");
        printf("enum expr_code {\n");
        i = 0;
}

{
        if(NF == 0){
                next;
        }
        ## skip comment 
        if($1 == "#"){
                next;
        }
        # generate entry
        printf("\t%s = %d,\n",$1,i++);
}

END {
        printf("};\n");
        printf("/* END */\n");
}

