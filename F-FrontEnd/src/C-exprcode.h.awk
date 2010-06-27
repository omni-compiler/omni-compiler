# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
BEGIN{
        printf("/* generated from C-exprcode.def */\n");
        printf("#ifndef _C_EXPRCODE_H_\n#define _C_EXPRCODE_H_\n");
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
        printf("\t%s = %d,\n", $2,i++);
}

END {
        printf("\tEXPR_CODE_END\n};\n");
        printf("/* END */\n");
        printf("#define MAX_CODE %d\n", i);
        printf("#endif /* _C_EXPRCODE_H_ */\n");
}

