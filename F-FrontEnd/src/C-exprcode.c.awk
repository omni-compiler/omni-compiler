# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
BEGIN{
        printf("#include \"F-front.h\"\n");
        printf("/* generated from C-exprcode.def */\n");
        printf("struct expr_code_info expr_code_info[] = {\n");
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
        if(NF == 2) 
                printf("/* %d */\t{\t'%s',\t\"%s\",\tNULL},\n",i++,$1,$2);
        else if(NF == 3) 
                printf("/* %d */\t{\t'%s',\t\"%s\",\t\"%s\"},\n",i++,$1,$2,$3);
}

END {
        printf("};\n");
        printf("/* END */\n");
}

