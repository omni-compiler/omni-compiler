/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
extern int ompc_main(int argc, char *argv[]);
extern void ompc_init(int argc, char *argv[]);
extern void ompc_terminate(int);

int
main(int argc,char *argv[])
{
    ompc_init(argc,argv);
    ompc_terminate(ompc_main(argc,argv));
    return 0;
}

