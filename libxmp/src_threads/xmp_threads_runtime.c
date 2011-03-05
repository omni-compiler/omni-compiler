extern void ompc_init(int argc,char *argv[]);
extern void ompc_terminate (int);

void _XMP_threads_init(int argc, char *argv[]) {
  ompc_init(argc,argv);
}

void _XMP_threads_finalize(int ret) {
  ompc_terminate(ret);
}
