#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

#define USE_ABT 1

typedef void (*cfunc)(void**);

void tasklet_initialize(int argc, char *argv[]);
void tasklet_exec_main(cfunc f);
void tasklet_finalize();

void tasklet_create(cfunc f, int narg, void **args, 
                    int n_in, void **in_data, int n_out, void **out_data);

void tasklet_wait_all();

