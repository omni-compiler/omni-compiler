#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#define TRUE  1
#define FALSE 0
#define MAX_BUF      4096
#define MAX_NAME_LEN 256
#define TMP_DIR      "/tmp/"
#define MAX_INIT     256

char command_buf[MAX_BUF];
char buf[MAX_BUF];
char *tmp_dir;
char nm_output[MAX_NAME_LEN];
char init_func_source[MAX_NAME_LEN];
char init_func_object[MAX_NAME_LEN];
char *module_init_names[MAX_INIT];
int n_module_init = 0;
int debug_flag    = FALSE;
char *cc_command  = "cc";
char *cc_option   = "";
char *INIT_MODULE_OBJ, *NM_PREFIX, *INIT_PREFIX;
char *MODULE_INIT_NAME, *MODULE_INIT_NAME_;
char *MODULE_INIT_ENTRY_NAME;

int main(int argc, char *argv[])
{
    int i, pid, len;
    int init_name_len, init_name_len_;
    char *arg, *prog;
    FILE *fp;

    tmp_dir = TMP_DIR;
    prog    = argv[0];

    argc--;
    argv++;
    if(argc > 0 && strcmp(argv[0],"--debug") == 0){
	argc--;
	argv++;
	debug_flag = TRUE;
	tmp_dir = "";
    }

    if(argc > 0 && strcmp(argv[0],"--cc") == 0){
	argc--;
	argv++;
	cc_command = strdup(argv[0]);
	argc--;
	argv++;
    }

    if(argc > 0 && strcmp(argv[0],"--PID") == 0){
      argc--;
      argv++;
      pid = atoi(argv[0]);
      argc--;
      argv++;
    }

    if(argc > 0 && strcmp(argv[0],"--F") == 0){
      argc--;
      argv++;
      if(debug_flag) printf("Fortran mode\n");

      INIT_MODULE_OBJ   = "_xmpf_module_INIT.o";
      NM_PREFIX         = "_xmpf_nm_";
      INIT_PREFIX       = "_xmpf_init_";
      MODULE_INIT_NAME  = "_xmpf_module_init_";
      MODULE_INIT_NAME_ = "_xmpf_module_init__";
      MODULE_INIT_ENTRY_NAME = "xmpf_module_init__";
    }
    else if(argc > 0 && strcmp(argv[0],"--C") == 0){
      argc--;
      argv++;
      if(debug_flag) printf("C mode\n");

      INIT_MODULE_OBJ   = "_xmpc_module_INIT.o";
      NM_PREFIX         = "_xmpc_nm_";
      INIT_PREFIX       = "_xmpc_init_";
      MODULE_INIT_NAME  = "_xmpc_module_init_";
      MODULE_INIT_NAME_ = "_xmpc_module_init__";
      MODULE_INIT_ENTRY_NAME = "xmpc_module_init";
    }
    else{
      fprintf(stderr, "error. worong arg is used. The arg must be --F or --C.\n");
      exit(1);
    }

    if(argc > 0 && strcmp(argv[0],"--OPTION") == 0){
      argc--;
      argv++;
      cc_option = strdup(argv[0]);
      argc--;
      argv++;
    }

    pid = getpid();
    strcpy(command_buf,"nm");
    for(i = 0; i < argc; i++){
	arg = argv[i];
	len = strlen(arg);
	if(len > 2 && strcmp(&arg[len-2],".o") == 0){
	    strcat(command_buf," ");
	    strcat(command_buf,argv[i]);
	}
    }
    sprintf(nm_output,"%s%s%d",tmp_dir,NM_PREFIX,pid);
    strcat(command_buf," > ");
    strcat(command_buf,nm_output);
    if(debug_flag) printf("command = '%s'\n",command_buf);
    if(system(command_buf) < 0){
	fprintf(stderr,"error in execting '%s'\n",command_buf);
	exit(1);
    }
    fp = fopen(nm_output,"r");
    if(fp == NULL){
	fprintf(stderr,"cannot open '%s'\n",nm_output);
	exit(1);
    }
    init_name_len = strlen(MODULE_INIT_NAME);
    init_name_len_ = strlen(MODULE_INIT_NAME_);
    while(fscanf(fp,"%s",buf) == 1){
      if(strncmp(buf,".jwe",4) == 0 || 
	 strncmp(buf,"jpj.",4) == 0) continue; // for K computer

      len = strlen(buf);
      if(len > init_name_len && 
	 strcmp(buf+(len-init_name_len),MODULE_INIT_NAME) == 0){
	module_init_names[n_module_init++] = strdup(buf);
      } 
      else if(len > init_name_len_ && 
	      strcmp(buf+(len-init_name_len_),MODULE_INIT_NAME_) == 0){
	module_init_names[n_module_init++] = strdup(buf);
      }
      else{
	// for the Cray
	// In Cray machines, when module name "foo", 
	// a subroutine for "foo" is converted to "foo_xmpf_module_init_$foo_".
	int module_name_len = (len - init_name_len - 2) / 2;
	if(len > init_name_len &&
	   strncmp(buf+module_name_len,MODULE_INIT_NAME,init_name_len) == 0){
	  module_init_names[n_module_init++] = strdup(buf);
	}
      }
    }
    fclose(fp);
    if(!debug_flag) unlink(nm_output);

    sprintf(init_func_source,"%s%s%d.c",tmp_dir,INIT_PREFIX,pid);
    //    sprintf(init_func_object,"%s%s%d.o",tmp_dir,INIT_PREFIX,pid);
    strcpy(init_func_object,INIT_MODULE_OBJ);
    fp = fopen(init_func_source,"w");
    if(fp == NULL){
	fprintf(stderr,"cannot open '%s'\n",init_func_source);
	exit(1);
    }
    for(i=0; i<n_module_init;i++){
      char *name = module_init_names[i];
      fprintf(fp,"extern void %s();\n",name);
    }
    fprintf(fp,"\n");
    fprintf(fp,"void %s(){\n",MODULE_INIT_ENTRY_NAME);

    for(i=0; i<n_module_init;i++){
      char *name = module_init_names[i];
      if(strchr(name,'.') != NULL)
	fprintf(fp,"asm(\"call\t%s\");\n",name);  // asm("call func"); 
      else
	fprintf(fp,"\t%s();\n",name);
	
    }
    fprintf(fp,"}\n");
    fclose(fp);
    sprintf(command_buf,"%s -c -o %s %s %s",
	    cc_command, init_func_object, init_func_source, cc_option);

    if(debug_flag) printf("command = '%s'\n",command_buf);
    if(system(command_buf) < 0){
	fprintf(stderr,"error in execting '%s'\n",command_buf);
	exit(1);
    }
    
    if(!debug_flag) unlink(init_func_source);
    
    exit(0);
}
