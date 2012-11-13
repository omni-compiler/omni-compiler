#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define TRUE 1
#define FALSE 0

#define MAX_BUF 4096
#define INIT_MODULE_OBJ "_xmpf_module_INIT.o"

char command_buf[MAX_BUF];
char buf[MAX_BUF];

#define MAX_NAME_LEN 256
#define TMP_DIR "/tmp/"
#define NM_PREFIX "_xmpf_nm_"
#define INIT_PREFIX "_xmpf_init_"

char *tmp_dir;
char nm_output[MAX_NAME_LEN];
char init_func_source[MAX_NAME_LEN];
char init_func_object[MAX_NAME_LEN];

#define MODULE_INIT_F_NAME "_xmpf_module_init_"
#define MODULE_INIT_F_NAME_ "_xmpf_module_init__"

#define MODULE_INIT_ENTRY_NAME "xmpf_module_init__"
#define MAX_INIT_F 256

char *module_init_f_names[MAX_INIT_F];
int n_module_init_f = 0;

int debug_flag = FALSE;

int main(int argc, char *argv[])
{
    int i, pid, len;
    int initf_name_len;
    int initf_name_len_;
    char *arg;
    FILE *fp;
    char *prog;

    tmp_dir = TMP_DIR;
    prog = argv[0];

    argc--;
    argv++;
    if(argc > 0 && strcmp(argv[0],"--debug") == 0){
	argc--;
	argv++;
	debug_flag = TRUE;
	tmp_dir = "";
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
    initf_name_len = strlen(MODULE_INIT_F_NAME);
    initf_name_len_ = strlen(MODULE_INIT_F_NAME_);
    while(fscanf(fp,"%s",buf) == 1){
	len = strlen(buf);
	if(len > initf_name_len && 
	   strcmp(buf+(len-initf_name_len),MODULE_INIT_F_NAME) == 0){
	    module_init_f_names[n_module_init_f++] = strdup(buf);
	} else 
	if(len > initf_name_len_ && 
	   strcmp(buf+(len-initf_name_len_),MODULE_INIT_F_NAME_) == 0){
	    module_init_f_names[n_module_init_f++] = strdup(buf);
	}
    }
    fclose(fp);
    if(!debug_flag) unlink(nm_output);

    sprintf(init_func_source,"%s%s%d.c",tmp_dir,INIT_PREFIX,pid);
    // sprintf(init_func_object,"%s%s%d.o",tmp_dir,INIT_PREFIX,pid);
    strcpy(init_func_object,INIT_MODULE_OBJ);
    fp = fopen(init_func_source,"w");
    if(fp == NULL){
	fprintf(stderr,"cannot open '%s'\n",init_func_source);
	exit(1);
    }
    fprintf(fp,"%s(){\n",MODULE_INIT_ENTRY_NAME);
    for(i=0; i<n_module_init_f;i++)
	fprintf(fp,"\t%s();\n",module_init_f_names[i++]);
    fprintf(fp,"}\n");
    fclose(fp);
    sprintf(command_buf,"cc -c -o %s %s",
	    init_func_object, init_func_source);

    if(debug_flag) printf("command = '%s'\n",command_buf);
    if(system(command_buf) < 0){
	fprintf(stderr,"error in execting '%s'\n",command_buf);
	exit(1);
    }
    
    if(!debug_flag) unlink(init_func_source);
    
    exit(0);
}
