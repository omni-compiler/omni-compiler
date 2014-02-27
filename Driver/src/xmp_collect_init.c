#ifdef OMNI_OS_LINUX
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* ! _GNU_SOURCE */
#endif /* OMNI_OS_LINUX */
#include <inttypes.h>
#include <stdint.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/param.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <alloca.h>

#define TRUE  1
#define FALSE 0
#define MAX_BUF      4096
#define MAX_NAME_LEN PATH_MAX
#define TMP_DIR      "/tmp/"
#define MAX_INIT     256

char *command_buf;
char buf[MAX_BUF];
char *tmp_dir;
char init_func_source[MAX_NAME_LEN];
char init_func_object[MAX_NAME_LEN];
char *module_init_names[MAX_INIT];
int n_module_init = 0;
int debug_flag    = FALSE;
char *cc_command  = "cc";
char *cc_option   = "";
char *INIT_MODULE_OBJ, *INIT_PREFIX;
char *MODULE_INIT_NAME, *MODULE_INIT_NAME_;
char *MODULE_INIT_ENTRY_NAME;
int pid = -INT_MAX;
int is_Mac = FALSE;
int is_K_FC = FALSE; // only Fortran compiler on the K
int is_AIX = FALSE;

#define IS_VALID_STRING(s)	\
    (((s) != NULL && *(s) != '\0') ? TRUE : FALSE)

int main(int argc, char *argv[])
{
    int i, len;
    int init_name_len, init_name_len_;
    char *arg;
    FILE *fp;
    char **files = (char **)alloca(sizeof(char *) * argc);
    size_t n_files = 0;
    size_t n_files_strlen = 0;

    tmp_dir = TMP_DIR;

    argv++;
    while (*argv != NULL) {
        if (strcmp(*argv, "--debug") == 0) {
            debug_flag = TRUE;
            tmp_dir = "";
        } else if (strcmp(*argv, "--cc") == 0) {
            if (IS_VALID_STRING(argv + 1) == TRUE) {
                argv++;
                cc_command = strdup(*argv);
            } else {
                fprintf(stderr, "error: C compiler is not specfied.\n");
                return 1;
            }
        } else if (strcmp(*argv, "--PID") == 0) {
            if (IS_VALID_STRING(argv + 1) == TRUE) {
                argv++;
                if (IS_VALID_STRING(init_func_object) == TRUE) {
                    fprintf(stderr, "warning: The specified PID is just "
                            "ignored since an output file is already "
                            "specified.\n");
                    goto next;
                } else {
                    char *eptr = NULL;
                    int tmp = strtol(*argv, &eptr, 10);
                    if (*eptr == '\0') {
                        if (tmp >= 0) {
                            pid = tmp;
                        } else {
                            fprintf(stderr, "error: invalid PID: '%d'\n", pid);
                            return 1;
                        }
                    } else {
                        fprintf(stderr, "error: invalid PID: '%s'\n",
                                *argv);
                        return 1;
                    }
                }
            } else {
                fprintf(stderr, "error: PID is not specified.\n");
                return 1;
            }
        } else if (strcmp(*argv, "--F") == 0 ||
                   strcmp(*argv, "--C") == 0) {
            if (IS_VALID_STRING(INIT_MODULE_OBJ) == TRUE) {
                fprintf(stderr, "error: --F and --C are exclusive.\n");
                return 1;
            } else {
                switch ((int)((*argv)[2])) {
                    case 'F': {
                        if (debug_flag) fprintf(stderr, "Fortran mode\n");
                        INIT_MODULE_OBJ   = "_xmpf_module_INIT.o";
                        INIT_PREFIX       = "_xmpf_init_";
                        MODULE_INIT_NAME  = "_xmpf_module_init_";
                        MODULE_INIT_NAME_ = "_xmpf_module_init__";
                        MODULE_INIT_ENTRY_NAME = "xmpf_module_init__";
                        break;
                    }
                    case 'C': {
                        if (debug_flag) fprintf(stderr, "C mode\n");
                        INIT_MODULE_OBJ   = "_xmpc_module_INIT.o";
                        INIT_PREFIX       = "_xmpc_init_";
                        MODULE_INIT_NAME  = "_xmpc_module_init_";
                        MODULE_INIT_NAME_ = "_xmpc_module_init__";
                        MODULE_INIT_ENTRY_NAME = "xmpc_module_init";
                        break;
                    }
                }
            }
        } else if (strcmp(*argv, "--OPTION") == 0) {
            if (IS_VALID_STRING(argv + 1) == TRUE) {
                argv++;
                cc_option = strdup(*argv);
            } else {
                fprintf(stderr, "error: A option for the C compiler is not "
                        "specified.\n");
                return 1;
            }
        } else if (strcmp(*argv, "-o") == 0) {
            if (IS_VALID_STRING(argv + 1) == TRUE) {
                argv++;
                if (pid >= 0) {
                    fprintf(stderr, "warning: The specified output file is "
                            "just ignored since a PID is already "
                            "specified.\n");
                    goto next;
                } else {
                    snprintf(init_func_object, sizeof(init_func_object),
                             "%s", *argv);
                }
            } else {
                fprintf(stderr, "error: An output file is not specified.\n");
                return 1;
            }
        } else {
            files[n_files++] = *argv;
            n_files_strlen += (strlen(*argv) + 1);
        }

        next:
        argv++;
    }

    if (n_files == 0) {
        fprintf(stderr, "error: no files are specified.\n");
        return 1;
    }
    if (pid < 0 && IS_VALID_STRING(init_func_object) == FALSE) {
        fprintf(stderr, "error: --P nor -o is not specified.\n");
        return 1;
    }
    if (IS_VALID_STRING(INIT_MODULE_OBJ) == FALSE) {
        fprintf(stderr, "error: --C nor --F is not specified.\n");
        return 1;
    }

    command_buf = (char *)malloc(sizeof(char) *
                                 (sizeof(NM) + 1 +
                                  n_files_strlen +
                                  1));
    if (command_buf == NULL) {
        fprintf(stderr, "error: can't allocate a buffer for command "
                "execution.\n");
        return 1;
    }

    //    pid = getpid();
    //    strcpy(command_buf,"nm");
    strcpy(command_buf, NM);
    for (i = 0; i < n_files; i++){
	arg = files[i];
	len = strlen(arg);
	if (len > 2 && 
	    (strcmp(&arg[len-2], ".o") == 0 || strcmp(&arg[len-2], ".a") == 0)){
	    strcat(command_buf, " ");
	    strcat(command_buf, arg);
	}
    }

    fp = popen(command_buf, "r");
    if (fp == NULL){
	fprintf(stderr, "error: %s execution failure.\n", NM);
        return 1;
    }
    init_name_len = strlen(MODULE_INIT_NAME);
    init_name_len_ = strlen(MODULE_INIT_NAME_);
    char prev[MAX_BUF];
    char *prev_ans="T";
    while(fscanf(fp,"%s",buf) == 1){
      if(strncmp(buf,".jwe",4) == 0 || strncmp(buf,"jpj.",4) == 0){
	is_K_FC = TRUE;
	continue; // Fortran compiler on the K computer
      }

      if(strncmp(buf,"_xmpc_init_all",14) == 0 || 
	 strncmp(buf,"_xmpf_main_",11) == 0) is_Mac = TRUE;

      if(strncmp(buf,".xmpf_main_",11)==0||
	 strncmp(buf,".xmpc_init_all",14)==0) is_AIX = TRUE;
      // On Mac OS X (Darwin), all module is added "_".
      // For example, __shadow_xmpc_module_init_ -> ___shadow_xmpc_module_init_
      
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

    sprintf(init_func_source,"%s%s%d.c",tmp_dir,INIT_PREFIX,pid);
    if (pid >= 0 && IS_VALID_STRING(init_func_object) == FALSE) {
      sprintf(init_func_object,"%s%s%d.o",tmp_dir,INIT_PREFIX,pid);
    }
    //    strcpy(init_func_object,INIT_MODULE_OBJ);
    fp = fopen(init_func_source,"w");
    if (fp == NULL){
      fprintf(stderr,"cannot open '%s'\n",init_func_source);
      return 1;
    }
    
    if(!is_K_FC){
      for(i=0; i<n_module_init;i++){
	char *name = module_init_names[i];
	if(is_Mac)
        strcpy(name, name+sizeof(char)); // Remove the first charactor of function name
	if(is_AIX)                                   // ___shadow_xmpc_module_init_ -> __shadow_xmpc_module_init_
	  {
	    if(strncmp(name,".",1)==0)
	      {
		fprintf(fp,"asm(\".extern  %s[GL]\");\n",name);
		break;
	      }
	  }
	else
	  fprintf(fp,"extern void %s();\n",name);
     }
      fprintf(fp,"\n");
    }
    
    fprintf(fp,"void %s(){\n",MODULE_INIT_ENTRY_NAME);

    for(i=0; i<n_module_init;i++){
      char *name = module_init_names[i];
     if(strchr(name,'.') != NULL){
       if(is_AIX)
	 {
	   fprintf(fp, "asm(\"mr 31,1\");\n"); 
	   fprintf(fp, "asm(\"nop\");\n");
	   fprintf(fp,"\t%s();\n",strcpy(name,name+1));
	   break;
	 }
       else
	 {
	   fprintf(fp, "asm(\"call %s\");\n", name);  // asm("call func"); 
	   fprintf(fp, "asm(\"nop\");");
	 }
      }
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
        return 1;
    }

    if(!debug_flag) unlink(init_func_source);
    
    return 0;
}
