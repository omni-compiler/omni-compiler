#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void input_file(FILE *in_file, FILE *out_file);

int main(int argc, char *argv[])
{
  int c;
  int ac;
  char *arg;
  char *cl_file = NULL;
  char *out_file_name = NULL;
  char *str_name = NULL;
  FILE *out_file, *in_file;
  
  cl_file = NULL;
  for(ac = 1; ac < argc; ac++){
    arg = argv[ac];
    if(arg[0] == '-'){
      if(strcmp(arg,"-o") == 0){
	if(++ac >= argc) goto bad_arg;
	out_file_name = argv[ac];
      } else if(strcmp(arg,"-n") == 0){
	if(++ac >= argc) goto bad_arg;
	str_name = argv[ac];
      } else {
	fprintf(stderr,"unknown option: '%s'\n",arg);
	exit(1);
      }
    } else break;
  }

  if(out_file_name == NULL)
    out_file = stdout;
  else{
    if((out_file = fopen(out_file_name,"w")) == NULL){
      fprintf(stderr,"cannot open: '%s'\n",out_file_name);
      exit(1);
    }
    // printf("out_file_name: '%s'\n",out_file_name);
  }
  
  if(str_name == NULL)  str_name = "";
  fprintf(out_file,"char _cl_prog_%s[] = \"",str_name);

  while(ac < argc){
    arg = argv[ac++];
    if(arg[0] == '-') goto bad_arg;
    // printf("file '%s'\n",arg);
    if((in_file = fopen(arg,"r")) == NULL){
      fprintf(stderr,"cannot open: '%s'\n",arg);
      exit(1);
    }
    input_file(in_file,out_file);
    fclose(in_file);
  }
  fprintf(out_file,"\";\n");
  fclose(out_file);
  exit(0);

 bad_arg:
  fprintf(stderr,"bad args\n");
  exit(1);
}
  
void input_file(FILE *in_file, FILE *out_file)
{
  int c;
  
  while((c = getc(in_file)) != EOF){
    if(c == '\n') {
      putc('\\', out_file);
      putc('n',out_file);
      putc('\\',out_file);
    }
    if(c == '\"'){
      putc('\\',out_file);
      putc('\"',out_file);
      continue;
    }
    if(c == '\\'){
      putc('\\',out_file);
      putc('\\',out_file);
      continue;
    }
    putc(c,out_file);
  }
}

