#include <stdio.h>

/* convertion from old format to new format */
main(int argc, char *argv[])
{
  char *fname1,*fname2;
  FILE *fp1,*fp2;
  struct  {
    char log_type;	/* major type */
    char proc_id;	/* processor id */
    short arg1;		/* minor type */
    int arg2;
    double time_stamp;
  } d1;  /* old */
  struct tlog_record {
    unsigned short int proc_id;	/* processor id */
    char log_type;	/* major type */
    char arg1;		/* minor type */
    int arg2;
    double time_stamp;
  } d2;

  if(argc != 3){
    printf("bad arg\n");
    exit(1);
  }

  fname1 = argv[1];
  fname2 = argv[2];
  fp1 = fopen(fname1,"r");
  fp2 = fopen(fname2,"w");
  while(fread(&d1,sizeof(d1),1,fp1) > 0){
    d2.time_stamp = d1.time_stamp;
    d2.log_type = d1.log_type;
    d2.arg1 = d2.arg2 = 0;
    d2.proc_id = (d1.proc_id << 8);  /* swap */
    fwrite(&d2,sizeof(d2),1,fp2);
  }
  fclose(fp1);
  fclose(fp2);
  exit(0);
}

