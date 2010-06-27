/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char *
chkIntSize(sz)
     int sz;
{
    if (sizeof(char) == sz) {
	return "char";
    } else if (sizeof(int) == sz) {
	return "int";
    } else if (sizeof(short) == sz) {
	return "short";
    } else if (sizeof(long int) == sz) {
	return "long int";
    } else if (sizeof(long long int) == sz) {
	return "long long int";
    } else {
	return "unknown";
    }
}


int
main(argc, argv)
     int argc;
     char *argv[];
{
    int sz;

    if (argc < 2) {
	return 1;
    }
    sz = atoi(argv[1]) / 8;

    printf("%s\n", chkIntSize(sz));
    return 0;
}
