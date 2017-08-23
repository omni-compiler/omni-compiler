extern void xmp_init(int argc, char **argv);
extern int ixmp_sub();
extern void xmp_finalize(int);
int main(int argc, char **argv) {

  xmp_init(argc, argv);

  ixmp_sub();

  xmp_finalize(0);

  return 0;
}
