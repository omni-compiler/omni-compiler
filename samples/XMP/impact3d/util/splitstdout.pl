#!/usr/bin/perl
#
# extract and split STDOUT
#
# coded by Sakagami,H. 07/09/17
#
$fileopen = 0;

while (<>) {
   chomp;
   if (/impact3d-(serial|mpi1|mpi2|mpi3|hpf1|hpf2|hpf3|xmp1|xmp2|xmp2r|xmp3|xmp3r)-.* started at/) {
      if ($fileopen == 1) {
         close OUT;
         $fileopen = 0;
      }
      $file = $_;
      $file =~ s/ started at .*$/\.list/;
      open(OUT, ">$file") || die "can't open file $file.\n";
      $fileopen = 1;
      print "$file\n";
      print OUT "$_\n";
   } elsif (/impact3d-(serial|mpi1|mpi2|mpi3|hpf1|hpf2|hpf3|xmp1|xmp2|xmp2r|xmp3|xmp3r)-.*  ended  at/) {
      print OUT "$_\n";
      close OUT;
      $fileopen = 0;
   } else {
      if ($fileopen == 1) {
         print OUT "$_\n";
      }
   }
}

if ($fileopen == 1) {
   close OUT;
}
