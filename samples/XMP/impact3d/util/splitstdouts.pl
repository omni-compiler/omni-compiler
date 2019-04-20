#!/usr/bin/perl
#
# extract and split STDOUT
#
# coded by Sakagami,H. 07/09/17
#
$fileopen = 0;

while (<>) {
   chomp;
   if (/impact3d-(serials|mpi1s|mpi2s|mpi3s|hpf1s|hpf2s|hpf3s|xmp1s|xmp2s|xmp2sr|xmp3s|xmp3sr)-.* started at/) {
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
   } elsif (/impact3d-(serials|mpi1s|mpi2s|mpi3s|hpf1s|hpf2s|hpf3s|xmp1s|xmp2s|xmp2sr|xmp3s|xmp3sr)-.*  ended  at/) {
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
