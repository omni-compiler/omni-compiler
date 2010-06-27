# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
my $ERROR_H = "c-error.h";
my $TMP_FILE = "__tmp__${ERROR_H}";
my $count = 0;

open(IN, $ERROR_H) || die "$!";
open(OUT, ">".$TMP_FILE) || die "$!";

while(my $l = <IN>) {
	if($l =~ /#define C([A-Z]+)_([0-9]+)\s+"([A-Z][0-9]{3}:[^"]+)"/) {
		print OUT $l;
	} elsif($l =~ /#define C([A-Z]+)_([0-9]+)\s+"([^"]+)"/) {
		my ($k, $n, $m, $kh) = ($1, $2, $3);
		$k =~ /^(.)/;
		$kh = $1;
		print OUT "#define C${k}_${n} \"${kh}${n}: ${m}\"\n";
		++$count;
	} else {
		print OUT $l;
	}
}

close(IN);
close(OUT);

if($count > 0) {
	unlink $ERROR_H || die "$!";
	rename $TMP_FILE, $ERROR_H || die "$!";
	print "modified ${count} line(s)\n";
} else {
	unlink $TMP_FILE || die "$!";
	print "no modification\n";
}

