#!/usr/bin/perl

# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
my $genSrcDir = shift || die;

$irnode = "$genSrcDir/IRNode.java";
if(!-e $irnode) { print "${irnode} is not found. Build .rng file first."; exit 1; }

@files = glob("${genSrcDir}/*.java");
$tmpf = "__tmp__";

foreach $file (@files) {
    open(IN, $file) || die;
    open(OUT, ">$tmpf") || die;

    while(<IN>) {
        print OUT;
        if(/^package/) {
            print OUT "\n";
            print OUT "import xcodeml.binding.*;\n";
        } 
    }
    close IN;
    close OUT;
    unlink $file;
    rename $tmpf, $file;
}

unlink $irnode;

