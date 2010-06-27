# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
use strict;

my $cfile = "c-token.c";
my $hfile = "c-token.h";

open(IN, "c-parser.h") || die "$!";
open(COUT, ">${cfile}") || die "$!";

print COUT<<_EOL_;
#include "c-token.h"

const CTokenInfo s_CTokenInfos[] = {
_EOL_

my $fmt = "    { %-30s, %3d },\n";

while(<IN>) {
	if(/^#define\s+(IDENTIFIER)\s+([0-9]+)\s*$/) {
		printf COUT $fmt, "\"$1\"", $2;
		last;
	}
}

my $count = 1;

while(<IN>) {
	if(/^#define\s+([a-zA-Z0-9_]+)\s+([0-9]+)$/) {
		printf COUT $fmt, "\"$1\"", $2;
		++$count;
	} else {
		last;
	}
}

print COUT<<_EOL_;
};

_EOL_

close IN;
close COUT;

open(HOUT, ">${hfile}") || die "$!";

print HOUT<<_EOL_;
#ifndef _C_TOKEN_H_
#define _C_TOKEN_H_

typedef struct {
	const char *name;
	const int code;
} CTokenInfo;

#define CTOKENINFO_SIZE ${count}

extern const CTokenInfo s_CTokenInfos[CTOKENINFO_SIZE];

#endif /* _C_TOKEN_H_ */

_EOL_

close HOUT;

print "Created ${cfile}, ${hfile}\n";


