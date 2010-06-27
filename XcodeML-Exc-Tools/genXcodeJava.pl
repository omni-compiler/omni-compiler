#!/usr/bin/perl

# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $

use strict;

my $src = "src/exc/object/Xcode.java";

open(OUT, ">".$src) || die "$!";
open(IN, "Xcode.def") || die "$!";

my $prog = $0;

print OUT<<_EOL_;
/*
 * This class is generated by ${prog}
 */
package exc.object;

import xcodeml.XmException;

import xcodeml.c.binding.gen.*;
import xcodeml.f.binding.gen.*;

public enum Xcode
{
_EOL_

my $DYN_START_NUM = 1001;
my $dynStarted = 0;
my $n = 0;

while(<IN>) {
    if(/^\s*#\s*(.+)$/) {
        print OUT "     // $1\n";
        next;
    }
    chomp;
    s/^\s*//;
    s/\s*$//;
    next if(/^$/);
    split(/\s+/);
    my($k, $xc, $cl, $clf) = @_;
    if(length($cl) == 0 || $cl eq "-") {
        $cl = "null";
    } else {
        $cl = "${cl}.class";
    }
    if(length($clf) == 0 || $clf eq "-") {
        $clf = "null";
    } else {
        $clf = "${clf}.class";
    }

    if(!$dynStarted && $xc =~ /^DYN_/) {
        $dynStarted = 1;
        $n = $DYN_START_NUM;
    }

    printf OUT ("    %-32s( %3d, '${k}', ${cl}, ${clf}),\n", $xc, $n);
    ++$n;
}

close IN;

print OUT<<_EOL_;
    ;

    private static final int ASSIGN_START_NUM = ${DYN_START_NUM};
    private int int_val;
    private char kind;
    private Class<?> xmc_class, xmf_class;
    private static int assign_index = ASSIGN_START_NUM;

    private Xcode(int int_val, char kind, Class<?> xmc_class, Class<?> xmf_class)
    {
        this.int_val = int_val;
        this.kind = kind;
        this.xmc_class = xmc_class;
        this.xmf_class = xmf_class;
    }

    public int toInt()
    {
        return int_val;
    }

    public String toXcodeString()
    {
        return toString();
    }

    public static Xcode assign() throws XmException
    {
        Xcode x = get(assign_index++);
        if(x == null)
            throw new XmException("too many Xcode assigned.");
        return x;
    }

    public static Xcode get(int intVal)
    {
        for(Xcode x : values()) {
            if(x.toInt() == intVal)
                return x;
        }
        return null;
    }

    public Class<?> getXcodeML_C_Class()
    {
        return xmc_class;
    }

    public Class<?> getXcodeML_F_Class()
    {
        return xmf_class;
    }

    public boolean isBuiltinCode()
    {
        return toInt() < ASSIGN_START_NUM;
    }
    
    public boolean isAssignedCode()
    {
        return !isBuiltinCode();
    }

    /** return true if this object is binary operation. */
    public boolean isBinaryOp()
    {
        return kind == 'B';
    }

    /** return true if this object is unary operation. */
    public boolean isUnaryOp()
    {
        return kind == 'U';
    }

    /** return ture if this object is an assignment with binary operation */
    public boolean isAsgOp()
    {
        return kind == 'A';
    }

    /** return true if this object is a terminal object */
    public boolean isTerminal()
    {
        return kind == 'T';
    }

    public boolean isFstatement()
    {
        switch(this) {
        case RETURN_STATEMENT:              case GOTO_STATEMENT:
        case F_DO_STATEMENT:                case F_DO_WHILE_STATEMENT:
        case F_IF_STATEMENT:                case F_WHERE_STATEMENT:
        case F_SELECT_CASE_STATEMENT:       case STATEMENT_LABEL:
        case F_CASE_LABEL:
        case F_ASSIGN_STATEMENT:            case F_POINTER_ASSIGN_STATEMENT:
        case F_CYCLE_STATEMENT:             case F_EXIT_STATEMENT:
        case F_CONTINUE_STATEMENT:
        case F_ALLOCATE_STATEMENT:          case F_BACKSPACE_STATEMENT:
        case F_CLOSE_STATEMENT:             case F_DEALLOCATE_STATEMENT:
        case F_END_FILE_STATEMENT:          case F_INQUIRE_STATEMENT:
        case F_NULLIFY_STATEMENT:           case F_OPEN_STATEMENT:
        case F_PRINT_STATEMENT:             case F_READ_STATEMENT:
        case F_REWIND_STATEMENT:            case F_WRITE_STATEMENT:
        case F_PAUSE_STATEMENT:             case F_STOP_STATEMENT:
        case F_ENTRY_DECL:                  case F_FORMAT_DECL:
        case F_DATA_DECL:
        case PRAGMA_LINE:                   case TEXT:
            return true;
        }
        return false;
    }

    public boolean isDefinition()
    {
        switch(this) {
        case FUNCTION_DEFINITION:
        case F_MODULE_DEFINITION:
            return true;
        }
        return false;
    }
}

_EOL_

close OUT;

print "Generated ${src}\n";


