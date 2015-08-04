/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import exc.object.Ident;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.Xobject;
import exc.object.Xtype;

/**
 * OpenMP pragma codes
 */
public enum OMPpragma
{
    /*
     * directive
     */
    PARALLEL,           /* parallel <clause_list> */
    FOR,                /* (for|do) <clause_list> */
    TASK,               /* task <clause list>*/
    SIMD,               /* simd <clause list>*/
    DECLARE,            /* declare <clause_list> */
    SECTIONS,           /* sections <clause_list> */
    SECTION,            /* section */
    SINGLE,             /* single <clause list> */
    MASTER,             /* master */
    CRITICAL,           /* critical <name> */
    BARRIER,            /* barrier */
    ATOMIC,             /* atomic */
    FLUSH,              /* flush <namelist> */
    ORDERED,            /* ordered */
    THREADPRIVATE,      /* threadprivate <namelist> */
    PARALLEL_FOR,       /* parallel <clause_list> */
    PARALLEL_SECTIONS,  /* parallel <clause_list> */
    FUNCTION_BODY,
    
    /*
     * clause
     */
    DATA_NONE,           /* using internally */
    DATA_DEFAULT,
    DATA_PRIVATE,
    DATA_SHARED,
    DATA_FIRSTPRIVATE,
    DATA_LASTPRIVATE,
    DATA_COPYPRIVATE,
    DATA_COPYIN,
    _DATA_PRIVATE_SHARED, /* using internally */
    DIR_NUM_THREADS,

    /* DATA_REDUCTION_* values are synchronized with
       OMPC_REDUCTION_* values in ompc_reduction.h */
    DATA_REDUCTION_PLUS(1),
    DATA_REDUCTION_MINUS(2),
    DATA_REDUCTION_MUL(3),
    DATA_REDUCTION_LOGAND(4),
    DATA_REDUCTION_LOGOR(5),
    DATA_REDUCTION_MIN(6),
    DATA_REDUCTION_MAX(7),
    DATA_REDUCTION_BITAND(8),   // C
    DATA_REDUCTION_BITOR(9),    // C
    DATA_REDUCTION_BITXOR(10),  // C
    DATA_REDUCTION_LOGEQV(11),  // Fortran
    DATA_REDUCTION_LOGNEQV(12), // Fortran
    DATA_REDUCTION_IAND(13),    // Fortran
    DATA_REDUCTION_IOR(14),     // Fortran
    DATA_REDUCTION_IEOR(15),    // Fortran

    DATA_DEPEND_IN,
    DATA_DEPEND_OUT,
    DATA_DEPEND_INOUT,

    DATA_FINAL,
    DIR_UNTIED,
    DIR_MERGEABLE,

    DIR_ORDERED,
    DIR_IF,
    DIR_NOWAIT,
    DIR_SCHEDULE,
    
    /*
     * default clause value
     */
    DEFAULT_NONE,
    DEFAULT_SHARED,
    DEFAULT_PRIVATE,
    _DEFAULT_NOT_SET,       /* using internally */

    /*
     * sched clause value
     */
    SCHED_NONE,
    SCHED_STATIC,
    SCHED_DYNAMIC,
    SCHED_GUIDED,
    SCHED_RUNTIME,
    SCHED_AFFINITY,
    ;

    private String name;
    
    private int id;
    
    private OMPpragma()
    {
    }
    
    private OMPpragma(int id)
    {
        this.id = id;
    }

    public boolean isDataReduction()
    {
        switch(this) {
        case DATA_REDUCTION_PLUS:
        case DATA_REDUCTION_MINUS:
        case DATA_REDUCTION_MUL:
        case DATA_REDUCTION_BITAND:
        case DATA_REDUCTION_BITOR:
        case DATA_REDUCTION_BITXOR:
        case DATA_REDUCTION_LOGAND:
        case DATA_REDUCTION_LOGOR:
        case DATA_REDUCTION_MIN:
        case DATA_REDUCTION_MAX:
        case DATA_REDUCTION_LOGEQV:
        case DATA_REDUCTION_LOGNEQV:
        case DATA_REDUCTION_IAND:
        case DATA_REDUCTION_IOR:
        case DATA_REDUCTION_IEOR:
            return true;
        }
        return false;
    }
    
    public String getName()
    {
        if(name == null)
            name = toString().toLowerCase();
        return name;
    }
    
    public int getId()
    {
        return id;
    }
    
    public static OMPpragma valueOf(Xobject x)
    {
        return valueOf(x.getString());
    }

    /** create reduction expression */
    public Xobject getReductionExpr(Xobject l, Xobject r)
    {
        Xcode binop = null;
        String func = null;
        
        switch(this) {
        case DATA_REDUCTION_PLUS:       binop = Xcode.PLUS_EXPR; break;
        case DATA_REDUCTION_MINUS:      binop = Xcode.MINUS_EXPR; break;
        case DATA_REDUCTION_MUL:        binop = Xcode.MUL_EXPR; break;
        case DATA_REDUCTION_BITAND:     binop = Xcode.BIT_AND_EXPR; break;
        case DATA_REDUCTION_BITOR:      binop = Xcode.BIT_OR_EXPR; break;
        case DATA_REDUCTION_BITXOR:     binop = Xcode.BIT_XOR_EXPR; break;
        case DATA_REDUCTION_LOGAND:     binop = Xcode.LOG_AND_EXPR; break;
        case DATA_REDUCTION_LOGOR:      binop = Xcode.LOG_OR_EXPR; break;
        case DATA_REDUCTION_LOGEQV:     binop = Xcode.F_LOG_EQV_EXPR; break;
        case DATA_REDUCTION_LOGNEQV:    binop = Xcode.F_LOG_NEQV_EXPR; break;
        case DATA_REDUCTION_MIN:        func = "min"; break;
        case DATA_REDUCTION_MAX:        func = "max"; break;
        case DATA_REDUCTION_IAND:       func = "iand"; break;
        case DATA_REDUCTION_IOR:        func = "ior"; break;
        case DATA_REDUCTION_IEOR:       func = "ieor"; break;
        default: throw new IllegalStateException(this.toString());
        }
        
        if(binop != null) {
            return Xcons.List(binop, l, r);
        } else {
            return Xcons.functionCall(
                Ident.FidentNotExternal(func, Xtype.FnumericalAllFunctionType), Xcons.List(l, r));
        }
    }
}
