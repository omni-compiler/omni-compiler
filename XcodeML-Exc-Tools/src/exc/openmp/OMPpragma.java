/* -*- Mode: java; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
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
    OMP_NONE,
    /*
     * directive
     */
    PARALLEL,           /* parallel <clause_list> */
    FOR,                /* (for|do) <clause_list> */
    TASK,               /* task <clause list>*/

    SIMD,               /* simd <clause list>*/
    LOOP_SIMD,    
    PARALLEL_LOOP_SIMD,    

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

    TARGET,             /* target <clause_list> */
    TARGET_DATA,        /* target data <clause_list> */
    TARGET_ENTER_DATA,
    TARGET_EXIT_DATA,
    TARGET_UPDATE,

    TARGET_SIMD,
    TARGET_PARALLEL,    /* target parallel <clause_list> */
    TARGET_PARALLEL_LOOP, /* target parallel for <clause_list> */
    TARGET_PARALLEL_LOOP_SIMD,

    TARGET_TEAMS,       /* target teams <clause_list> */
    TARGET_TEAMS_DISTRIBUTE, /* target teams distribute <clause_list>  */
    TARGET_TEAMS_DISTRIBUTE_SIMD,
    TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP, /* target teams distribute parallel for <clause_list> */
    TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD,

    TEAMS,              /* teams <clause_list> */
    TEAMS_DISTRIBUTE,   /* teams distribute <clause_list> */
    TEAMS_DISTRIBUTE_SIMD,   
    TEAMS_DISTRIBUTE_PARALLEL_LOOP, /* teams distribute parallel for <clause_list> */
    TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD,

    DISTRIBUTE,         /* distribute <clause_list> */
    DISTRIBUTE_SIMD,
    DISTRIBUTE_PARALLEL_LOOP, /* distribute parallel for <clause_list> */
    DISTRIBUTE_PARALLEL_LOOP_SIMD,

    DECLARE_REDUCTION,    /* declare reduction <id> <type-list> <combiner> <init-clause> */
    DECLARE_SIMD,      /* declare simd <clause_list> */
    DECLARE_TARGET,  /* declare target <extended-list> <clause_list> */
    DECLARE_TARGET_START, /* declare target */
    DECLARE_TARGET_END,   /* end declare target */

    /*
     * clause
     */
    DATA_NONE,           /* using internally */
    DATA_DEFAULT,   /* default(shared|none) */
    DATA_PRIVATE,   /* private(list) */
    DATA_SHARED,    /* shared(list) */
    DATA_FIRSTPRIVATE, /* firstprivate(list) */
    DATA_LASTPRIVATE,  /* lastprivate(list) */
    DATA_COPYPRIVATE, /* copyprivate(list) */
    DATA_COPYIN,  /* copyin(list) */
    _DATA_PRIVATE_SHARED, /* using internally */

    TARGET_DATA_MAP, /* map([to|from|tofrom|alloc]: list) */
    TARGET_DEVICE,  /* device(expr) */
    TARGET_UPDATE_TO, /* to(list) */
    TARGET_UPDATE_FROM, /* from(list) */
    DIR_NUM_THREADS, /* num_threads(expr) */
    NUM_TEAMS,  /* num_teams(expr) */
    THREAD_LIMIT, /* thread_limit(expr) */
    USE_DEVICE_PTR, /* use_device_ptr(list) */
    IS_DEVICE_PTR, /* is_device_ptr(list) */
    DEFAULTMAP, /* defaultmap(tofrom:scalar) */
    DEPEND,   /* depend([in|out|inout]: list */
    COLLAPSE, /* collapse(n) */
    DIST_SCHEDULE, /* dist_schedule(kind[,chunk_size]) */
    PROC_BIND,  /* proc_bind([master|cloase|spread]) */
    DATA_LINEAR, /* linaer( ) */

    SIMD_SAFELEN, /* safelen(length) */
    SIMD_SIMDLEN, /* simdlen(length) */
    SIMD_ALIGNED, /* aligned(arg-list[:alignment:) */
    SIMD_UNIFORM, /* uniform(ag-list) */
    SIMD_INBRANCH,  /* inbranch */
    SIMD_NOTINBRANCH, /* notinbranch */

    DIR_FINAL,  /* final(logical-expr) */
    DIR_UNTIED, /* untied */
    DIR_MERGEABLE, /* mergeable */
    DIR_GRAINSIZE,  /* grainsize(size) */

    DIR_ORDERED, /* orderd [(n)] */
    DIR_IF,      /* if(expression) */
    DIR_NOWAIT,  /* nowait */
    DIR_SCHEDULE,  /* schedule(block,cyclic, ...) */

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

    /*
     * for depend([in|out|inout]:)
     */
    DATA_DEPEND_IN,
    DATA_DEPEND_OUT,
    DATA_DEPEND_INOUT,

    /*
     * default clause value
     */
    DEFAULT_NONE,  /* default(none) */
    DEFAULT_SHARED, /* default(shared)  */
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
	try {
	    return valueOf(x.getString());
	} catch (IllegalArgumentException e) {
	    OMP.fatal("unknown OpenMP pragama: "+ x.getString());
	    return null;
        }
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

    public boolean isGlobalDirective(){
      switch(this){
      case THREADPRIVATE:
      case DECLARE_REDUCTION:
      case DECLARE_SIMD:
      case DECLARE_TARGET:
      case DECLARE_TARGET_START:
      case DECLARE_TARGET_END:
        return true;
      default:
        return false;
      }
    }
}
