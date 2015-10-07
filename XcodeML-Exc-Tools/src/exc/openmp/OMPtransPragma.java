/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import exc.object.*;
import exc.util.MachineDep;
import exc.xcodeml.XcodeMLtools;
import exc.block.*;

import java.io.File;
import java.util.*;

import xcodeml.IXobject;
import xcodeml.util.XmLog;
import xcodeml.util.XmOption;

/**
 * AST transformation by OpenMP directives.
 */
public class OMPtransPragma
{
    protected XobjectDef current_def;
    protected XobjectFile env;

    public String mainFunc;
    public String barrierFunc;
    public String currentThreadBarrierFunc;
    public String doParallelFunc;
    public String doParallelIfFunc;
    public String parallelTaskFunc;
    public String parallelTaskIfFunc;
    	
    public String defaultShedFunc;
    public String blockShedFunc;
    public String cyclicShedFunc;
    public String staticShedInitFunc;
    public String staticShedNextFunc;
    public String dynamicShedInitFunc;
    public String dynamicShedNextFunc;
    public String guidedShedInitFunc;
    public String guidedShedNextFunc;
    public String runtimeShedInitFunc;
    public String runtimeShedNextFunc;
    public String affinityShedInitFunc;
    public String affinityShedNextFunc;

    public String orderedInitFunc;
    public String orderedSetLoopIdFunc;
    public String orderedBeginFunc;
    public String orderedEndFunc;

    public String threadIdFunc;
    public String sectionInitFunc;
    public String sectionIdFunc;
    public String doSingleFunc;
    public String enterCriticalFunc;
    public String exitCriticalFunc;
    public String isMasterFunc;
    public String atomicLockFunc;
    public String atomicUnlockFunc;
    public String threadLockFunc;
    public String threadUnlockFunc;
    public String bcopyFunc; // not supported in Xcode/Fortran
    public String isLastFunc;

    public String flushFunc;
    public String getThdprvFunc; // not supported in Xcode/Fortran
    public String copyinThdprvFunc; // not supported in Xcode/Fortran

    public String allocShared; // not supported in Xcode/Fortran
    public String freeShared; // not supported in Xcode/Fortran

    public String allocaFunc; // not supported in Xcode/Fortran

    public String doReduction; // not supported in Xcode/Fortran
    public String doReductionInit; // not supported in Xcode/Fortran

    public String OMPfuncPrefix;
    public String criticalLockPrefix;
    public String reductionLockPrefix; // not supported in Xcode/Fortran
    
    public String setNumThreadsFunc;
    public String getNumThreadsFunc;
    public String getThreadNumFunc;
    public String getMaxThreadsFunc;
    
    public String setBoundsFuncPrefix;
    public String saveArrayFunc;
    public String loadArrayFunc;
    
    public final String FallocatedFunc = "allocated";
    public final String Fsize = "size";
    public final String FlboundFunc = "lbound";
    public final String FuboundFunc = "ubound";
    public final String FdoParallelWrapperPrefix = "ompf_dop";
    public final String FParallelTaskWrapperPrefix = "ompf_ptask";
    public final String FdummyFunc = "ompf_func";
    public final String FgetThdprvFuncPrefix = "ompf_getthdprv_";
    
    public static final String PROP_KEY_FINTERNAL_MODULE = "OMPtransPragma.FinternalModuel";
    
    public static final String ARGV_VAR = "_ompc_argv";
    public static final String ARGS_VAR = "_ompc_args";
    public static final String SHARE_VAR_PREFIX = "pp_";
    public static final String LOCAL_VAR_PREFIX = "p_";
    public static final String COPYPRV_VAR_PREFIX = "cp_";
    public static final String TEMP_VAR_PREFIX = "t_";
    public static final String THDPRV_LOCAL_VAR_PREFIX = "tp_";
    public static final String THDPRV_STORE_PREFIX = "ts_";
    public static final String COND_VAR = LOCAL_VAR_PREFIX + "_cond";
    public static final String NARG_VAR = LOCAL_VAR_PREFIX + "_narg";
    public static final String THDNUM_VAR = LOCAL_VAR_PREFIX + "_tn";
    public static final String THDNUM_LOOP_VAR = LOCAL_VAR_PREFIX + "_tnlp";
    public static final String FINTERNAL_MODULE_PREFIX = "ompf_internal_";
    public static final String FTHDPRV_COMMON_PREFIX = "tc_";
    
    public static final int F_MAX_PARALLEL_PARAMS = 64;
    
    public OMPtransPragma()
    {
        if(XmOption.isLanguageC()) {
            mainFunc = "ompc_main";
            OMPfuncPrefix = "ompc_func";
            barrierFunc = "ompc_barrier";
            currentThreadBarrierFunc = "ompc_current_thread_barrier";
            doParallelFunc = "ompc_do_parallel";
            doParallelIfFunc = "ompc_do_parallel_if";
    
            defaultShedFunc = "ompc_default_sched";
            blockShedFunc = "ompc_static_bsched";
            cyclicShedFunc = "ompc_static_csched";
            staticShedInitFunc = "ompc_static_sched_init";
            staticShedNextFunc = "ompc_static_sched_next";
            dynamicShedInitFunc = "ompc_dynamic_sched_init";
            dynamicShedNextFunc = "ompc_dynamic_sched_next";
            guidedShedInitFunc = "ompc_guided_sched_init";
            guidedShedNextFunc = "ompc_guided_sched_next";
            runtimeShedInitFunc = "ompc_runtime_sched_init";
            runtimeShedNextFunc = "ompc_runtime_sched_next";
            affinityShedInitFunc = "ompc_affinity_sched_init";
            affinityShedNextFunc = "ompc_affinity_sched_next";
    
            orderedInitFunc = "ompc_init_ordered";
            orderedSetLoopIdFunc = "ompc_set_loop_id";
            orderedBeginFunc = "ompc_ordered_begin";
            orderedEndFunc = "ompc_ordered_end";
    
            threadIdFunc = "ompc_thread_id";
            sectionInitFunc = "ompc_section_init";
            sectionIdFunc = "ompc_section_id";
            doSingleFunc = "ompc_do_single";
            enterCriticalFunc = "ompc_enter_critical";
            exitCriticalFunc = "ompc_exit_critical";
            isMasterFunc = "ompc_is_master";
            atomicLockFunc = "ompc_atomic_lock";
            atomicUnlockFunc = "ompc_atomic_unlock";
            threadLockFunc = "ompc_thread_lock";
            threadUnlockFunc = "ompc_thread_unlock";
            bcopyFunc = "ompc_bcopy";
            isLastFunc = "ompc_is_last";
    
            flushFunc = "ompc_flush";
            getThdprvFunc = "ompc_get_thdprv"; // pointer
            copyinThdprvFunc = "ompc_copyin_thdprv";
    
            allocShared = "ompc_alloc_shared"; // pointer
            freeShared = "ompc_free_shared";
    
            allocaFunc = "alloca";
    
            doReduction = "ompc_reduction";
            doReductionInit = "ompc_reduction_init";
    
            criticalLockPrefix = "ompc_lock_critical";
            reductionLockPrefix = "ompc_lock_reduction";
            
            setNumThreadsFunc = "ompc_set_num_threads";
            getNumThreadsFunc = "ompc_get_num_threads";
            getThreadNumFunc = "ompc_get_thread_num";
            getMaxThreadsFunc = "ompc_get_max_threads";
            
        } else {
            
            mainFunc = "ompf_main";
            OMPfuncPrefix = "ompf_func";
            barrierFunc = "ompf_barrier";
            currentThreadBarrierFunc = "ompf_current_thread_barrier";
            doParallelFunc = "ompf_do_parallel";
            doParallelIfFunc = "ompf_do_parallel_if";

            parallelTaskFunc = "ompf_parallel_task";
            parallelTaskIfFunc = "ompf_parallel_task_if";
    
            defaultShedFunc = "ompf_default_sched";
            blockShedFunc = "ompf_static_bsched";
            cyclicShedFunc = "ompf_static_csched";
            staticShedInitFunc = "ompf_static_sched_init";
            staticShedNextFunc = "ompf_static_sched_next";
            dynamicShedInitFunc = "ompf_dynamic_sched_init";
            dynamicShedNextFunc = "ompf_dynamic_sched_next";
            guidedShedInitFunc = "ompf_guided_sched_init";
            guidedShedNextFunc = "ompf_guided_sched_next";
            runtimeShedInitFunc = "ompf_runtime_sched_init";
            runtimeShedNextFunc = "ompf_runtime_sched_next";
            affinityShedInitFunc = "ompf_affinity_sched_init";
            affinityShedNextFunc = "ompf_affinity_sched_next";
    
            orderedInitFunc = "ompf_init_ordered";
            orderedSetLoopIdFunc = "ompf_set_loop_id";
            orderedBeginFunc = "ompf_ordered_begin";
            orderedEndFunc = "ompf_ordered_end";
    
            threadIdFunc = "ompf_thread_id";
            sectionInitFunc = "ompf_section_init";
            sectionIdFunc = "ompf_section_id";
            doSingleFunc = "ompf_do_single";
            enterCriticalFunc = "ompf_enter_critical";
            exitCriticalFunc = "ompf_exit_critical";
            isMasterFunc = "ompf_is_master";
            atomicLockFunc = "ompf_atomic_lock";
            atomicUnlockFunc = "ompf_atomic_unlock";
            threadLockFunc = "ompf_thread_lock";
            threadUnlockFunc = "ompf_thread_unlock";
            bcopyFunc = null;
            isLastFunc = "ompf_is_last";
    
            flushFunc = "ompf_flush";
            getThdprvFunc = null;
            copyinThdprvFunc = null;
    
            allocShared = null;
            freeShared = null;
    
            allocaFunc = null;
    
            doReduction = "ompf_reduction";
            doReductionInit = null;
    
            criticalLockPrefix = "ompf_lock_critical";
            reductionLockPrefix = null;
            
            setNumThreadsFunc = "ompf_set_num_threads";
            getNumThreadsFunc = "ompf_get_num_threads";
            getThreadNumFunc = "ompf_get_thread_num";
            getMaxThreadsFunc = "ompf_get_max_threads";
            setBoundsFuncPrefix = "ompf_set_bounds_";
            saveArrayFunc = "ompf_save_array_header";
            loadArrayFunc = "ompf_load_array_header";
        }
    }

    // pass3: do transformation for OMP pragma
    public void run(FuncDefBlock def)
    {
        Block b;
        Block fblock = def.getBlock();
        current_def = def.getDef();
        env = current_def.getFile();

        OMP.debug("pass3:");
        BlockIterator i = new bottomupBlockIterator(fblock);
        for(i.init(); !i.end(); i.next()) {
            b = i.getBlock();
            if(b.Opcode() == Xcode.OMP_PRAGMA) {
                b = transPragma((PragmaBlock)b);
                if(b != null)
                    i.setBlock(b);
            }
        }

        Block body = fblock.getBody().getHead();
        transFunctionBody((CompoundBlock)body, fblock.getBody().getDecls());
    }

    public Ident OMPfuncIdent(String name)
    {
        if(name == null)
            throw new IllegalArgumentException("may be generating unsupported function call");
        
        Xtype t;
        if(XmOption.isLanguageC()) {
            if(name == getThdprvFunc || name == allocShared || name == allocaFunc)
                t = Xtype.Function(Xtype.Pointer(Xtype.voidType));
            else
                t = Xtype.Function(Xtype.intType);
        } else {
            if(name == getMaxThreadsFunc || name == getThreadNumFunc ||
                name == getNumThreadsFunc || name == sectionIdFunc)
                t = Xtype.FintFunctionType;
            else if(name == isLastFunc || name == isMasterFunc ||
                name == doSingleFunc ||
                name == staticShedNextFunc || name == dynamicShedNextFunc ||
                name == guidedShedNextFunc || name == runtimeShedNextFunc ||
                name == affinityShedNextFunc)
                t = Xtype.FlogicalFunctionType;
            else
                t = Xtype.FsubroutineType;
        }
        return current_def.declExternIdent(name, t);
    }

    public Ident OMPFintrinsicIdent(String name)
    {
        Xtype t;
        
        if(name == FallocatedFunc) {
            t = Xtype.FlogicalFunctionType;
        } else {
            t = Xtype.FintFunctionType;
        }
        
        return Ident.Fident(name, t ,false, false, env);
    }
    
    // pass3:
    // write pragma
    public Block transPragma(PragmaBlock pb, XobjectDef current_def)
    {
        this.current_def = current_def;
        env = current_def.getFile();
        return transPragma(pb);
    }

    Block transPragma(PragmaBlock pb)
    {
        OMPinfo i = (OMPinfo)pb.getProp(OMP.prop);
        OMP.debug("Pragma:"+i.pragma);
        switch(i.pragma) {
        case BARRIER: /* barrier */
            return transBarrier(pb, i);

        case PARALLEL: /* new parallel section */
            return transParallelRegion(pb, i);

        case FOR: /* for <clause_list> */
	    if(pb.getBody().getHead().Opcode() == Xcode.OMP_PRAGMA)
		return null;
            return transFor(pb, i);

        case SECTIONS: /* sections <clause_list> */
            return transSections(pb, i);

        case SINGLE:
            return transSingle(pb, i);

        case MASTER: /* master */
            return transMaster(pb, i);

        case CRITICAL: /* critical <name> */
            return transCritical(pb, i);

        case ATOMIC:
            return transAtomic(pb, i);

        case FLUSH:
            return transFlush(pb, i);

        case ORDERED:
            return transOrdered(pb, i);
        
        case TASK:
        	return transTaskRegion(pb,i);
	
        case SIMD:
        	return null;
        default:
             OMP.fatal("unknown pragma");
//             ignore it
            return null;
        }
    }

    public Block transBarrier(PragmaBlock b, OMPinfo i)
    {
        return Bcons.Statement(OMPfuncIdent(barrierFunc).Call(null));
    }
    
    public Block transParallelRegion(PragmaBlock b, OMPinfo i)
    {
        if(XmOption.isLanguageC())
            return transCparallelRegion(b, i);
        return transFparallelRegion(b, i);
    }
    
    public Block transCparallelRegion(PragmaBlock b, OMPinfo i)
    {
        Ident func_id = current_def.declStaticIdent(
            env.genSym(OMPfuncPrefix), Xtype.Function(Xtype.voidType));
        Block bp;

        // move blocks to body
        BlockList body = Bcons.emptyBody(i.getIdList(), null);
        addDataSetupBlock(body, i);
        threadprivateCSetup(body, i);
        body.add(Bcons.COMPOUND(b.getBody()));
        bp = dataUpdateBlock(i);
        if(bp != null)
            body.add(bp);
        addCcalleeBlock(body, func_id, i);
        
        // new block to be replaced
        BlockList ret_body = Bcons.emptyBody();
        addCcallerBlock(ret_body, func_id, i);
        
        return Bcons.COMPOUND(ret_body);
    }
    
    private void addCcallerBlock(BlockList ret_body, Ident func_id, OMPinfo i)
    {
        List<Xobject> region_args = i.getRegionArgs();
        
        if(region_args.size() == 0) {
            if(i.hasIfExpr())
                ret_body.add(Bcons.Statement(OMPfuncIdent(doParallelIfFunc).Call(
                    Xcons.List(i.getIfExpr(), func_id.Ref()))));
            else
                ret_body.add(Bcons.Statement(OMPfuncIdent(doParallelFunc).Call(
                    Xcons.List(func_id.Ref()))));
            return;
        }
        
        BasicBlock bblock = new BasicBlock();
        // caller side, create local reference structure
        Xtype voidP_t = Xtype.Pointer(Xtype.voidType);
        Ident av = Ident.Local(ARGV_VAR, Xtype.Array(voidP_t, region_args.size()));
        ret_body.addIdent(av);
        for(int ii = 0; ii < region_args.size(); ii++) {
            Xobject s = region_args.get(ii);
            s = Xcons.Set(av.Index(ii), Xcons.Cast(voidP_t, s));
            bblock.add(s);
        }
        
        if(i.getNumThreads() != null)
            bblock.add(OMPfuncIdent(setNumThreadsFunc).Call(Xcons.List(i.getNumThreads())));
        
        if(i.getIfExpr() != null)
            bblock.add(OMPfuncIdent(doParallelIfFunc).Call(
                Xcons.List(i.getIfExpr(), func_id.Ref(), av.Ref())));
        else
            bblock.add(OMPfuncIdent(doParallelFunc).Call(Xcons.List(func_id.Ref(), av.Ref())));
        ret_body.add(bblock);
    }
    
    private void addCcalleeBlock(BlockList body, Ident func_id, OMPinfo i)
    {
        Xobject param_id_list = Xcons.IDList();
        
        if(i.getRegionParams().size() != 0) {
            BasicBlock bblock = new BasicBlock();
            Xtype voidP_t = Xtype.Pointer(Xtype.voidType);
            Ident pv = Ident.Param(ARGS_VAR, Xtype.Pointer(voidP_t));
            param_id_list.add(pv);
            for(int ii = 0; ii < i.getRegionParams().size(); ii++) {
                Xobject s = i.getRegionParams().get(ii).Ref();
                s = Xcons.Set(s, Xcons.Cast(s.Type(), pv.Index(ii)));
                bblock.add(s);
            }
            body.insert(bblock);
        }

        // make prototype for func_id
        ((FunctionType)func_id.Type()).setFuncParamIdList(param_id_list);
        current_def.insertBeforeThis(XobjectDef.Func(func_id,
            param_id_list, null, body.toXobject()));
    }

    private void addFcallerBlock(BlockList ret_body, Ident func_id, Ident dop_func_id, OMPinfo i)
    {
        List<Xobject> region_args = i.getRegionArgs();
        BasicBlock bblock = new BasicBlock();
        XobjList funcArgs = Xcons.List(func_id.getAddr());
        
        for(Xobject a : region_args) {
            funcArgs.add(a);
        }

        if(i.getNumThreads() != null)
            bblock.add(OMPfuncIdent(setNumThreadsFunc).Call(Xcons.List(i.getNumThreads())));
        
        if(i.hasIfExpr())
            funcArgs.insert(i.getIfExpr());
        bblock.add(dop_func_id.Call(funcArgs));
        
        ret_body.add(bblock);
    }
    
    private void addFtaskCallerBlock(BlockList ret_body, Ident func_id, Ident dop_func_id, OMPinfo i)
    {
        List<Xobject> region_args = i.getRegionArgs();
        BasicBlock bblock = new BasicBlock();
        XobjList funcArgs = Xcons.List(func_id.getAddr());
        
        for(Xobject a : region_args) {
            funcArgs.add(a);
        }

        if(i.getNumThreads() != null)
            bblock.add(OMPfuncIdent(setNumThreadsFunc).Call(Xcons.List(i.getNumThreads())));
        
        if(i.hasFinalExpr())
    		funcArgs.insert(i.getFinalExpr());
        if(i.hasIfExpr())
            funcArgs.insert(i.getIfExpr());
        
        bblock.add(dop_func_id.Call(funcArgs));
        
        ret_body.add(bblock);
    }

    
    private void addFcalleeBlock(Ident func_id, Ident dop_func_id, BlockList body,
        XobjList id_list, XobjList decls, BlockList orgBody, OMPinfo i)
    {
        XobjList func_param_list = Xcons.List();
        for(Xobject x : id_list)
            func_param_list.add(x);
        
        if(orgBody.getDecls() != null) {
            HashMap<String, Ident> nameMap = new HashMap<String, Ident>();
            for(Xobject x : id_list) {
                Ident id = (Ident)x;
                nameMap.put(id.getName(), id);
            }
            
            FidentCollector ic = new FidentCollector(orgBody, nameMap);
            ic.collectIdentName(body.toXobject(), i);
            ic.copyDecls(id_list, decls, body, i, false);
        }
        
        completeFstructTypeSymbol(orgBody.getIdentList(), (XobjList)id_list);
        
        ((FunctionType)func_id.Type()).setFuncParamIdList(func_param_list);
        current_def.insertBeforeThis(XobjectDef.Func(func_id,
            id_list, decls, body.toXobject()));
    }
    
    private void transFioStatements(BlockList body)
    {
        if(!XmOption.isLanguageF() || !XmOption.isAtomicIO())
            return;
        
        topdownBlockIterator ite = new topdownBlockIterator(body);
        for(ite.init(); !ite.end(); ite.next()) {
            Block b = ite.getBlock();
            if(b == null)
                continue;
            BasicBlock bb = b.getBasicBlock();
            if(bb == null)
                continue;
            for(Statement s = bb.getHead(); s != null; s = s.getNext()) {
                Xobject x = s.getExpr();
                if(x == null)
                    continue;
                switch(x.Opcode()) {
                case F_PRINT_STATEMENT:
                case F_WRITE_STATEMENT:
                case F_READ_STATEMENT:
                case F_OPEN_STATEMENT:
                case F_CLOSE_STATEMENT:
                case F_INQUIRE_STATEMENT:
                case F_REWIND_STATEMENT:
                case F_END_FILE_STATEMENT:
                case F_BACKSPACE_STATEMENT:
                    break;
                default:
                    continue;
                }
                if(x.isAtomicStmt())
                    continue;
                x.setIsAtomicStmt(true);
                b.insert(OMPfuncIdent(atomicLockFunc).Call());
                b.add(OMPfuncIdent(atomicUnlockFunc).Call());
            }
        }
    }

    /** Fortran: add type name symbol used in src_id_list to dst_id_list */
    private void completeFstructTypeSymbol(XobjList src_id_list, XobjList dst_id_list)
    {
        List<Ident> stid_list = new ArrayList<Ident>();
        for(Xobject x : src_id_list) {
        	if(StorageClass.FTYPE_NAME.equals(((Ident)x).getStorageClass()))
        		stid_list.add((Ident)x);
        }
        
        for(Xobject x : dst_id_list) {
        	Xtype t = x.Type();
        	if(t == null)
        		continue;
    		if(t.isFarray())
    			t = t.getRef();
            if(t != null && t.isStruct()) {
            	Ident tag = t.getTagIdent();
            	if(tag == null)
            		throw new IllegalStateException();
            	if(dst_id_list.find(tag.getName(), IXobject.FINDKIND_TAGNAME) == null)
            		stid_list.add(t.getTagIdent());
            }
        }
        for(Ident stid : stid_list)
        	dst_id_list.add(stid);
    }
    
    private void addFdoParallelWrapperBlock(Ident dop_func_id, XobjList func_params,
            BlockList orgBody, OMPinfo i)
        {
            XobjList funcArgs = Xcons.List();
            int nargs = func_params.Nargs();
            OMP.debug("nargs 0=" + nargs);
            
            for(Xobject a : func_params) {
                //argument of character type passed with character length to runtime function
                OMP.debug("func_params =" + a.Type().toString());
                if(a.Type().isFcharacter())
                    ++nargs;
                funcArgs.add(a);
            }
            OMP.debug("nargs=" + nargs);

            if(XmOption.isLanguageF() && nargs > OMPtransPragma.F_MAX_PARALLEL_PARAMS) {
                XmLog.fatal(orgBody.getHeadLineNo(), "too many variables in parallel region.");
            }
            
            Ident fid = Ident.FidentNotExternal(FdummyFunc, Xtype.FsubroutineType);
            func_params.insert(fid);
            funcArgs.insert(fid);

            Xobject cond = null;
            if(i.hasIfExpr()) {
                cond = Ident.Fident(COND_VAR, Xtype.FlogicalType);
                func_params.insert(cond);
            } else {
                cond = Xcons.FlogicalConstant(true);
            }
            funcArgs.insert(cond);

            
            XobjList body_id_list = Xcons.List();
            HashMap<String, Ident> nameMap = new HashMap<String, Ident>();
            FidentCollector ic = new FidentCollector(orgBody, nameMap);
            for(Xobject a : func_params) {
                if(a == fid) {
                    a = a.copy();
                    a.setType(Xtype.FexternalSubroutineType);
                }
                body_id_list.add(a);
                ic.collectIdentInIdent((Ident)a); 
            }
            for(Ident id : nameMap.values())
                body_id_list.add(id);
            
            completeFstructTypeSymbol(orgBody.getIdentList(), body_id_list);

            //funcArgs.insert(Xcons.IntConstant(nargs));
            BlockList body = new BlockList();

            // There are two kind of do_parallel functions.
            // 1. ompf_do_parallel_(int nargs, cfunc f, ...)
            //
            // 2. ompf_do_parallel_1_(cfunc f, void *a1)
            //    ompf_do_parallel_2_(cfunc f, void *a1, void *a2)
            //    ...
            //    ompf_do_parallel_64_(cfunc f, void *a1, void *a2, ... , void *a64)
            //
            // va_list is used in ompf_do_parallel_ and is not used in ompf_do_parallel_N_.
            // In x86-64 with gfortran, it is likely to fail to call function in which va_list is used.
            // So we call ompf_do_parallel_N_ instead of ompf_do_parallel_.
            
            Ident idDoParallel = Ident.FidentNotExternal(
                doParallelFunc + "_" + nargs,
                Xtype.FsubroutineType);
            body.add(idDoParallel.Call(funcArgs));
            
            ((FunctionType)dop_func_id.Type()).setFuncParamIdList(func_params);
            XobjList decls = Xcons.List();
            nameMap.clear();
            ic.copyDecls(body_id_list, decls, body, i, true);
            
            XobjectDef def = XobjectDef.Func(dop_func_id,
                body_id_list, decls, body.toXobject());
            
            current_def.insertBeforeThis(def);

            // add interface of runtime function
            XobjList dopar_params = (XobjList)idDoParallel.Type().getFuncParam();
            if(dopar_params == null) {
                dopar_params = Xcons.List();
                ((FunctionType)idDoParallel.Type()).setFuncParam(dopar_params);
            }
            XobjList paramDecls = Xcons.List();
            //Ident dummyNarg = Ident.Fident(NARG_VAR, Xtype.FintType);
            Ident dummyFunc = Ident.FidentNotExternal(FdummyFunc, Xtype.FsubroutineType);
            //paramDecls.add(Xcons.List(Xcode.VAR_DECL, dummyNarg));
            //dopar_params.add(dummyNarg);
            if(!i.hasIfExpr()) {
                Ident c = Ident.Fident("cond", Xtype.FlogicalType);
                paramDecls.add(Xcons.List(Xcode.VAR_DECL, c));
                dopar_params.add(c);
            }
            
            paramDecls.add(Xcons.FinterfaceFunctionDecl(dummyFunc, null));
            for(Xobject a : func_params) {
                a = convertFidentTypeForParamDecl((Ident)a, i);
                paramDecls.add(Xcons.List(Xcode.VAR_DECL, a));
                dopar_params.add(a);
            }
            
            copyFuseDecl(orgBody.getDecls(), paramDecls);
            
            def.getFuncDecls().add(
                Xcons.FinterfaceFunctionDecl(idDoParallel, paramDecls));
        }
        
    private void addFtaskWrapperBlock(Ident dop_func_id, XobjList func_params,
            BlockList orgBody, OMPinfo i)
        {
            XobjList funcArgs = Xcons.List();
            int nargs = func_params.Nargs();
            OMP.debug("nargs 0="+nargs);

            for(Xobject a : func_params) {
                //argument of character type passed with character length to runtime function
                OMP.debug("func_params =" + a.Type().toString());
                if(a.Type().isFcharacter())
                    ++nargs;
                funcArgs.add(a);
            }
            OMP.debug("nargs ="+nargs);
            if(XmOption.isLanguageF() && nargs > OMPtransPragma.F_MAX_PARALLEL_PARAMS) {
                XmLog.fatal(orgBody.getHeadLineNo(), "too many variables in parallel region.");
            }
            
            Ident fid = Ident.FidentNotExternal(FdummyFunc, Xtype.FsubroutineType);
            func_params.insert(fid);
            funcArgs.insert(fid);

            Xobject cond = null;
            
            if(i.hasFinalExpr()) {
                cond = Ident.Fident("TASK_"+COND_VAR+"_F", Xtype.FlogicalType);
                func_params.insert(cond);
            } else {
                cond = Xcons.FlogicalConstant(false);
            }            
            funcArgs.insert(cond);
            
            if(i.hasIfExpr()) {
                cond = Ident.Fident("TASK_"+COND_VAR, Xtype.FlogicalType);
                func_params.insert(cond);
            } else {
                cond = Xcons.FlogicalConstant(true);
            }
            funcArgs.insert(cond);
            
            XobjList body_id_list = Xcons.List();
            HashMap<String, Ident> nameMap = new HashMap<String, Ident>();
            FidentCollector ic = new FidentCollector(orgBody, nameMap);
            for(Xobject a : func_params) {
                if(a == fid) {
                    a = a.copy();
                    a.setType(Xtype.FexternalSubroutineType);
                }
                body_id_list.add(a);
                ic.collectIdentInIdent((Ident)a); 
            }
            for(Ident id : nameMap.values())
                body_id_list.add(id);
            
            completeFstructTypeSymbol(orgBody.getIdentList(), body_id_list);
            //funcArgs.insert(Xcons.IntConstant(nargs));
            BlockList body = new BlockList();

            // There are two kind of do_parallel functions.
            // 1. ompf_do_parallel_(int nargs, cfunc f, ...)
            //
            // 2. ompf_do_parallel_1_(cfunc f, void *a1)
            //    ompf_do_parallel_2_(cfunc f, void *a1, void *a2)
            //    ...
            //    ompf_do_parallel_64_(cfunc f, void *a1, void *a2, ... , void *a64)
            //
            // va_list is used in ompf_do_parallel_ and is not used in ompf_do_parallel_N_.
            // In x86-64 with gfortran, it is likely to fail to call function in which va_list is used.
            // So we call ompf_do_parallel_N_ instead of ompf_do_parallel_.
            
            Ident idDoParallel = Ident.FidentNotExternal(
                parallelTaskFunc + "_" + nargs,
                Xtype.FsubroutineType);
            body.add(idDoParallel.Call(funcArgs));

            ((FunctionType)dop_func_id.Type()).setFuncParamIdList(func_params);
            XobjList decls = Xcons.List();
            nameMap.clear();
            ic.copyDecls(body_id_list, decls, body, i, true);
            
            XobjectDef def = XobjectDef.Func(dop_func_id,
                body_id_list, decls, body.toXobject());
            
            current_def.insertBeforeThis(def);

            // add interface of runtime function
            XobjList dopar_params = (XobjList)idDoParallel.Type().getFuncParam();
            if(dopar_params == null) {
                dopar_params = Xcons.List();
                ((FunctionType)idDoParallel.Type()).setFuncParam(dopar_params);
            }
            XobjList paramDecls = Xcons.List();
            //Ident dummyNarg = Ident.Fident(NARG_VAR, Xtype.FintType);
            Ident dummyFunc = Ident.FidentNotExternal(FdummyFunc, Xtype.FsubroutineType);
            //paramDecls.add(Xcons.List(Xcode.VAR_DECL, dummyNarg));
            //dopar_params.add(dummyNarg);

            if(!i.hasFinalExpr()) {
                Ident c = Ident.Fident("cond_F", Xtype.FlogicalType);
                paramDecls.add(Xcons.List(Xcode.VAR_DECL, c));
                dopar_params.add(c);
            }            
            
            if(!i.hasIfExpr()) {
                Ident c = Ident.Fident("cond", Xtype.FlogicalType);
                paramDecls.add(Xcons.List(Xcode.VAR_DECL, c));
                dopar_params.add(c);
            }
            

            paramDecls.add(Xcons.FinterfaceFunctionDecl(dummyFunc, null));
            for(Xobject a : func_params) {
                a = convertFidentTypeForParamDecl((Ident)a, i);
                paramDecls.add(Xcons.List(Xcode.VAR_DECL, a));
                dopar_params.add(a);
            }
            
            copyFuseDecl(orgBody.getDecls(), paramDecls);
            
            def.getFuncDecls().add(
                Xcons.FinterfaceFunctionDecl(idDoParallel, paramDecls));
        }
        
    private void copyFuseDecl(Xobject src_decls, Xobject dst_decls)
    {
        if(src_decls == null)
            return;
        
        XobjList use_list = Xcons.List();
        
        for(Xobject x : (XobjList)src_decls) {
            if(x.Opcode() == Xcode.F_USE_DECL || x.Opcode() == Xcode.F_USE_ONLY_DECL)
                use_list.add(x);
        }
        
        use_list.reverse();
        for(Xobject x : use_list)
            dst_decls.insert(x);
    }
    
    private Ident convertFidentTypeForParamDecl(Ident id, OMPinfo i)
    {
        Xtype t = id.Type();
        switch(t.getKind()) {
        case Xtype.BASIC:
            if(t.isFcharacter() && t.getFlen() != null) {
                t = t.copy();
                id = (Ident)id.copy();
                id.setType(t);
                ((BasicType)t).setFlen(Xcons.IntConstant(-1)); //variable length
            }
            break;
        case Xtype.F_ARRAY:
            t = t.copy();
            id = (Ident)id.copy();
            id.setType(t);
            if(t.isFassumedShape())
                t.convertFindexRange(false, false, i.getBlock());
            else {
                Xobject[] sizeExprs = t.getFarraySizeExpr();
                for(int ii = 0; ii < t.getNumDimensions(); ++ii) {
                    // remove symbols in index range and set dummy size.
                    sizeExprs[ii] = Xcons.FindexRange(
                        Xcons.IntConstant(1), Xcons.IntConstant(1));
                }
                t.convertFindexRange(true, true, i.getBlock());
            }
            break;
        }
        return id;
    }

    public Block transFparallelRegion(PragmaBlock b, OMPinfo i)
    {
        Ident func_id = current_def.declExternIdent(
            env.genExportSym(OMPfuncPrefix, current_def.getName()), Xtype.FsubroutineType);
        Ident dop_func_id = Ident.FidentNotExternal(
            env.genExportSym(FdoParallelWrapperPrefix, current_def.getName()), Xtype.FsubroutineType);
        Block bp;

        // move blocks to body
        BlockList body = Bcons.emptyBody(i.getIdList(), null);
        addDataSetupBlock(body, i);
        body.add(Bcons.COMPOUND(b.getBody()));
        
        bp = dataUpdateBlock(i);
        if(bp != null)
	  body.add(bp);

        FunctionBlock fb = getParentFuncBlock(b);
        if(fb == null)
            throw new IllegalStateException("parent is not FunctionBlock");

        XobjList calleeDelcls = Xcons.List();
        BlockList fbBody = fb.getBody();

        OMP.debug("transParallelRegion"+fbBody.getDecls().toString());
        
        threadprivateFSetup(calleeDelcls, fbBody.getDecls(), body, i, true);
        OMP.debug("fbBody:"+fbBody.getDecls().toString());
        XobjList id_list = Xcons.List();
        for(Ident x : i.getRegionParams())
            id_list.add(x);
        
        XobjList dop_id_list = Xcons.List();
        for(Xobject a : id_list)
            dop_id_list.add(a);
        
        transFioStatements(fbBody);
        
        // rewrite array index range if needed.
        OMPrewriteExpr r = new OMPrewriteExpr();
        for(Xobject a: (XobjList)i.getIdList()){
            r.run(a, b, i.env);}
        
        addFcalleeBlock(func_id, dop_func_id, body, id_list, calleeDelcls, fbBody, i);
        
        addFdoParallelWrapperBlock(dop_func_id, dop_id_list, fbBody, i);
        
        BlockList ret_body = Bcons.emptyBody();
        addFcallerBlock(ret_body, func_id, dop_func_id, i);
        
        // add interface of ompf_dop_#
        XobjList dop_params = Xcons.List();
        Ident dummyFunc = Ident.FidentNotExternal(FdummyFunc, Xtype.FsubroutineType);
        dop_params.add(Xcons.FinterfaceFunctionDecl(dummyFunc, null));
        
        for(Xobject a : dop_id_list) {
            a = convertFidentTypeForParamDecl((Ident)a, i);
            // rewrite array index range if needed.
            r.run(a, b, i.env);
            dop_params.add(Xcons.List(Xcode.VAR_DECL, a));
        }
        Xobject fbDecls = fbBody.getDecls();
        copyFuseDecl(fbDecls, dop_params);
        
        if(fbDecls == null) {
            fbDecls = Xcons.List();
            fbBody.setDecls(fbDecls);
        }
        
        fbDecls.add(Xcons.FinterfaceFunctionDecl(dop_func_id, dop_params));

        return Bcons.COMPOUND(ret_body);
    }

    public Block transFtaskRegion(PragmaBlock b, OMPinfo i)
    {
        Ident func_id = current_def.declExternIdent(
            env.genExportSym(OMPfuncPrefix, current_def.getName()), Xtype.FsubroutineType);
        Ident dop_func_id = Ident.FidentNotExternal(
            env.genExportSym(FParallelTaskWrapperPrefix, current_def.getName()), Xtype.FsubroutineType);
        Block bp;
        // move blocks to body
        BlockList body = Bcons.emptyBody(i.getIdList(), null);
        addDataSetupBlock(body, i);
        body.add(Bcons.COMPOUND(b.getBody()));
      
        bp = dataUpdateBlock(i);
        if(bp != null)
        	body.add(bp);

        FunctionBlock fb = getParentFuncBlock(b);
        if(fb == null)
            throw new IllegalStateException("parent is not FunctionBlock");

        XobjList calleeDelcls = Xcons.List();
        BlockList fbBody = fb.getBody();
        
        OMP.debug("transFtaskRegion"+fbBody.getDecls().toString());
        
        threadprivateFSetup(calleeDelcls, fbBody.getDecls(), body, i, true);
        
        XobjList id_list = Xcons.List();
        for(Ident x : i.getRegionParams())
            id_list.add(x);
        
        XobjList dop_id_list = Xcons.List();
        for(Xobject a : id_list)
            dop_id_list.add(a);
        
        transFioStatements(fbBody);
        
        // rewrite array index range if needed.
        OMPrewriteExpr r = new OMPrewriteExpr();
        for(Xobject a: (XobjList)i.getIdList()){
        	OMP.debug("Task"+a.toString()+" "+b.toXobject().toString()+" "+i.pragma);
            r.run(a, b, i.env);}
        
        addFcalleeBlock(func_id, dop_func_id, body, id_list, calleeDelcls, fbBody, i);
        
        addFtaskWrapperBlock(dop_func_id, dop_id_list, fbBody, i);
        
        BlockList ret_body = Bcons.emptyBody();
        addFtaskCallerBlock(ret_body, func_id, dop_func_id, i);
        
        // add interface of ompf_dop_#
        XobjList dop_params = Xcons.List();
        Ident dummyFunc = Ident.FidentNotExternal(FdummyFunc, Xtype.FsubroutineType);
        dop_params.add(Xcons.FinterfaceFunctionDecl(dummyFunc, null));
        
        for(Xobject a : dop_id_list) {
            a = convertFidentTypeForParamDecl((Ident)a, i);
            // rewrite array index range if needed.
            r.run(a, b, i.env);
            dop_params.add(Xcons.List(Xcode.VAR_DECL, a));
        }
        Xobject fbDecls = fbBody.getDecls();
        copyFuseDecl(fbDecls, dop_params);
        
        if(fbDecls == null) {
            fbDecls = Xcons.List();
            fbBody.setDecls(fbDecls);
        }
        
        fbDecls.add(Xcons.FinterfaceFunctionDecl(dop_func_id, dop_params));

        return Bcons.COMPOUND(ret_body);
    }
    
    private static FunctionBlock getParentFuncBlock(Block b)
    {
        Block pb = b.getParentBlock();
        if(pb == null)
            return null;
        if(pb instanceof FunctionBlock)
            return (FunctionBlock)pb;
        return getParentFuncBlock(pb);
    }
    
    public Block transFor(PragmaBlock b, OMPinfo i)
    {
        ForBlock for_block = (ForBlock)b.getBody().getHead();
        for_block.Canonicalize(); // canonicalize again
        Block bp;

        if(!for_block.isCanonical())
            OMP.fatal("transFor: not canonical");

        BlockList loop_body = Bcons.emptyBody(Xcons.IDList(), null);
        CompoundBlock comp_block = (CompoundBlock)Bcons.COMPOUND(loop_body);
        Xobject ind_var = for_block.getInductionVar();
        BlockList ret_body = Bcons.emptyBody(i.getIdList(), null);

        addDataSetupBlock(ret_body, i);

        boolean is_ind_ptr = ind_var.Type().isPointer();
        boolean needs_vb = XmOption.isLanguageC() &&
        	(MachineDep.sizeEquals(ind_var.Type(), Xtype.indvarPtrType, b) == false);
        Xtype indvarType = XmOption.isLanguageC() ? Xtype.indvarType : Xtype.FuintPtrType;
        Xtype uintPtrType = MachineDep.getUintPtrType();
        Xtype btype = XmOption.isLanguageC() ? ind_var.Type() : indvarType;
        Xtype step_type = is_ind_ptr ? Xtype.intType : ind_var.Type();
        Ident lb_var = Ident.Local(env.genSym(ind_var.getName()), btype);
        Ident ub_var = Ident.Local(env.genSym(ind_var.getName()), btype);
        Ident step_var = Ident.Local(env.genSym(ind_var.getName()), step_type);
        ret_body.addIdent(lb_var);
        ret_body.addIdent(ub_var);
        ret_body.addIdent(step_var);
        Xobject step_addr = step_var.getAddr();
        Xobject step_ref = step_var.Ref();

        // indvar_t type (void pointer type) variables
        Ident vlb_var = null, vub_var = null;
        Xobject vlb, vub, vlb_addr, vub_addr;

        if(needs_vb) {
            vlb_var = Ident.Local(env.genSym(ind_var.getName()), Xtype.indvarType);
            vub_var = Ident.Local(env.genSym(ind_var.getName()), Xtype.indvarType);
            vlb = vlb_var.Ref();
            vub = vub_var.Ref();
            vlb_addr = vlb_var.getAddr();
            vub_addr = vub_var.getAddr();
            
            ret_body.addIdent(vlb_var);
            ret_body.addIdent(vub_var);
            
            loop_body.add(Xcons.Set(lb_var.Ref(), Xcons.Cast(lb_var.Type(),
                Xcons.Cast(uintPtrType, vlb))));
            loop_body.add(Xcons.Set(ub_var.Ref(), Xcons.Cast(ub_var.Type(),
                Xcons.Cast(uintPtrType, vub))));
            loop_body.add((Block)for_block);
        } else {
            vlb = lb_var.Ref();
            vub = ub_var.Ref();
            vlb_addr = lb_var.getAddr();
            vub_addr = ub_var.getAddr();
            loop_body.add((Block)for_block);
        }
        
        BasicBlock bb = new BasicBlock();
        ret_body.add(bb);
        
        if(for_block.getInitBBlock() != null) {
            for(Statement s : for_block.getInitBBlock()) {
                if(s.getNext() == null)
                    break;
                s.remove();
                bb.add(s);
            }
        }
        
        bb.add(Xcons.Set(lb_var.Ref(), for_block.getLowerBound()));
        bb.add(Xcons.Set(ub_var.Ref(), for_block.getUpperBound()));
        
        if(needs_vb) {
            bb.add(Xcons.Set(vlb, Xcons.Cast(indvarType,
                Xcons.Cast(uintPtrType, lb_var.Ref()))));
            bb.add(Xcons.Set(vub, Xcons.Cast(indvarType,
                Xcons.Cast(uintPtrType, ub_var.Ref()))));
        }
        
        if(is_ind_ptr) {
            bb.add(Xcons.Set(step_var.Ref(),
                Xcons.binaryOp(Xcode.MUL_EXPR, for_block.getStep(),
                    Xcons.SizeOf(ind_var.Type().getRef()))));
        } else {
            bb.add(Xcons.Set(step_var.Ref(), for_block.getStep()));
        }

        if(i.getNumThreads() != null)
            bb.add(OMPfuncIdent(setNumThreadsFunc).Call(Xcons.List(i.getNumThreads())));
        
        for_block.setLowerBound(lb_var.Ref());
        for_block.setUpperBound(ub_var.Ref());
        if(is_ind_ptr) {
            for_block.setStep(Xcons.binaryOp(Xcode.DIV_EXPR, step_ref,
                Xcons.SizeOf(ind_var.Type().getRef())));
        } else {
            for_block.setStep(step_ref);
        }

        if(i.ordered) {
            boolean use_alt = false;
            Xobject arg = ind_var;
            
            bb.add(OMPfuncIdent(orderedInitFunc).Call(
                Xcons.List(vlb, step_var.Ref())));
            Xobject setLoopId;
            if(XmOption.isLanguageC())
                setLoopId = OMPfuncIdent(orderedSetLoopIdFunc).Call(
                    Xcons.List(Xcons.Cast(indvarType, ind_var)));
            else {
                use_alt = !MachineDep.sizeEquals(btype, ind_var.Type(), b);
                if(use_alt) {
                    arg = Ident.Fident(env.genSym(ind_var.getName()), btype);
                    ret_body.addIdent((Ident)arg);
                    arg = ((Ident)arg).getAddr();
                }
                setLoopId = OMPfuncIdent(orderedSetLoopIdFunc).Call(
                    Xcons.List(arg));
            }
            
            for_block.getBody().insert(setLoopId);
            if(use_alt)
                for_block.getBody().insert(Xcons.Set(arg, ind_var));
        }
        
        boolean needs_reinit_vb = needs_vb;
        
        switch(i.sched) {
        case SCHED_NONE:
            bb.add(OMPfuncIdent(defaultShedFunc).Call(
                Xcons.List(vlb_addr, vub_addr, step_addr)));
            ret_body.add(comp_block);
            break;
        case SCHED_STATIC:
            if(i.sched_chunk == null) {
                /* same as SCHED_NONE */
                bb.add(OMPfuncIdent(blockShedFunc).Call(
                    Xcons.List(vlb_addr, vub_addr, step_addr)));
                ret_body.add(comp_block);
                needs_reinit_vb = false;
                break;
            }
            if(i.sched_chunk.Opcode() == Xcode.INT_CONSTANT && i.sched_chunk.getInt() == 1) {
                /* cyclic schedule */
                Ident step_var0 = Ident.Local(env.genSym(ind_var.getName()), step_type);
                ret_body.addIdent(step_var0);
                bb.add(Xcons.Set(step_var0.Ref(), step_var.Ref()));

                bb.add(OMPfuncIdent(cyclicShedFunc).Call(
                    Xcons.List(vlb_addr, vub_addr, step_addr)));
                ret_body.add(comp_block);

                /* adjust last value of loop variable */
                Xobject adjust = Xcons.Set(ind_var,
                    Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.binaryOp(
                        Xcode.MINUS_EXPR, ind_var, step_var.Ref()), step_var0.Ref()));
                ret_body.add(Bcons.Statement(adjust));
                needs_reinit_vb = false;
                break;
            }

            /* otherwise, block scheduling */
            bb.add(OMPfuncIdent(staticShedInitFunc).Call(
                Xcons.List(vlb, vub, step_ref, i.sched_chunk)));
            ret_body.add(Bcons.DO_WHILE(OMPfuncIdent(staticShedNextFunc).Call(
                Xcons.List(vlb_addr, vub_addr)), comp_block));
            break;

        case SCHED_AFFINITY:
            Xobject arg = Xcons.List(vlb, vub, step_ref);
            arg.add(i.sched_chunk.getArg(0));
            arg.add(i.sched_chunk.getArg(1));
            arg.add(i.sched_chunk.getArg(2));
            arg.add(i.sched_chunk.getArg(3));
            bb.add(OMPfuncIdent(affinityShedInitFunc).Call(arg));
            ret_body.add(Bcons.DO_WHILE(OMPfuncIdent(affinityShedNextFunc).Call(
                Xcons.List(vlb_addr, vub_addr)), comp_block));
            break;

        case SCHED_DYNAMIC:
            if(i.sched_chunk == null)
                i.sched_chunk = Xcons.IntConstant(1);
            bb.add(OMPfuncIdent(dynamicShedInitFunc).Call(
                Xcons.List(vlb, vub, step_ref, i.sched_chunk)));
            ret_body.add(Bcons.DO_WHILE(OMPfuncIdent(dynamicShedNextFunc).Call(
                Xcons.List(vlb_addr, vub_addr)), comp_block));
            break;
        case SCHED_GUIDED:
            if(i.sched_chunk == null)
                i.sched_chunk = Xcons.IntConstant(1);
            bb.add(OMPfuncIdent(guidedShedInitFunc).Call(
                Xcons.List(vlb, vub, step_ref, i.sched_chunk)));
            ret_body.add(Bcons.DO_WHILE(OMPfuncIdent(guidedShedNextFunc).Call(
                Xcons.List(vlb_addr, vub_addr)), comp_block));
            break;
        case SCHED_RUNTIME:
            bb.add(OMPfuncIdent(runtimeShedInitFunc).Call(
                Xcons.List(vlb, vub, step_ref)));
            ret_body.add(Bcons.DO_WHILE(OMPfuncIdent(runtimeShedNextFunc).Call(
                Xcons.List(vlb_addr, vub_addr)), comp_block));
            break;
        default:
            OMP.fatal("unknown schedule: " + i.sched);
        }
        
        if(needs_reinit_vb) {
            loop_body.add(Xcons.Set(vlb, Xcons.Cast(indvarType,
                Xcons.Cast(uintPtrType, lb_var.Ref()))));
            loop_body.add(Xcons.Set(vub, Xcons.Cast(indvarType,
                Xcons.Cast(uintPtrType, ub_var.Ref()))));
        }
        
        bp = dataUpdateBlock(i);
        if(bp != null)
            ret_body.add(bp);

        if(!i.no_wait)
            ret_body.add(Bcons.Statement(OMPfuncIdent(barrierFunc).Call(null)));

        return Bcons.COMPOUND(ret_body);
    }

    public Block transSections(PragmaBlock b, OMPinfo i)
    {
        Block bp;
        BlockList ret_body = Bcons.emptyBody(i.getIdList(), null);
        int n = 0;

        addDataSetupBlock(ret_body, i);

        n = 0;
        for(bp = b.getBody().getHead(); bp != null; bp = bp.getNext())
            n++;

        BasicBlock bblock = new BasicBlock();
        if(i.getNumThreads() != null)
            bblock.add(OMPfuncIdent(setNumThreadsFunc).Call(Xcons.List(i.getNumThreads())));
        bblock.add(OMPfuncIdent(sectionInitFunc).Call(Xcons.List(Xcons.IntConstant(n))));
        ret_body.add(bblock);
        
        Xobject label = null;
        
        if(XmOption.isLanguageC()) {
            label = Xcons.Symbol(Xcode.IDENT, env.genSym("L"));
            ret_body.add(Bcons.LABEL(label));
        }
        
        BlockList body = Bcons.emptyBody();
        
        n = 0;
        while((bp = b.getBody().getHead()) != null) {
            bp.remove();
            if(XmOption.isLanguageC()) {
                body.add(Bcons.CASE(Xcons.IntConstant(n)));
                body.add(bp);
                body.add(Bcons.GOTO(label));
            } else {
		BlockList bpp = new BlockList(bp);
                body.add(Bcons.FcaseLabel(Xcons.List(
                    Xcons.List(Xcode.F_VALUE, Xcons.IntConstant(n))), bpp, null));
            }
            ++n;
        }
        
        Xobject cond = OMPfuncIdent(sectionIdFunc).Call(null);
        
        if(XmOption.isLanguageC()) {
            ret_body.add(Bcons.SWITCH(BasicBlock.Cond(cond), body));
        } else {
            // "case default
            //    exit"
            BlockList bl = new BlockList();
            bl.add(Xcons.List(Xcode.F_EXIT_STATEMENT));
            body.add(Bcons.FcaseLabel(null, bl, null));

            ret_body.add(Bcons.Fdo(null, null,
                Bcons.blockList(Bcons.FselectCase(cond, body, null)), null));
        }

        bp = dataUpdateBlock(i);
        if(bp != null)
            ret_body.add(bp);

        if(!i.no_wait)
            ret_body.add(Bcons.Statement(OMPfuncIdent(barrierFunc).Call(null)));
        return Bcons.COMPOUND(ret_body);
    }
    
    public Block transSingle(PragmaBlock b, OMPinfo i)
    {
        BlockList then_body = Bcons.emptyBody();
        BlockList else_body = null;
        BlockList ret_body = Bcons.emptyBody(i.getIdList(), null);

        addDataSetupBlock(then_body, i);
        then_body.add(Bcons.COMPOUND(b.getBody()));
        boolean hasCopyPrv = i.hasCopyPrivate();
        
        if(hasCopyPrv) {
            else_body = Bcons.emptyBody();
            else_body.add(Bcons.Statement(
                OMPfuncIdent(barrierFunc).Call(null)));
            
            for(OMPvar v : i.getVarList()) {
                if(v.is_copy_private) {
                    Xobject sz = XmOption.isLanguageC() ? v.getSize() : null;
                    then_body.add(getVarCopyStatement(v.id.Type(),
                        v.getCopyPrivateAddr(), v.getPrivateAddr(), sz));
                    else_body.add(getVarCopyStatement(v.id.Type(),
                        v.getPrivateAddr(), v.getCopyPrivateAddr(), sz));
                }
            }
            
            then_body.add(Bcons.Statement(
                OMPfuncIdent(barrierFunc).Call(null)));
        }
        
        ret_body.add(
            Bcons.IF(BasicBlock.Cond(OMPfuncIdent(doSingleFunc).Call(null)),
            then_body, else_body));
        
        if(!i.no_wait)
            ret_body.add(Bcons.Statement(OMPfuncIdent(barrierFunc).Call(null)));
        
        return Bcons.COMPOUND(ret_body);
    }

    public Block transMaster(PragmaBlock b, OMPinfo i)
    {
        return Bcons.IF(BasicBlock.Cond(OMPfuncIdent(isMasterFunc).Call()), b.getBody(), null);
    }

    public Block transCritical(PragmaBlock b, OMPinfo i)
    {
        String critical_lock_name = criticalLockPrefix;
        if(i.arg != null)
            critical_lock_name +=  "_" + i.arg.getName();
        Ident lock_id = current_def.declGlobalIdent(critical_lock_name, Xtype.VoidPtrOrFuintPtr());
        BlockList body = b.getBody();
        body.insert(OMPfuncIdent(enterCriticalFunc).Call(Xcons.List(lock_id.getAddr())));
        body.add(OMPfuncIdent(exitCriticalFunc).Call(Xcons.List(lock_id.getAddr())));
        return Bcons.COMPOUND(body);
    }
    
    private Xobject getFtempVar(BlockList bl, Xobject x)
    {
        if(x.isVarRef())
            return x;
        
        Xtype t = x.Type().copy();
        t.setIsFintentIN(false);
        t.setIsFintentOUT(false);
        t.setIsFintentINOUT(false);
      
        if(t.isBasic() && (t.getBasicType() == BasicType.F_NUMERIC ||
            t.getBasicType() == BasicType.F_NUMERIC_ALL)) {
            return x;
        }
        
        Ident var = Ident.Fident(env.genSym(TEMP_VAR_PREFIX), t);
        bl.add(Xcons.Set(var.getAddr(), x));
        
        return var;
    }

    public Block transAtomic(PragmaBlock b, OMPinfo i)
    {
        BlockList body = b.getBody();
        BasicBlock bb = body.getHead().getBasicBlock();
        Statement s = bb.getHead();
        
        if(XmOption.isLanguageC()) {
            s.insert(OMPfuncIdent(atomicLockFunc).Call(null));
            Xobject x = s.getExpr();
            switch(x.Opcode()) {
            case POST_INCR_EXPR:    case POST_DECR_EXPR:
            case PRE_INCR_EXPR:     case PRE_DECR_EXPR:
                break;
            default:
                Xobject rv = x.right();
                Ident rv_var = Ident.Local(env.genSym(TEMP_VAR_PREFIX), rv.Type());
                body.addIdent(rv_var);
                s.insert(Xcons.Set(rv_var.Ref(), rv));
                x.setRight(rv_var.Ref());
                break;
            }
            Xobject lv = x.operand();
            Ident addr_var = Ident.Local(env.genSym(TEMP_VAR_PREFIX), Xtype.Pointer(lv.Type()));
            body.addIdent(addr_var);
            x.setOperand(Xcons.PointerRef(addr_var.Ref()));
            s.insert(Xcons.Set(addr_var.Ref(), Xcons.AddrOf(lv)));
            s.add(OMPfuncIdent(atomicUnlockFunc).Call(null));
        } else {
            body.add(OMPfuncIdent(atomicLockFunc).Call(null));
            
            Xobject x = s.getExpr(); // (OperatorCode var operand func?)
            Xobject var = x.getArg(0);
            Xobject op1 = getFtempVar(body, x.getArg(1));
            Xobject op2 = getFtempVar(body, x.getArg(2));
            Xobject rv = null;
            
            if(x.Opcode().isBinaryOp()) {
                rv = Xcons.List(x.Opcode(), var.Type(), op1, op2);
            } else {
                Xobject isIntrinsic = Xcons.IntConstant(1);
                rv = Xcons.List(Xcode.FUNCTION_CALL, var.Type(),
                    x.getArg(3), Xcons.List(op1, op2), isIntrinsic);
            }
            body.add(Xcons.Set(var, rv));
            body.add(OMPfuncIdent(atomicUnlockFunc).Call(null));
            body.removeFirst();
        }
        
        return Bcons.COMPOUND(body);
    }

    Block transFlush(PragmaBlock b, OMPinfo i)
    {
        if(i.flush_vars == null || XmOption.isLanguageF()) {
            return Bcons.Statement(OMPfuncIdent(flushFunc).Call(
                Xcons.List(Xcons.IntConstant(0), Xcons.IntConstant(0))));
        } else {
            BasicBlock bb = new BasicBlock();
            for(Xobject v : i.flush_vars) {
                // flush_vars is list of flushFunc arguments
                bb.add(OMPfuncIdent(flushFunc).Call(v));
            }
            return Bcons.BasicBlock(bb);
        }
    }

    public Block transOrdered(PragmaBlock b, OMPinfo i)
    {
        BlockList body = b.getBody();
        body.insert(OMPfuncIdent(orderedBeginFunc).Call(null));
        body.add(OMPfuncIdent(orderedEndFunc).Call(null));
        return Bcons.COMPOUND(body);
    }
    
    public Block transTaskRegion(PragmaBlock b, OMPinfo i)
    {
        if(XmOption.isLanguageC())
            return transCtaskRegion(b, i);
        return transFtaskRegion(b, i);
    }
    public Block transCtaskRegion(PragmaBlock b,OMPinfo i){return null;}
    
    public Block transSimd(PragmaBlock b, OMPinfo i)
    {
        BlockList body = b.getBody();
        return Bcons.COMPOUND(body);
    }

    private Xobject getFallocateLocalArray(Xobject varAry, Xobject varShare, Xtype t, Xobject lastDimSize)
    {
        Xobject[] indices = new Xobject[t.getNumDimensions() + 1];
        for(int ii = 0; ii < indices.length - 1; ++ii) {
            indices[ii] = Xcons.FarrayIndex(
                OMPFintrinsicIdent(Fsize).Call(
                    Xcons.List(varShare, Xcons.IntConstant(ii + 1))));
        }
        indices[indices.length - 1] = Xcons.FarrayIndex(lastDimSize);
        
        return Xcons.Fallocate(varAry, indices);
    }
    
    private Xobject getVarCopyStatement(Xtype t, Xobject dstAddr, Xobject srcAddr, Xobject size)
    {
        if(t.isArray())
            return OMPfuncIdent(bcopyFunc).Call(Xcons.List(dstAddr, srcAddr, size));
        return Xcons.Set(Xcons.PointerRef(dstAddr), Xcons.PointerRef(srcAddr));
    }

    protected void addDataSetupBlock(BlockList bl, OMPinfo i)
    {
    	OMP.debug("addDataSetUPBlock");
    	boolean flag = false;
        for(OMPvar v : i.getVarList()) {
            if(v.is_first_private || v.is_reduction || v.is_copyin || v.isVariableArray()) {
                flag = true;
                break;
            }
        }
        if(!flag)
            return;
    	OMP.debug("addDataSetUPBlock 2");

        BasicBlock bb = new BasicBlock();
        BlockList tmpbl = new BlockList();

        for(OMPvar v : i.getVarList()) {
            Xobject privAddr = v.getPrivateAddr();
            if(v.isVariableArray() && privAddr != null) {
                //alloca_f.Declared(); // make already declared
                bb.add(Xcons.Set(privAddr, Xcons.Cast(privAddr.Type(),
                    OMPfuncIdent(allocaFunc).Call(Xcons.List(v.getSize())))));
            }
        }
        
        for(OMPvar v : i.getVarList()) {
            if(v.is_first_private) {
                bb.add(getVarCopyStatement(v.id.Type(), v.getPrivateAddr(), v.getSharedAddr(),
                    XmOption.isLanguageC() ? v.getSize() : null));
            }
        }

        // reduction
        boolean hasReduction = false;
        
        for(OMPvar v : i.getVarList()) {
            if(!v.is_first_private && v.is_reduction) {
                hasReduction = true;
                break;
            }
        }

        if(hasReduction) {
            
            if(XmOption.isLanguageC()) {
                for(OMPvar v : i.getVarList()) {
                    if(v.is_first_private || !v.is_reduction)
                        continue;
                    bb.add(Xcons.Set(Xcons.PointerRef(v.getPrivateAddr()), v.reductionInitValue()));
                }
            } else {

                tmpbl.add(OMPfuncIdent(threadLockFunc).Call());
                
                for(OMPvar v : i.getVarList()) {
                    if(v.is_first_private || !v.is_reduction)
                        continue;

                    tmpbl.add(Bcons.IF(Xcons.unaryOp(Xcode.LOG_NOT_EXPR,
                        OMPFintrinsicIdent(FallocatedFunc).Call(Xcons.List(v.getSharedArray()))),
                        getFallocateLocalArray(v.getSharedArray(), v.getSharedAddr(), v.id.Type(),
                            OMPfuncIdent(getNumThreadsFunc).Call()), null));
                }
                
                tmpbl.add(OMPfuncIdent(threadUnlockFunc).Call());

                for(OMPvar v : i.getVarList()) {
                    if(v.is_first_private || !v.is_reduction)
                        continue;

                    Xtype tp = v.getPrivateAddr().Type().copy();
                    tp.unsetIsFtarget();
                    tp.setIsFpointer(true);
                    v.getPrivateAddr().setType(tp);
                    
                    tmpbl.add(Xcons.FpointerAssignment(v.getPrivateAddr(),
                        Xcons.FarrayRef(v.getSharedArray(), Xcons.binaryOp(Xcode.PLUS_EXPR,
                            OMPfuncIdent(getThreadNumFunc).Call(), Xcons.IntConstant(1)))));
                    tmpbl.add(Xcons.Set(v.getPrivateAddr(), v.reductionInitValue()));
                }
            }
        }

        // copyin
        for(OMPvar v : i.getVarList()) {
            if(v.is_first_private || !v.is_copyin)
                continue;
            
            Ident thdprv_id = v.id;
            Ident local_id = v.getThdPrvLocalId();
            if(local_id == null)
                OMP.fatal("bad copyin: " + thdprv_id.getName());
            
            if(XmOption.isLanguageC()) {
                bb.add(OMPfuncIdent(copyinThdprvFunc).Call(
                    Xcons.List(local_id.Ref(), thdprv_id.getAddr(),
                        Xcons.SizeOf(thdprv_id.Type()))));
            } else {
                tmpbl.add(Bcons.IF(
                    Xcons.List(Xcode.LOG_NOT_EXPR, OMPfuncIdent(isMasterFunc).Call()),
                    Xcons.Set(local_id.Ref(), thdprv_id.getAddr()), null));
                tmpbl.add(OMPfuncIdent(barrierFunc).Call());
            }
        }

        if(!bb.isEmpty())
            bl.add(bb);
        if(!tmpbl.isEmpty()) {
            for(Block b = tmpbl.getHead(); b != null; b = b.getNext()) {
                bl.add(b);
            }
        }
    	OMP.debug("addDataSetUPBlock End");
        
    }
    
    private void addCreductionUpdateBlock(BlockList body, OMPinfo i)
    {
        BasicBlock bb = new BasicBlock();
        
        for(OMPvar v : i.getVarList()) {
            if(!v.is_reduction)
                continue;
            
            int bt;
            if(v.id.Type().isEnum()) {
                bt = BasicType.INT;
            } else if(v.id.Type().isNumeric()) {
                bt = v.id.Type().getBasicType();
            } else {
                OMP.fatal("bad reduction variable type");
                continue;
            }
            
            bb.add(OMPfuncIdent(doReduction).Call(
                Xcons.List(v.getPrivateAddr(), v.getSharedAddr(), Xcons.IntConstant(bt),
                    Xcons.IntConstant(v.reduction_op.getId()))));
        }

        body.add(bb);
    }
    
    private void addFreductionUpdateBlock(BlockList body, OMPinfo i)
    {
        boolean hasReduction = false;
        
        for(OMPvar v : i.getVarList()) {
            if(v.is_reduction) {
                hasReduction = true;
                break;
            }
        }
        
        if(!hasReduction)
            return;
        
        body.add(OMPfuncIdent(currentThreadBarrierFunc).Call());
        Xobject varThdNum = Ident.Fident(THDNUM_LOOP_VAR, Xtype.intType, env).getAddr();
        BlockList doBody = Bcons.emptyBody();
        
        for(OMPvar v : i.getVarList()) {
            if(!v.is_reduction)
                continue;
            
            Xobject varRdcShare = v.getSharedAddr();
            Xobject varRdcAry = v.getSharedArray();
            varRdcAry.setIsDelayedDecl(true);
            
            doBody.add(Xcons.Set(varRdcShare,
                v.reduction_op.getReductionExpr(varRdcShare,
                    Xcons.FarrayRef(varRdcAry, varThdNum))));
        }
        
        Xobject idxRange = Xcons.FindexRange(
            Xcons.IntConstant(1), OMPfuncIdent(getNumThreadsFunc).Call());
        
        body.add(Bcons.IF(OMPfuncIdent(isMasterFunc).Call(),
            Bcons.Fdo(varThdNum, idxRange, doBody, null), null));
        body.add(OMPfuncIdent(currentThreadBarrierFunc).Call());
    }

    public Block dataUpdateBlock(OMPinfo i)
    {
        boolean reduction_flag = false;
        boolean lastprivate_flag = false;
        for(OMPvar v : i.getVarList()) {
            if(v.is_last_private)
                lastprivate_flag = true;
            if(v.is_reduction)
                reduction_flag = true;
        }
        if(!lastprivate_flag && !reduction_flag)
            return null;

        BlockList body = Bcons.emptyBody();
        if(reduction_flag) {
            if(XmOption.isLanguageC()) {
                addCreductionUpdateBlock(body, i);
            } else {
                addFreductionUpdateBlock(body, i);
            }
        }

        if(lastprivate_flag) {
            BasicBlock bb = new BasicBlock();
            for(OMPvar v : i.getVarList()) {
                if(v.is_last_private) {
                    bb.add(getVarCopyStatement(v.id.Type(),
                        v.getSharedAddr(), v.getPrivateAddr(),
                        (XmOption.isLanguageC() ? v.getSize() : null)));
                }
            }
            body.add(Bcons.IF(OMPfuncIdent(isLastFunc).Call(null), Bcons.BasicBlock(bb), null));
        }
        return Bcons.COMPOUND(body);
    }

    public void transFunctionBody(CompoundBlock b, XobjectDef current_def)
    {
        this.current_def = current_def;
        env = current_def.getFile();
        transFunctionBody(b, current_def.getFuncDecls());
    }

    public void transFunctionBody(CompoundBlock b, Xobject funcBlockDecls)
    {
        OMPinfo i;
        if((i = (OMPinfo)b.getProp(OMP.prop)) == null)
            return;
        BlockList body = b.getBody();
        threadprivateSetup(funcBlockDecls, body, i, false);
        transFioStatements(body);
    }

    /**
     * C: add block for threadprivate variable setup
     */
    private void threadprivateCSetup(BlockList body, OMPinfo i)
    {
        BasicBlock bb = new BasicBlock();
        for(OMPvar v : i.getThdPrvVarList()) {
            Ident org_id = v.id;
            Ident local_id = v.getThdPrvLocalId();
            body.addIdent(local_id);
            Xobject initAddr = org_id.isCglobalVarOrFvar() ?
                org_id.getAddr() : v.getSharedAddr();

            Xobject x = OMPfuncIdent(getThdprvFunc).Call(
                Xcons.List(
                    Xcons.Symbol(Xcode.VAR_ADDR, Xtype.Pointer(org_id.Type()),
                    THDPRV_STORE_PREFIX + org_id.getName()),
                    Xcons.SizeOf(org_id.Type()), initAddr));
            bb.add(Xcons.Set(local_id.Ref(), Xcons.Cast(local_id.Type(), x)));
        }
        body.insert(bb);
    }

    /**
     * Fortran: get or create internal module.
     */
    private FmoduleBlock getFinternalModule(Xobject orgBodyDecls)
    {
        FmoduleBlock mod = (FmoduleBlock)env.getProp(PROP_KEY_FINTERNAL_MODULE);
        if(mod != null)
            return mod;
        
        String name = new File(env.getSourceFileName()).getName().trim().toLowerCase();
        if(name.length() == 0)
            name = "blank";
        int idx = name.lastIndexOf('.');
        name = FINTERNAL_MODULE_PREFIX + name.substring(0, idx)
            .replaceAll("[ ./$%&<>{}~|\\`\"'@+*]", "_");
        
        mod = new FmoduleBlock(name, env);
        env.setProp(PROP_KEY_FINTERNAL_MODULE, mod);
        
        copyFuseDecl(orgBodyDecls, mod.getBody().getDecls());
        
        return mod;
    }

    /**
     * Fortran: add identifier to function parameter list and declarations.
     */
    private static void addFparamToList(Xobject ids, XobjList decls, String name, Xtype t)
    {
        Ident param = Ident.Fident(name, t);
        ids.add(param);
        decls.add(Xcons.List(Xcode.VAR_DECL,
            Xcons.Symbol(Xcode.VAR, param.Type(), param.getName())));
    }
    
    /**
     * Fortran: add ompf_save_array_header interface declation
     */
    private Ident addFsaveArrayFuncInterface(Xobject orgBodyDecls, XobjList decls, Xtype varAryType)
    {
        XobjList paramDecls = Xcons.List();
        XobjList params = Xcons.IDList();
        
        Xtype varAry_nos = varAryType.copy();
        varAry_nos.unsetIsFsave();
        addFparamToList(params, paramDecls, "vendor", Xtype.FintType);
        addFparamToList(params, paramDecls, "src", varAry_nos);
        addFparamToList(params, paramDecls, "dst", Xtype.FuintPtrType);
        
        Ident idFunc = current_def.declStaticIdent(
            saveArrayFunc, new FunctionType(Xtype.FvoidType));
        ((FunctionType)idFunc.Type()).setFuncParamIdList(params);
        copyFuseDecl(orgBodyDecls, paramDecls);

        // add interface for loadAddrFunc
        decls.add(Xcons.FinterfaceFunctionDecl(idFunc, paramDecls));
        
        return idFunc;
    }

    /**
     * Fortran: add ompf_load_array_header interface decl.
     */
    private Ident addFloadArrayFuncInterface(Xobject orgBodyDecls, XobjList decls, Xtype varAryType)
    {
        XobjList paramDecls = Xcons.List();
        XobjList params = Xcons.IDList();
        
        Xtype varAry_nos = varAryType.copy();
        varAry_nos.unsetIsFsave();
        addFparamToList(params, paramDecls, "vendor", Xtype.FintType);
        addFparamToList(params, paramDecls, "src", Xtype.FuintPtrType);
        addFparamToList(params, paramDecls, "dst", varAry_nos);
        addFparamToList(params, paramDecls, "ndims", Xtype.FintType);
        
        Ident idFunc = current_def.declStaticIdent(
            loadArrayFunc, new FunctionType(Xtype.FvoidType));
        ((FunctionType)idFunc.Type()).setFuncParamIdList(params);
        copyFuseDecl(orgBodyDecls, paramDecls);

        // add interface for loadAddrFunc
        decls.add(Xcons.FinterfaceFunctionDecl(idFunc, paramDecls));
        
        return idFunc;
    }
    
    /**
     * Fortran: add ompf_set_bounds interface decl.
     */
    private Ident addFsetBoundsFuncInterface(Xobject orgBodyDecls,
        XobjList decls, int ndims, Xtype varPrvType, Xtype varShareType)
    {
        XobjList paramDecls = Xcons.List();
        XobjList params = Xcons.IDList();
        Xtype varPrv_nop = varPrvType.copy();
        varPrv_nop.setIsFpointer(false);
        addFparamToList(params, paramDecls, "vendor", Xtype.FintType);
        addFparamToList(params, paramDecls, "src", varPrv_nop);
        addFparamToList(params, paramDecls, "dst", varPrv_nop);
        
        for(int ii = 0; ii < ndims; ++ii) {
            addFparamToList(params, paramDecls, "lb" + (ii + 1), Xtype.FintType);
        }
        
        Ident idFunc = current_def.declStaticIdent(
            setBoundsFuncPrefix + ndims, new FunctionType(Xtype.FvoidType));
        ((FunctionType)idFunc.Type()).setFuncParamIdList(params);
        copyFuseDecl(orgBodyDecls, paramDecls);

        // add interface for setBoundsFunc
        decls.add(Xcons.FinterfaceFunctionDecl(idFunc, paramDecls));
        
        return idFunc;
    }
    
    /**
     * Fortran: add ompf_get_thdprv_* subroutine
     */
    private Ident addFgetThreadPrivateFunc(Xobject orgBodyDecls, OMPvar v, OMPinfo i,
        String str_fname, String mid_Name)
    {
        FmoduleBlock mod = getFinternalModule(orgBodyDecls);
        XobjList mod_id_list = (XobjList)mod.getBody().getIdentList();
        
        int vendor = XmOption.getCompilerVendor();
        Xobject constVendor = Xcons.IntConstant(vendor);
        Xobject varAry = v.getSharedArray();
        Xobject varPrv = v.getPrivateAddr();
        Xobject varShare = v.getSharedAddr();
        String cname = FTHDPRV_COMMON_PREFIX + mid_Name + "_" + v.id.getName();
        
        BlockList bl = Bcons.emptyBody();
        Block fbody = Bcons.COMPOUND(bl);
        Xtype ts = varShare.Type();
        if(ts.isFarray()) {
            ts = ts.copy();
            ts.convertFindexRange(true, false, i.getBlock());
        }
        varShare.setIsDelayedDecl(true);
        varShare.Type().setIsFtarget(true);
        Ident idShare = new Ident(varShare.getName(), StorageClass.FSAVE,
            ts, varShare, VarScope.LOCAL);
        Ident idPrv = new Ident(varPrv.getName(), StorageClass.FLOCAL,
            varPrv.Type(), varPrv, VarScope.LOCAL);
        XobjList fparams = Xcons.List(Xcode.ID_LIST, idShare, idPrv);
        Xtype ft = new FunctionType(null, Xtype.FvoidType, fparams, 0, false, null, null);
        Ident fname = Xcons.Ident(str_fname, StorageClass.FFUNC, ft,
            Xcons.Symbol(Xcode.FUNC_ADDR, ft, str_fname), VarScope.LOCAL);
        FunctionBlock fb = new FunctionBlock(
            fname, Xcons.IDList(), Xcons.List(), fbody, null, env);
        
        XobjList fdecls = (XobjList)fb.getBody().getDecls();
        if(fdecls == null) {
            fdecls = Xcons.List();
            fb.getBody().setDecls(fdecls);
        }
        copyFuseDecl(orgBodyDecls, fdecls);
        
        // idFlag is for distinguishing thread private storage is allocated or not
        Xtype flagt = Xtype.FlogicalType.copy(null);
        Xobject varFlag = Xcons.Symbol(Xcode.VAR, flagt, THDPRV_STORE_PREFIX + "_alc");
        varFlag.setIsDelayedDecl(true);
        
        Xtype hdrt = Xtype.FuintPtrType.copy();
        Xobject varHdr = Xcons.Symbol(Xcode.VAR, hdrt, THDPRV_STORE_PREFIX + "_hdr");
        varHdr.setIsDelayedDecl(true);
        
        Ident idFlag = new Ident(varFlag.getName(), StorageClass.FCOMMON,
            varFlag.Type(), varFlag, VarScope.LOCAL);
        Ident idHdr = new Ident(varHdr.getName(), StorageClass.FCOMMON,
            varHdr.Type(), varHdr, VarScope.LOCAL);
        fdecls.add(Xcons.List(Xcode.F_COMMON_DECL, Xcons.List(Xcode.F_VAR_LIST,
            Xcons.Symbol(Xcode.IDENT, cname),
            Xcons.List(Xcons.FvarRef(idHdr), Xcons.FvarRef(idFlag)))));
        
        // idAry is for thread private storage
        Ident idAry = new Ident(varAry.getName(), StorageClass.FSAVE,
            varAry.Type(), varAry, VarScope.LOCAL);
        
        mod_id_list.add(idAry);
        
        // add interface for loadAddrFunc
        Ident idLoadArrayFunc = addFloadArrayFuncInterface(orgBodyDecls, fdecls, varAry.Type());
        Ident idSaveArrayFunc = addFsaveArrayFuncInterface(orgBodyDecls, fdecls, varAry.Type());
        
        BlockList then11 = Bcons.emptyBody();
        BlockList then12 = Bcons.emptyBody();
        
        then12.add(getFallocateLocalArray(varAry, varShare, v.id.Type(),
            Xcons.binaryOp(Xcode.MINUS_EXPR, OMPfuncIdent(getNumThreadsFunc).Call(),
                Xcons.IntConstant(1))));
        then12.add(idSaveArrayFunc.Call(Xcons.List(constVendor, varAry, varHdr)));
        Xobject bssVal = null;
        Xtype brt = varAry.Type().getBaseRefType();
        if(brt.isNumeric()) {
            bssVal = Xcons.IntConstant(0);
        } else if(brt.isBool()) {
            bssVal = Xcons.FlogicalConstant(false);
        }
        if(bssVal != null) {
            then12.add(Xcons.Set(varAry, bssVal));
        }
        
        then12.add(OMPfuncIdent(flushFunc).Call());
        then12.add(Xcons.Set(varFlag, Xcons.FlogicalConstant(true)));

        then11.add(Bcons.Statement(OMPfuncIdent(threadLockFunc).Call()));
        then11.add(Bcons.IF(BasicBlock.Cond(Xcons.unaryOp(Xcode.LOG_NOT_EXPR, varFlag)), then12, null));
        then11.add(Bcons.Statement(OMPfuncIdent(threadUnlockFunc).Call()));
        
        bl.add(Bcons.IF(BasicBlock.Cond(
            Xcons.binaryOp(Xcode.LOG_AND_EXPR,
                Xcons.unaryOp(Xcode.LOG_NOT_EXPR, varFlag),
                Xcons.binaryOp(Xcode.LOG_GT_EXPR,
                    OMPfuncIdent(getNumThreadsFunc).Call(), Xcons.IntConstant(1)))),
            then11, null));
        
        BlockList then21 = Bcons.emptyBody();
        BlockList else21 = Bcons.emptyBody();
        
        Xobject naryDims = Xcons.IntConstant(idAry.Type().getNumDimensions());
        then21.add(Xcons.FpointerAssignment(varPrv, varShare));
        else21.add(idLoadArrayFunc.Call(Xcons.List(constVendor, varHdr, varAry, naryDims)));
        else21.add(Xcons.FpointerAssignment(varPrv,
            Xcons.FarrayRef(varAry, OMPfuncIdent(getThreadNumFunc).Call())));
        bl.add(Bcons.IF(BasicBlock.Cond(OMPfuncIdent(isMasterFunc).Call()), then21, else21));
        
        int norgDims = idAry.Type().getNumDimensions() - 1;
        
        if(norgDims > 0) {
            XobjList setBoundsArgs = Xcons.List(
                constVendor, varShare, varPrv);
            
            Xobject[] shareArySize = varShare.Type().getFarraySizeExpr();
            for(int ii = 0; ii < norgDims; ++ii) {
                setBoundsArgs.add(shareArySize[ii].getArg(0));
            }
            
            Ident idSetBoundsFunc = addFsetBoundsFuncInterface(
                orgBodyDecls, fdecls, norgDims, varPrv.Type(), varShare.Type());
            bl.add(Bcons.Statement(idSetBoundsFunc.Call(setBoundsArgs)));
        }
        
        mod.addFunctionBlock(fb);
        
        return fname;
    }

    /**
     * Fortran: add block for threadprivate variable setup
     */
    private void threadprivateFSetup(Xobject funcBlockDecls,
        Xobject orgBodyDecls, BlockList body, OMPinfo i, boolean inParallel)
    {
        if(i.getThdPrvVarList()==null || i.getThdPrvVarList().isEmpty())
            return;
        
        FmoduleBlock mod = getFinternalModule(orgBodyDecls);
        
        for(OMPvar v : i.getThdPrvVarList()) {
            Xobject varAry = v.getSharedArray();
            Xobject varPrv = v.getPrivateAddr();
            Xobject varShare = v.getSharedAddr();

            String midName = "";
            if(v.id.getStorageClass() == StorageClass.FCOMMON)
                midName =  v.id.getFcommonName();
            
            String str_fname_get = FgetThdprvFuncPrefix + midName + "_" + v.id.getName();
            Ident fname_get = null;
            
            if(mod.hasVar(varAry.getName())) {
                // already has get_thdprv function
                FunctionBlock fb_get = mod.getFunctionBlock(str_fname_get);
                fname_get = (Ident)fb_get.getNameObj();
            } else {
                fname_get = addFgetThreadPrivateFunc(orgBodyDecls, v, i, str_fname_get, midName);
            }
            
            XobjList decls = (XobjList)body.getDecls();
            
            if(decls == null) {
                decls = Xcons.List();
                body.setDecls(decls);
            }
            
            body.insert(fname_get.Call(Xcons.List((inParallel ? varShare : v.id), varPrv)));
        }
        
        funcBlockDecls.insert(Xcons.List(Xcode.F_USE_DECL, mod.getName()));
    }

    /**
     * add block for threadprivate variable setup
     */
    public void threadprivateSetup(Xobject funcBlockDecls, BlockList body, OMPinfo i, boolean inParallel)
    {
        if(OMP.leaveThreadPrivateFlag)
            return;
        if(i.getThdPrvVarList().isEmpty())
            return;
        
        if(XmOption.isLanguageC()) {
            threadprivateCSetup(body, i);
        } else {
            threadprivateFSetup(funcBlockDecls, funcBlockDecls, body, i, inParallel);
        }
    }
}
