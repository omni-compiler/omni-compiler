/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import exc.object.*;
import exc.xcodeml.XcodeMLtools;
import exc.block.*;

public class OMPtransPragmaBlock extends OMPtransPragma
{
    public boolean moveAutoFlag;

    public String bcastCopy = "ompc_bcast_copy";
    public String thdprvCopy = "ompc_bcast_thdprv";
    public String lastCopy = "ompc_last_copy";

    public OMPtransPragmaBlock()
    {
    }

    @Override
    public Block transBarrier(PragmaBlock b, OMPinfo i)
    {
        return OMPBlock.Barrier();
    }

    @Override
    public Block transParallelRegion(PragmaBlock b, OMPinfo i)
    {
        Xobject s;
        Block bp;

        // insert data setup
        BlockList body = Bcons.emptyBody(i.getIdList(), null);
        addDataSetupBlock(body, i);

        // caller side setup
        int nargs = i.getRegionArgs().size();
        BasicBlock caller_setup_bb = new BasicBlock();
        for(int ii = 0; ii < nargs; ii++) {
            s = i.getRegionArgs().get(ii);
            s = Xcons.List(Xcode.OMP_SETARG, Xcons.IntConstant(ii), s);
            caller_setup_bb.add(s);
        }

        // callee side, move blocks to body
        body.add(Bcons.COMPOUND(b.getBody())); // original body

        // insert setup argument
        BasicBlock bblock = new BasicBlock();
        for(int ii = 0; ii < i.getRegionParams().size(); ii++) {
            s = i.getRegionParams().get(ii).Ref();
            s = Xcons.Set(s, Xcons.List(Xcode.OMP_GETARG, s.Type(), Xcons.IntConstant(ii)));
            bblock.add(s);
        }
        body.insert(bblock);

        // add update block
        bp = dataUpdateBlock(i);
        if(bp != null)
            body.add(bp);

        // check threadprivate
        threadprivateSetup(null, body, i, true);

        return OMPBlock.ParallelRegion(nargs, caller_setup_bb, body, i.getIfExpr());
    }

    @Override
    public Block transFor(PragmaBlock b, OMPinfo i)
    {
        ForBlock for_block = (ForBlock)b.getBody().getHead();
        for_block.Canonicalize(); // canonicalize again
        Block bp;

        if(!for_block.isCanonical())
            OMP.fatal("transFor: not canonical");

        BlockList ret_body = Bcons.emptyBody(i.getIdList(), null);
        addDataSetupBlock(ret_body, i);

        BasicBlock bb = new BasicBlock();
        ret_body.add(bb);
        for(StatementIterator si = for_block.getInitBBlock().statements(); si.hasMoreStatement();) {
            Statement s = si.nextStatement();
            if(s.getNext() == null)
                break;
            s.remove();
            bb.add(s);
        }
        ret_body.add(OMPBlock.Forall(
            for_block.getInductionVar(),
            for_block.getLowerBound(),
            for_block.getUpperBound(),
            for_block.getStep(),
            for_block.getCheckOpcode(),
            for_block.getBody(),
            i.sched, i.sched_chunk, i.ordered));
        bp = dataUpdateBlock(i);
        if(bp != null)
            ret_body.add(bp);

        if(!i.no_wait)
            ret_body.add(OMPBlock.Barrier());
        return Bcons.COMPOUND(ret_body);
    }

    @Override
    public Block transSections(PragmaBlock b, OMPinfo i)
    {
        Block bp;
        BlockList ret_body = Bcons.emptyBody(i.getIdList(), null);
        int n = 0;

        addDataSetupBlock(ret_body, i);

        n = 0;
        for(bp = b.getBody().getHead(); bp != null; bp = bp.getNext())
            n++;

        ret_body.add(OMPBlock.Sections(n, b.getBody()));

        bp = dataUpdateBlock(i);
        if(bp != null)
            ret_body.add(bp);

        if(!i.no_wait)
            ret_body.add(OMPBlock.Barrier());
        return Bcons.COMPOUND(ret_body);
    }

    @Override
    public Block transSingle(PragmaBlock b, OMPinfo i)
    {
        BlockList body = Bcons.emptyBody(i.getIdList(), null);
        BlockList ret_body = Bcons.emptyBody();

        addDataSetupBlock(body, i);
        body.add(Bcons.COMPOUND(b.getBody()));
        ret_body.add(OMPBlock.Single(body));
        if(!i.no_wait)
            ret_body.add(OMPBlock.Barrier());
        return Bcons.COMPOUND(ret_body);
    }

    @Override
    public Block transMaster(PragmaBlock b, OMPinfo i)
    {
        return OMPBlock.Master(b.getBody());
    }

    @Override
    public Block transCritical(PragmaBlock b, OMPinfo i)
    {
        return OMPBlock.Critical(i.arg, b.getBody());
    }

    @Override
    public Block transAtomic(PragmaBlock b, OMPinfo i)
    {
        BlockList body = b.getBody();
        BasicBlock bb = body.getHead().getBasicBlock();
        Statement s = bb.getHead();
        Xobject x = s.getExpr(); // atomic update statment
        if(x.Opcode() != Xcode.POST_INCR_EXPR && x.Opcode() != Xcode.POST_DECR_EXPR &&
           x.Opcode() != Xcode.PRE_INCR_EXPR && x.Opcode() != Xcode.PRE_DECR_EXPR) {
            /* x op= rv */
            Xobject rv = x.right();
            Ident rv_var = Ident.Local(env.genSym("t"), rv.Type());
            body.addIdent(rv_var);
            s.insert(Xcons.Set(rv_var.Ref(), rv));
            x.setRight(rv_var.Ref());
        }
        Xobject lv = x.operand();
        Ident addr_var = Ident.Local(env.genSym("t"), Xtype.Pointer(lv.Type()));
        body.addIdent(addr_var);
        s.insert(Xcons.Set(addr_var.Ref(), Xcons.AddrOf(lv))); /* p = &lv */
        x.setOperand(Xcons.PointerRef(addr_var.Ref())); /* lv -> *p */
        s.insert(OMPfuncIdent(atomicLockFunc).Call(null));
        s.add(OMPfuncIdent(atomicUnlockFunc).Call(null));
        return Bcons.COMPOUND(body);
    }

    @Override
    public Block transFlush(PragmaBlock b, OMPinfo i)
    {
        return OMPBlock.Flush(i.flush_vars);
    }

    @Override
    public Block transOrdered(PragmaBlock b, OMPinfo i)
    {
        return OMPBlock.Ordered(b.getBody());
    }

    @Override
    public void addDataSetupBlock(BlockList bl, OMPinfo i)
    {
        BasicBlock bb = new BasicBlock();

        for(OMPvar v : i.getVarList()) {
            if(v.isVariableArray() && v.getPrivateAddr() != null)
                continue;
            if(v.is_first_private) {
                bb.add(Xcons.List(Xcode.OMP_BCAST,
                    v.getPrivateAddr(), v.getSharedAddr(), v.getSize()));
            }
        }

        for(OMPvar v : i.getVarList()) {

            if(v.isVariableArray() && v.getPrivateAddr() != null) {
                // variable length array
                Ident alloca_f = OMPfuncIdent(allocaFunc);
                bb.add(Xcons.Set(Xcons.PointerRef(v.getPrivateAddr()),
                    Xcons.Cast(v.getPrivateAddr().Type().getRef(),
                    alloca_f.Call(Xcons.List(v.getSize())))));
            }

            if(v.is_first_private)
                continue;

            if(v.is_reduction) {
                // only scalar variable
                /* issue OMP_SHARE to protect this variable from optimize */
                bb.add(Xcons.List(Xcode.OMP_SHARE, v.getPrivateAddr(), null));
                bb.add(Xcons.Set(Xcons.PointerRef(v.getPrivateAddr()), v.reductionInitValue()));
            } else if(v.is_copyin) {
                if(OMP.leaveThreadPrivateFlag) {
                    Ident id = v.id;
                    bb.add(Xcons.List(Xcode.OMP_BCAST_THDPRV, id.getAddr(), id.getAddr(),
                        Xcons.SizeOf(id.Type())));
                } else {
                    Ident thdprv_id = v.id;
                    Ident local_id = v.getThdPrvLocalId();
                    if(local_id == null)
                        OMP.fatal("bad copyin: " + thdprv_id.getName());
                    bb.add(Xcons.List(Xcode.OMP_BCAST_THDPRV, local_id.Ref(), thdprv_id.getAddr(),
                        Xcons.SizeOf(thdprv_id.Type())));
                }
            } else if(v.is_shared) {
                Ident id = v.id;
                bb.add(Xcons.List(Xcode.OMP_SHARE, v.getSharedAddr(), id));
            }
        }
        
        if(!bb.isEmpty())
            bl.add(bb);
    }

    @Override
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

        BasicBlock bb = new BasicBlock();
        if(reduction_flag) {
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
                bb.add(Xcons.List(Xcode.OMP_REDUCTION, v.getPrivateAddr(), v.getSharedAddr(),
                    Xcons.IntConstant(bt), Xcons.String(v.reduction_op.toString())));
            }
        }
        if(lastprivate_flag) {
            for(OMPvar v : i.getVarList()) {
                if(!v.is_last_private)
                    continue;
                bb.add(Xcons.List(Xcode.OMP_LAST_UPDATE,
                    v.getSharedAddr(), v.getPrivateAddr(), v.getSize()));
            }
        }
        return Bcons.BasicBlock(bb);
    }

    /*
     * transform OMPBlock
     */
    public void transBlock(FuncDefBlock def)
    {
        Block b;
        Block fblock = def.getBlock();
        current_def = def.getDef();
        env = current_def.getFile();

        if(OMP.debugFlag)
            System.out.println("pass3-transBlock:");

        cleanupOMPcode(fblock);

        BlockIterator i = new bottomupBlockIterator(fblock);
        for(i.init(); !i.end(); i.next()) {
            b = i.getBlock();
            switch(b.Opcode()) {
            case OMP_PARALLEL:
                i.setBlock(transParallelBlock((OMPparallelBlock)b));
                break;
            case OMP_FORALL:
                i.setBlock(transForallBlock((OMPforallBlock)b));
                break;
            case OMP_SECTIONS:
                i.setBlock(transSectionsBlock((OMPBlock)b));
                break;
            case OMP_CRITICAL:
                i.setBlock(transCriticalBlock((OMPBlock)b));
                break;
            case OMP_SINGLE:
                i.setBlock(transSingleBlock((OMPBlock)b));
                break;
            case OMP_MASTER:
                i.setBlock(transMasterBlock((OMPBlock)b));
                break;
            case OMP_ORDERED:
                i.setBlock(transOrderedBlock((OMPBlock)b));
                break;
            case OMP_ATOMIC:
                i.setBlock(transAtomicBlock((OMPBlock)b));
                break;
            case OMP_FLUSH:
                i.setBlock(transFlushBlock((OMPBlock)b));
                break;
            case OMP_BARRIER:
                i.setBlock(transBarrierBlock((OMPBlock)b));
                break;
            }
        }
    }

    void cleanupOMPcode(Block fblock)
    {
        for(BasicBlockExprIterator i = new BasicBlockExprIterator(fblock); !i.end(); i.next()) {
            Xobject x = i.getExpr();
            if(x == null)
                continue;
            switch(x.Opcode()) {
            case OMP_BCAST:
                i.setExpr(OMPfuncIdent(bcastCopy).Call(x));
                break;
            case OMP_BCAST_THDPRV:
                i.setExpr(OMPfuncIdent(thdprvCopy).Call(x));
                break;
            case OMP_SHARE: // ingore as default
                i.setExpr(null);
                break;
            case OMP_REDUCTION:
                i.setExpr(OMPfuncIdent(doReduction).Call(x));
                break;
            case OMP_LAST_UPDATE:
                i.setExpr(OMPfuncIdent(lastCopy).Call(x));
                break;
            }
        }
    }

    public Block transParallelBlock(OMPparallelBlock b)
    {
        Ident func_id = current_def.declStaticIdent(
            env.genExportSym(OMPfuncPrefix, current_def.getName()), Xtype
            .Function(Xtype.voidType));
        BlockList ret_body = Bcons.emptyBody();

        BasicBlock setup_bb = b.setupBasicBlock();
        BasicBlock end_bb = b.endBasicBlock();

        int nargs = b.getNarg();
        if(nargs != 0) {
            Xtype voidP_t = Xtype.Pointer(Xtype.voidType);
            Ident av = Ident.Local(ARGV_VAR, Xtype.Array(voidP_t, nargs));
            ret_body.addIdent(av);
            for(StatementIterator is = setup_bb.statements(); is.hasMoreStatement();) {
                Statement s = is.nextStatement();
                Xobject x = s.getExpr();
                if(x.Opcode() == Xcode.OMP_SETARG) {
                    Xobject av_idx = av.Index(x.getArg(0));
                    Xobject v = x.getArg(1);
                    
                    Xcode c = v.Opcode();
                    boolean isLocalVarAddr = v.isScopeLocal() &&
                        (c == Xcode.VAR_ADDR || c == Xcode.ARRAY_ADDR);
                    boolean isParamVarAddr = v.isScopeParam() &&
                        (c == Xcode.VAR_ADDR);
                    
                    if(OMP.moveAutoFlag &&
                        (isLocalVarAddr || isParamVarAddr)) {

                        Xobject size;
                        if(v.Type().isVariableArray()) {
                            size = v.Type().getArraySizeExpr().copy();
                        } else {
                            size = Xcons.SizeOf(v.Type());
                        }
                        Xobject vv = OMPfuncIdent(allocShared).Call(
                            Xcons.List(v, size));
                        end_bb.insert(OMPfuncIdent(freeShared).Call(
                            Xcons.List(av_idx, v, size)));
                        v = vv;
                    }
                    s.setExpr(Xcons.Set(av_idx, Xcons.Cast(voidP_t, v)));
                }
            }
            if(b.getIfExpr() != null)
                setup_bb.add(OMPfuncIdent(doParallelIfFunc).Call(
                    Xcons.List(b.getIfExpr(), func_id.Ref(),
                        av.Ref(), Xcons.IntConstant(nargs))));
            else
                setup_bb.add(OMPfuncIdent(doParallelFunc).Call(
                    Xcons.List(func_id.Ref(), av.Ref(),
                        Xcons.IntConstant(nargs))));
        } else {
            if(b.getIfExpr() != null)
                setup_bb.add(OMPfuncIdent(doParallelIfFunc).Call(
                    Xcons.List(b.getIfExpr(), func_id.Ref(), Xcons.IntConstant(0),
                        Xcons.IntConstant(0))));
            else
                setup_bb.add(OMPfuncIdent(doParallelFunc).Call(
                    Xcons.List(func_id.Ref(), Xcons.IntConstant(0),
                        Xcons.IntConstant(0))));
        }
        ret_body.add(setup_bb);
        ret_body.add(end_bb);

        // make parallel region fuction body
        BlockList body = b.getBody();
        Xobject param_id_list = Xcons.IDList();
        if(nargs != 0) {
            Xtype voidP_t = Xtype.Pointer(Xtype.voidType);
            Ident pv = Ident.Param(ARGS_VAR, Xtype.Pointer(voidP_t));
            param_id_list.add(pv);
            BasicBlock bb = body.getHead().getBasicBlock();
            for(StatementIterator is = bb.statements(); is.hasMoreStatement();) {
                Statement s = is.nextStatement();
                Xobject x = s.getExpr();
                if(x.isSet() && x.right().Opcode() == Xcode.OMP_GETARG) {
                    x.setRight(Xcons.Cast(x.left().Type(), pv.Index(x.right().operand())));
                }
            }
        }

        // make prototype for func_id
        ((FunctionType)func_id.Type()).setFuncParamIdList(param_id_list);

        FuncDefBlock para_fd = new FuncDefBlock(
            func_id, param_id_list, null, body, null, env);
        current_def.insertBeforeThis(para_fd.getDef());
        return Bcons.COMPOUND(ret_body);
    }

    public Block transForallBlock(OMPforallBlock b)
    {
        BlockList ret_body = Bcons.emptyBody();
        Xobject ind_var = b.getLoopVar();
        Ident lb_var = Ident.Local(env.genSym(ind_var.getName()), ind_var.Type());
        Ident ub_var = Ident.Local(env.genSym(ind_var.getName()), ind_var.Type());
        Ident step_var = Ident.Local(env.genSym(ind_var.getName()), ind_var.Type());
        ret_body.addIdent(lb_var);
        ret_body.addIdent(ub_var);
        ret_body.addIdent(step_var);

        BasicBlock bb = new BasicBlock();

        // move statement before iter_info into bb
        for(StatementIterator is = b.beginBasicBlock().statements(); is.hasMoreStatement();) {
            Statement s = is.nextStatement();
            s.remove();
            if(s == b.iter_info)
                break;
            bb.add(s);
        }

        bb.add(Xcons.Set(lb_var.Ref(), b.getLowerBound()));
        bb.add(Xcons.Set(ub_var.Ref(), b.getUpperBound()));
        bb.add(Xcons.Set(step_var.Ref(), b.getStep()));

        if(b.isOrdered()) {
            bb.add(OMPfuncIdent(orderedInitFunc).Call(Xcons.List(lb_var.Ref(), step_var.Ref())));
            b.getBody().insert(OMPfuncIdent(orderedSetLoopIdFunc).Call(Xcons.List(ind_var)));
        }

        Block loop_block = Bcons.FORall(
            ind_var, lb_var.Ref(), ub_var.Ref(), step_var.Ref(),
            b.getCheckOpcode(), b.getBody());

        /* add pre&end basic block if any */
        if(!b.beginBasicBlock().isEmpty() || !b.endBasicBlock().isEmpty()) {
            BlockList body = Bcons.emptyBody();
            body.add(b.beginBasicBlock());
            body.add(loop_block);
            body.add(b.endBasicBlock());
            loop_block = Bcons.COMPOUND(body);
        }

        Xobject sched_chunk = b.getChunk();
        switch(b.getSched()) {
        case SCHED_STATIC:
            if(sched_chunk != null) { /* same as SCHED_NONE */
                if(sched_chunk.Opcode() == Xcode.INT_CONSTANT && sched_chunk.getInt() == 1) {
                    /* cyclic schedule */
                    bb.add(OMPfuncIdent(cyclicShedFunc).Call(
                        Xcons.List(lb_var.getAddr(), ub_var.getAddr(), step_var.getAddr())));
                    break;
                }
                /* otherwise, block scheduling */
                bb.add(OMPfuncIdent(staticShedInitFunc).Call(
                    Xcons.List(lb_var.Ref(), ub_var.Ref(), step_var.Ref(), sched_chunk)));
                if(loop_block.Opcode() != Xcode.FOR_STATEMENT) {
                    BlockList body = Bcons.emptyBody();
                    body.add(loop_block.getBody().getHead().remove());
                    body.add(Bcons.WHILE(OMPfuncIdent(staticShedNextFunc).Call(
                        Xcons.List(lb_var.getAddr(), ub_var.getAddr())), loop_block.getBody()
                        .getHead().remove()));
                    body.add(loop_block.getBody().getHead().remove());
                    loop_block = Bcons.COMPOUND(body);
                } else
                    loop_block = Bcons.WHILE(OMPfuncIdent(staticShedNextFunc).Call(
                        Xcons.List(lb_var.getAddr(), ub_var.getAddr())), loop_block);
                break;
            }
            /* fall through */
        case SCHED_NONE:
            bb.add(OMPfuncIdent(defaultShedFunc).Call(
                Xcons.List(lb_var.getAddr(), ub_var.getAddr(), step_var.getAddr())));
            break;

        case SCHED_AFFINITY:
            Xobject arg = Xcons.List(lb_var.Ref(), ub_var.Ref(), step_var.Ref());
            arg.add(sched_chunk.getArg(0));
            arg.add(sched_chunk.getArg(1));
            arg.add(sched_chunk.getArg(2));
            arg.add(sched_chunk.getArg(3));
            bb.add(OMPfuncIdent(affinityShedInitFunc).Call(arg));
            loop_block = Bcons.WHILE(OMPfuncIdent(affinityShedNextFunc).Call(
                Xcons.List(lb_var.getAddr(), ub_var.getAddr())), loop_block);
            break;

        case SCHED_DYNAMIC:
            if(sched_chunk == null)
                sched_chunk = Xcons.IntConstant(1);
            bb.add(OMPfuncIdent(dynamicShedInitFunc).Call(
                Xcons.List(lb_var.Ref(), ub_var.Ref(), step_var.Ref(), sched_chunk)));
            loop_block = Bcons.WHILE(OMPfuncIdent(dynamicShedNextFunc).Call(
                Xcons.List(lb_var.getAddr(), ub_var.getAddr())), loop_block);
            break;
        case SCHED_GUIDED:
            if(sched_chunk == null)
                sched_chunk = Xcons.IntConstant(1);
            bb.add(OMPfuncIdent(guidedShedInitFunc).Call(
                Xcons.List(lb_var.Ref(), ub_var.Ref(), step_var.Ref(), sched_chunk)));
            loop_block = Bcons.WHILE(OMPfuncIdent(guidedShedNextFunc).Call(
                Xcons.List(lb_var.getAddr(), ub_var.getAddr())), loop_block);
            break;
        case SCHED_RUNTIME:
            bb.add(OMPfuncIdent(runtimeShedInitFunc).Call(
                Xcons.List(lb_var.Ref(), ub_var.Ref(), step_var.Ref())));
            loop_block = Bcons.WHILE(OMPfuncIdent(runtimeShedNextFunc).Call(
                Xcons.List(lb_var.getAddr(), ub_var.getAddr())), loop_block);
            break;
        default:
            OMP.fatal("unknown schedule: " + b.getSched());
        }

        ret_body.add(bb);
        ret_body.add(loop_block);
        return Bcons.COMPOUND(ret_body);
    }

    public Block transSectionsBlock(OMPBlock b)
    {
        Block bp;
        Xobject label = Xcons.Symbol(Xcode.IDENT, env.genSym("L"));
        BlockList ret_body = Bcons.emptyBody();
        BasicBlock bb = b.beginBasicBlock();
        bb.add(OMPfuncIdent(sectionInitFunc).Call(Xcons.List(Xcons.IntConstant(b.getNarg()))));
        ret_body.add(bb);
        ret_body.add(Bcons.LABEL(label));

        BlockList body = Bcons.emptyBody();
        int n = 0;
        while((bp = b.getBody().getHead()) != null) {
            bp.remove();
            body.add(Bcons.CASE(Xcons.IntConstant(n++)));
            body.add(bp);
            body.add(Bcons.GOTO(label));
        }
        ret_body.add(Bcons.SWITCH(OMPfuncIdent(sectionIdFunc).Call(null), Bcons.COMPOUND(body)));
        return Bcons.COMPOUND(ret_body);
    }

    public Block transCriticalBlock(OMPBlock b)
    {
        String critical_lock_name = criticalLockPrefix;
        if(b.getInfoExpr() != null)
            critical_lock_name += "_" + b.getInfoExpr().getName();
        Ident lock_id = current_def.declGlobalIdent(critical_lock_name, Xtype.Pointer(Xtype.voidType));
        BlockList body = b.getBody();
        body.insert(OMPfuncIdent(enterCriticalFunc).Call(Xcons.List(lock_id.getAddr())));
        body.add(OMPfuncIdent(exitCriticalFunc).Call(Xcons.List(lock_id.getAddr())));
        return Bcons.COMPOUND(body);
    }

    public Block transSingleBlock(OMPBlock b)
    {
        BlockList body = Bcons.emptyBody();
        body.add(b.beginBasicBlock());
        body.add(Bcons
            .IF(BasicBlock.Cond(OMPfuncIdent(doSingleFunc).Call(null)), b.getBody(), null));
        body.add(b.endBasicBlock());
        return Bcons.COMPOUND(body);
    }

    public Block transMasterBlock(OMPBlock b)
    {
        BlockList body = Bcons.emptyBody();
        body.add(b.beginBasicBlock());
        body.add(Bcons
            .IF(BasicBlock.Cond(OMPfuncIdent(isMasterFunc).Call(null)), b.getBody(), null));
        body.add(b.endBasicBlock());
        return Bcons.COMPOUND(body);
    }

    public Block transOrderedBlock(OMPBlock b)
    {
        BlockList body = b.getBody();
        body.insert(OMPfuncIdent(orderedBeginFunc).Call(null));
        body.insert(b.beginBasicBlock());
        body.add(OMPfuncIdent(orderedEndFunc).Call(null));
        body.add(b.endBasicBlock());
        return Bcons.COMPOUND(body);
    }

    public Block transAtomicBlock(OMPBlock b)
    {
        BlockList body = b.getBody();
        body.insert(OMPfuncIdent(atomicLockFunc).Call(null));
        body.insert(b.beginBasicBlock());
        body.add(OMPfuncIdent(atomicUnlockFunc).Call(null));
        body.add(b.endBasicBlock());
        return Bcons.COMPOUND(body);
    }

    public Block transFlushBlock(OMPBlock b)
    {
        BlockList body = Bcons.emptyBody();
        body.add(b.beginBasicBlock());
        Xobject flush_vars = b.getInfoExpr();
        if(flush_vars == null) {
            body.add(Bcons.Statement(OMPfuncIdent(flushFunc).Call(
                Xcons.List(Xcons.IntConstant(0), Xcons.IntConstant(0)))));
        } else {
            BasicBlock bb = new BasicBlock();
            for(XobjArgs a = flush_vars.getArgs(); a != null; a = a.nextArgs()) {
                Xobject v = a.getArg();
                bb.add(OMPfuncIdent(flushFunc).Call(v));
            }
            body.add(bb);
        }
        body.add(b.endBasicBlock());
        return Bcons.COMPOUND(body);
    }

    public Block transBarrierBlock(OMPBlock b)
    {
        BlockList body = Bcons.emptyBody();
        body.add(b.beginBasicBlock());
        body.add(OMPfuncIdent(barrierFunc).Call(null));
        body.add(b.endBasicBlock());
        return Bcons.COMPOUND(body);
    }
}
