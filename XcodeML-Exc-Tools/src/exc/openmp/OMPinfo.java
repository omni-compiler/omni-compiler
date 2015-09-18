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

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import xcodeml.util.XmOption;

/**
 * information for each OMP directive
 */
public class OMPinfo
{
    private OMPinfo parent;
    private Block block; /* back link */
    OMPpragma pragma; /* directives */
    Xobject arg;
    private List<OMPvar> varlist; /* variable list */
    private XobjList id_list;
    private List<OMPvar> thdprv_varlist; /* threadprivate variable list */
    private XcodeMLtools tools;
    // for OMPpragma.FLUSH
    List<Xobject> flush_vars;

    // for OMPpragma.FOR
    OMPpragma sched = OMPpragma.SCHED_NONE; /* for DIR_SCHEDULE */
    Xobject sched_chunk;
    boolean ordered;
    boolean untied;
    boolean mergeable;

    boolean simd;
    // for parallel region, OMPpragma.PARALLEL
    OMPpragma data_default = OMPpragma._DEFAULT_NOT_SET;
    Xobject num_threads;
    boolean no_wait;
    private Xobject if_expr,final_expr;
    private List<Xobject> region_args; // for parallel region
    private List<Ident> region_params;
    private Xobject thdnum_var;

    // for fuction body, OMPpragma.FUNCTION_BODY, OMPpragma.PARALLEL
    OMPfileEnv env;
    
    public static final String PROP_KEY_THDPRV = "OMPInfo.threadprivate";

    public OMPinfo(OMPpragma pragma, OMPinfo parent, Block b, OMPfileEnv env)
    {
        this.pragma = pragma;
        this.parent = parent;
        this.block = b;
        this.env = env;
        id_list = Xcons.IDList();
        varlist = new LinkedList<OMPvar>();
        region_args = new ArrayList<Xobject>();
        region_params = new ArrayList<Ident>();

        OMP.debug("OMPinfo");

        if(pragma == OMPpragma.PARALLEL ||pragma == OMPpragma.TASK) {
            /* check local tagnames, replicate them. */
            for(Block bb = b.getParentBlock(); bb != null; bb = bb.getParentBlock()) {
                BlockList b_list = bb.getBody();
                if(b_list == null)
                    continue;
                Xobject ids = b_list.getIdentList();
                if(ids == null)
                    continue;
                for(XobjArgs a = ids.getArgs(); a != null; a = a.nextArgs()) {
                    Ident id = (Ident)a.getArg();
                    /* check already declared or not. */
                    if(id.getStorageClass() == StorageClass.TAGNAME) {
                        for(XobjArgs aa = id_list.getArgs(); aa != null; aa = aa.nextArgs()) {
                            if(((Ident)aa.getArg()).getName().equals(id.getName())) {
                                id = null;
                                break;
                            }
                        }
                        if(id != null)
                            id_list.add(id.copy());
                    }
                }
            }
        }

        if(pragma == OMPpragma.PARALLEL || pragma == OMPpragma.FUNCTION_BODY ||pragma == OMPpragma.TASK) {
            thdprv_varlist = new ArrayList<OMPvar>();
        }
    }
    
    public Block getBlock()
    {
        OMP.debug("getBlock");
        return block;
    }
    
    List<OMPvar> getVarList()
    {
        OMP.debug("getVarList");
        return varlist;
    }
    
    private void addVar(OMPvar var)
    {
        OMP.debug("addVar");
        varlist.add(var);
    }

    OMPvar findOMPvar(String name)
    {
        OMP.debug("findOMPVar");
        for(OMPvar v : varlist)
            if(v.id.getName().equals(name))
                return v;
        return null;
    }

    OMPvar findOMPvar(Ident id)
    {
        OMP.debug("findOMPVar");
        for(OMPvar v : varlist)
            if(v.id == id)
                return v;
        
        OMPvar v = (OMPvar)id.getProp(PROP_KEY_THDPRV);
        
        return v;
    }

    // for checking implicit private in FOR directive
    boolean isPrivateOMPvar(String name)
    {
        OMP.debug("isPrivateOMPVar");
        OMPvar v = null;
        Ident id = null;
        for(Block b = block; b != null; b = b.getParentBlock()) {
            BlockList b_list = b.getBody();
            if(b_list == null)
                continue;
            // check local declaration
            if((id = b_list.findLocalIdent(name)) != null)
                break;
            OMPinfo i = (OMPinfo)b.getProp(OMP.prop);
            if(i == null)
                continue;
            // if declared as OMPvar, check it
            if((v = i.findOMPvar(name)) != null)
                break;
            if(i.pragma == OMPpragma.PARALLEL || i.pragma==OMPpragma.TASK)
                break; // not beyond parallel region
        }
        if(id != null) {
            switch(id.getStorageClass()) {
            case PARAM:
            case FPARAM:
            case AUTO:
            case REGISTER:
                return true;
            }
        }
        if(v != null && v.is_private)
            return true;
        else
            return false;
    }

    //
    // build OMP variable symbol table
    //
    OMPvar declOMPvar(String name, OMPpragma atr)
    {
        OMP.debug("declOMPVar");
        if(atr == null)
            return null;
        
        OMPvar v = null;
        Ident id = null;

        if(OMP.debugFlag)
            System.out.println("declOMPvar name=" + name + ", atr=" + atr);

        // check special case: firstprivate and lastprivate
        if((v = this.findOMPvar(name)) != null) {
            if((v.atr == OMPpragma.DATA_FIRSTPRIVATE && atr == OMPpragma.DATA_COPYPRIVATE) ||
                (atr == OMPpragma.DATA_FIRSTPRIVATE && v.atr == OMPpragma.DATA_COPYPRIVATE)) {
                OMP.error(block.getLineNo(),
                    "copyprivate variable cannot be firstprivate");
                return null;
            }
            
            if((v.atr == OMPpragma.DATA_FIRSTPRIVATE && atr == OMPpragma.DATA_LASTPRIVATE)
                || (atr == OMPpragma.DATA_FIRSTPRIVATE && v.atr == OMPpragma.DATA_LASTPRIVATE)) {
                v.is_first_private = true;
                v.is_last_private = true;
                return v;
            }
        }

        // find local data scope attribute
        for(Block b = block; b != null; b = b.getParentBlock()) {
            BlockList b_list = b.getBody();
            if(b_list == null)
                continue;
            // check local declaration
            if((id = b_list.findLocalIdent(name)) != null)
                break;
            OMPinfo i = (OMPinfo)b.getProp(OMP.prop);
            if(i == null)
                continue;
            // if declared as OMPvar, check it
            if((v = i.findOMPvar(name)) != null)
                break;
            if(i.pragma == OMPpragma.PARALLEL||i.pragma == OMPpragma.TASK)
                break; // not beyond parallel region
        }
        if(id != null) {
            switch(id.getStorageClass()) {
            case PARAM:
            case FPARAM:
            case AUTO:
            case REGISTER:
                if(atr != OMPpragma.DATA_PRIVATE)
                    OMP.warning(block.getLineNo(), "local variable '" + name
                        + "' is already private");
                return null;
            }
        }
        if(v != null) {
            if(v.is_private) {
                if(atr == OMPpragma.DATA_COPYPRIVATE) {
                    v.is_copy_private = true;
                    v.setCopyPrivateAddr(createCopyPrivateAddr(v));
                } else {
                    OMP.warning(block.getLineNo(), "variable '" + name
                        + "' is already defined as private");
                }
                return v;
            }
            // already declared as shared
            if(v.is_shared && atr == OMPpragma.DATA_SHARED) {
                return v; // nested shared ????
            }
            // else {
            // OMP.error(block.getLineNo(),
            // "variable '"+name+"' is already defined");
            // return null;
            // }
        }

        id = block.findVarIdent(name);
        if(id == null) {
            OMP.error(block.getLineNo(), "undefined variable in clause, '" + name + "'");
            return null;
        }
        Xtype t = id.Type();
        if(t.isFunction()) {
            OMP.error(block.getLineNo(), "data '" + name + "' in cluase is a function");
            return null;
        }
        if(atr.isDataReduction()) {
            if((XmOption.isLanguageC()
                    && !t.isNumeric()) ||
                (XmOption.isLanguageF() && (
                    (!t.isFarray() && !t.isNumeric()) ||
                    (t.isFarray() && !t.getRef().isNumeric())))) {
                OMP.error(block.getLineNo(), "non-scalar variable '" + name
                    + "' in reduction clause");
                return null;
            }
        }
	if(id.getStorageClass()!=null)
        if(env.isThreadPrivate(id)) {
            if(atr != OMPpragma.DATA_COPYIN) {
                OMP.error(block.getLineNo(), "variable '" + name + "' is threadprivate");
                return null;
            }
            // reference threadprivate variable
            OMPvar thdprv_var = refOMPvar(block, id.getName());
            if(thdprv_var != null)
                v = declCopyinThreadPrivate(thdprv_var);
            return v;
        } else {
            // not threadprivate
            if(atr == OMPpragma.DATA_COPYIN) {
                OMP.error(block.getLineNo(), "non-threadprivate'" + name + "' in COPYIN clause");
                return null;
            }
        }

        v = new OMPvar(id, atr);
        
        if(t.isVariableArray()) {
            v.setSizeOfArray(Xcons.binaryOp(Xcode.MUL_EXPR,
                refOMPvarExpr(block, t.getArraySizeExpr().copy()),
                Xcons.SizeOf(t.getRef())));
        }

        if(v.is_private) {
            v.setPrivateAddr(createPrivateAddr(v));
        }

        if(v.is_copy_private) {
            v.setCopyPrivateAddr(createCopyPrivateAddr(v));
        }

        if(v.is_shared) {
            // need to be shared. declared in parallel region
            v.setSharedAddr(createSharedAddr(id));
        }
        
        if(v.is_reduction) {
            v.setSharedArray(createSharedArray(v, 0, true));
        }
        
        addVar(v);
        return v;
    }
    
    private Xobject createPrivateAddr(OMPvar v)
    {
        OMP.debug("createPrivateAddr");
        Ident id = v.id;
        Xtype t = id.Type();
        
        if(v.isVariableArray()) {
            /* for variable length array, convert local copy into pointer */
            t = Xtype.Pointer(id.Type().getRef());
        } else {
            t = t.copy();
            if(XmOption.isLanguageC()) {
                t.setIsConst(false);
            } else {
                t.unsetIsFsave();
                t.setIsFintentIN(false);
                t.setIsFintentOUT(false);
                t.setIsFintentINOUT(false);
                if(v.is_reduction ||
                    (v.is_shared && !v.is_first_private && !v.is_last_private && !v.is_private)) {
                    t.unsetIsFtarget();
                    t.setIsFpointer(true);
                    if(t.isFarray())
                        t.convertToAssumedShape();
                }
            }
        }
        
        Ident local_id = Ident.Local(OMPtransPragma.LOCAL_VAR_PREFIX + id.getName(), t);
        id_list.add(local_id);
        
        return id.Type().isArray() ? local_id.Ref() : local_id.getAddr();
    }
    
    private Xobject createCopyPrivateAddr(OMPvar v)
    {
        OMP.debug("createCopyPrivateAddr");
        Ident id = v.id;
        Xtype t = id.Type();
        
        if(v.isVariableArray()) {
            /* for variable length array, convert local copy into pointer */
            t = Xtype.Pointer(id.Type().getRef());
        } else {
            t = t.copy();
            t.setIsConst(false);
        }
        
        Ident local_id = Ident.Local(OMPtransPragma.COPYPRV_VAR_PREFIX + id.getName(), t);
        id_list.add(local_id);
        
        return id.Type().isArray() ? local_id.Ref() : local_id.getAddr();
    }
    
    private Xobject createSharedAddr(Ident id)
    {
        OMP.debug("createSharedAddr");
        Block b;
        BlockList b_list;
        OMPinfo i;
        OMPvar v;

        // check private declaration for global.
        StorageClass stg = id.getStorageClass();
        
        if(stg.isBSS()) {
            // search local declaration for global
            for(b = block; b != null; b = b.getParentBlock()) {
                // if declared in local context.
                if(stg != StorageClass.FCOMMON && (b_list = b.getBody()) != null
                    && b_list.findLocalIdent(id) != null)
                    break;

                // if declared as OMPvar, return it address (fall through)
                if((i = (OMPinfo)b.getProp(OMP.prop)) != null && (v = i.findOMPvar(id)) != null
                    && v.is_private)
                    break;
            }
            if(b == null)
                return id.Type().isArray() ? id.Ref() : id.getAddr();
        }

        // search parallel section to be declared.
        for(b = block; b != null; b = b.getParentBlock()) {
            // if declared as local variable, reference it.
            if((b_list = b.getBody()) != null && b_list.findLocalIdent(id) != null)
                return id.Type().isArray() ? id.Ref() : id.getAddr();

            if((i = (OMPinfo)b.getProp(OMP.prop)) == null)
                continue;

            // if declared as OMPvar, return it address
            if((v = i.findOMPvar(id.getName())) != null)
                return v.getAddr();

            if(i.pragma == OMPpragma.PARALLEL || i.pragma == OMPpragma.TASK) {
                return addRegionVar(b, i, id);
            }
        }
        OMP.fatal("sharedAddr: id is not found, id =" + id);
        return null;
    }
    
    private Xobject createSharedArray(OMPvar v, int sizeOffset, boolean isRedction)
    {
        OMP.debug("createSharedArray");
        if(!XmOption.isLanguageF())
            return null;
        
        Ident id = v.id;
        Xtype t = id.Type();
        
        Xobject[] sizeExprs = new Xobject[t.getNumDimensions() + 1];
        for(int i = 0; i < sizeExprs.length; ++i) {
            sizeExprs[i] = Xcons.FindexRangeOfAssumedShape();
        }
        t = Xtype.Farray(t.isFarray() ? t.getRef() : t, sizeExprs);
        if(isRedction) {
            t.setIsFpointer(false);
            t.setIsFallocatable(true);
            t.setIsFtarget(true);
        } else {
            t.setIsFpointer(true);
            t.setIsFallocatable(false);
            t.unsetIsFtarget();
        }
        t.setIsFsave(true);
        Ident local_id = Ident.Fident(OMPtransPragma.THDPRV_STORE_PREFIX + id.getName(), t);
        local_id.getAddr().setIsDelayedDecl(false);
        
        return local_id.getAddr();
    }
    
    private Xobject addRegionVar(Block b, OMPinfo i, Ident id)
    {
        OMP.debug("addRegionVar");
        // make local variable to pass across parallel section.
        if(XmOption.isLanguageF() && id.Type() != null && id.Type().isFparameter())
            return null;
        
        // get reference outside block.
        String local_name = OMPtransPragma.SHARE_VAR_PREFIX + id.getName();
        OMP.debug(local_name);
        if(i.id_list.hasIdent(local_name))
            return i.id_list.getIdent(local_name).Ref();
        
        Xobject addr;
        OMPvar v;
        if((v = refOMPvar(b.getParentBlock(), id.getName())) != null)
            addr = v.getAddr();
        else
            addr = id.getAddr();
        if(addr == null)
            throw new NullPointerException("addr");
        i.addRegionArg(addr);
        
        Xtype t, torg = id.Type();
        
        if(torg.isArray()) {
            //convert array to pointer
            //  ( int [n] -> int * )
            //  ( int (*)[n] -> int ** )
            t = Xtype.Pointer(torg.getRef());
        } else {
            if(XmOption.isLanguageC())
                t = Xtype.Pointer(torg);
            else {
                if(torg.isStruct()) {
                    t = torg;
                } else {
                    t = torg.copy();
//                    t.setIsFtarget(true);
                    t.unsetIsFtarget();
                    t.setIsFparameter(false);
                    t.unsetIsFsave();
                    if(t.isFarray()) {
                        t.convertFindexRange(true, true, block);
                    }
                }
            }
        }
        
        Ident id_local = Ident.Local(local_name, t);

        // check already defined or not.
        for(int k = 0; k < i.region_params.size(); k++) {
            if(((i.region_params.get(k))).getName().equals(id_local.getName()))
                return id_local.Ref();
        }

        i.id_list.add(id_local);
        i.addRegionParam(id_local);
        
        
        return id_local.Ref();
    }
    
    private void addRegionVarInExpr(Xobject x, Block b, OMPinfo i)
    {
        OMP.debug("addRegionVarInExpr");
        if(x == null)
            return;
        topdownXobjectIterator ite = new topdownXobjectIterator(x);
        for(ite.init(); !ite.end(); ite.next()) {
            Xobject a = ite.getXobject();
            if(a != null && a.isVarRef()) {
                Ident ai = i.getIdList().findVarIdent(a.getName());
                if(ai != null)
                    continue;
                ai = block.findVarIdent(a.getName());
                if(ai == null) {
                    throw new NullPointerException("not found ident in array size : " + a.getName());
                }
                
                OMPvar v = findOMPvar(ai);
                
                if(v != null)
                    continue;

                refOMPvar(b, a.getName(), OMPpragma.DATA_SHARED);
            }
        }
    }
    
    void declOMPVarsInType(XobjList names)
    {
        OMP.debug("addRegionVarInType");
        OMPinfo i = null;
        Block b = null;

        for(b = block; b != null; b = b.getParentBlock()) {
            // if declared as local variable, reference it.
            i = (OMPinfo)b.getProp(OMP.prop);
            if(i != null)
                break;
        }
        
        if(i == null)
            return;

        for(Xobject a : names) {
            OMPvar v = findOMPvar(a.getName());
            if(v == null)
                return;

            Xtype t = v.id.Type();
        
            switch(t.getKind()) {
            case Xtype.BASIC:
                if(XmOption.isLanguageF())
                    addRegionVarInExpr(t.getFlen(), b, i);
                break;
            case Xtype.ARRAY:
                addRegionVarInExpr(t.getArraySizeExpr(), b, i);
                break;
            case Xtype.F_ARRAY:
                for(Xobject x : t.getFarraySizeExpr())
                    addRegionVarInExpr(x, b, i);
                break;
            }
        }
    }

    OMPvar declThreadPrivate(Ident id)
    {
        OMP.debug("declThreadPrivate");
        for(OMPvar v : varlist)
            if(v.id == id)
                return v;

        OMPvar v = new OMPvar(id, OMPpragma.DATA_NONE);
        id.setProp(PROP_KEY_THDPRV, v);
        
        if(OMP.leaveThreadPrivateFlag) {
            v.setSharedAddr(id.getAddr());
            addVar(v);
            return v;
        }
        
        Xtype pt;
        if(XmOption.isLanguageC()) {
            pt = Xtype.Pointer(id.Type());
        } else {
            pt = id.Type().copy();
            pt.unsetIsFsave();
            pt.unsetIsFtarget();
            pt.setIsFpointer(true);
            if(pt.isFarray())
                pt.convertToAssumedShape();
        }
        
        Ident id_local = Ident.Local(OMPtransPragma.THDPRV_LOCAL_VAR_PREFIX + id.getName(), pt);
        v.setThdPrvLocalId(id_local);
        Xobject localAddr = id.Type().isArray() ? Xcons.PointerRef(id_local.Ref()) : id_local.Ref();
        if(v.id.isCglobalVarOrFvar()) {
            if(XmOption.isLanguageF()) {
                Xtype t = id.Type().copy();
                t.unsetIsFsave();
                if(t.isFarray())
                    t.convertFindexRange(true, false, block);
                Ident id_share = Ident.Fident(OMPtransPragma.SHARE_VAR_PREFIX + id.getName(), t);
                v.setPrivateAddr(localAddr);
                v.setSharedAddr(id_share);
                v.setSharedArray(createSharedArray(v, -1, false));
                addRegionArg(v.id);
                addRegionParam(id_share);
            } else {
                v.setSharedAddr(localAddr);
            }
        } else {
            v.setPrivateAddr(localAddr);
            v.setSharedAddr(createSharedAddr(id));
        }
        addThdPrvVar(v);
        addVar(v);
        return v;
    }

    OMPvar declCopyinThreadPrivate(OMPvar thdprv_var)
    {
        OMP.debug("declCopyinThreadPrivate");
        OMPvar v = new OMPvar(thdprv_var.id, OMPpragma.DATA_COPYIN);
        v.setPrivateAddr(thdprv_var.getPrivateAddr());
        v.setSharedAddr(thdprv_var.getSharedAddr());
        v.setSharedArray(thdprv_var.getSharedArray());
        v.setThdPrvLocalId(thdprv_var.getThdPrvLocalId());
        addVar(v);
        boolean exists = false;
        for(OMPvar vv : getThdPrvVarList()) {
            if(vv.id == v.id) {
                exists = true;
                break;
            }
        }
        if(!exists)
            addThdPrvVar(v);
        return v;
    }
    
    static OMPvar findOMPvarRecurse(Block b, String name)
    {
        OMP.debug("findOMPvarRecurse");

        OMPinfo i;
        OMPvar v;
        
        for(; b != null; b = b.getParentBlock()) {
            BlockList b_list = b.getBody();
            if(b_list != null && b_list.findLocalIdent(name) != null)
                return null; // defined local variable inside.
            if((i = (OMPinfo)b.getProp(OMP.prop)) != null) {
                if((v = i.findOMPvar(name)) != null)
                    return v;
            }
        }
        
        return null;
    }
    
    // reference OMPvar in 'block'
    static OMPvar refOMPvar(Block b, String name)
    {
        OMP.debug("refOMPvar"+name);
        return refOMPvar(b, name, null);
    }
    
    private static OMPvar refOMPvar(Block b, String name, OMPpragma override_default)
    {
        OMP.debug("refOMPvar");
        OMPinfo i;
        OMPvar v;
        Ident id;
        for(; b != null; b = b.getParentBlock()) {
            BlockList b_list = b.getBody();
            if(b_list != null && b_list.findLocalIdent(name) != null)
                return null; // defined local variable inside.
            if((i = (OMPinfo)b.getProp(OMP.prop)) == null)
                continue;
            if((v = i.findOMPvar(name)) != null)
                return v;
                
            switch(i.pragma) {
            case PARALLEL:
            case TASK:
                /* not found, then declare variable as default attribute */
                if((id = i.env.findThreadPrivate(b, name)) != null)
                    return i.declThreadPrivate(id);

                id = b.findVarIdent(name);

                if(id == null)
                    return null; // genrated variable???
                
                if(id.Type() != null && id.Type().isFparameter())
                    return null;
                
                OMPpragma scope = null;

                if(i.data_default == OMPpragma._DEFAULT_NOT_SET && id.isInductionVar())
                    scope = OMPpragma.DATA_PRIVATE;
                else if(i.data_default == OMPpragma.DEFAULT_SHARED ||
                    i.data_default == OMPpragma._DEFAULT_NOT_SET)
                    scope = OMPpragma.DATA_SHARED;
                else if(i.data_default == OMPpragma.DEFAULT_PRIVATE) {
                    if(id.getStorageClass().isFuncParam())
                        scope = OMPpragma._DATA_PRIVATE_SHARED;
                    else
                        scope = OMPpragma.DATA_PRIVATE;
                } else if(override_default != null){
                    scope = override_default;
                } else {
                    OMP.error(b.getLineNo(), "unknown data attribute on '" + name
                        + "', default data attribute is 'none'");
                }
                return i.declOMPvar(name, scope);
                
            case FUNCTION_BODY:
                if((id = i.env.findThreadPrivate(b, name)) != null)
                    return i.declThreadPrivate(id);
                break;
            }
        }
        /* reference from orphan-directives */
        return null;
    }

    static OMPvar findOMPvarBySharedOrPrivate(Block b, String name)
    {
        OMP.debug("findOMPvarBySharedOrPrivate");
        if(name.startsWith(OMPtransPragma.LOCAL_VAR_PREFIX)) {
            return findOMPvarRecurse(b, name.substring(OMPtransPragma.LOCAL_VAR_PREFIX.length()));
        } else if(name.startsWith(OMPtransPragma.SHARE_VAR_PREFIX)) {
            return findOMPvarRecurse(b, name.substring(OMPtransPragma.SHARE_VAR_PREFIX.length()));
        }
        return null;
    }
    
    static Xobject refOMPvarExpr(Block b, Xobject x)
    {
        OMP.debug("refOMPvarExpr");
        OMPvar v;
        if(x == null)
            return null;
        XobjectIterator i = new bottomupXobjectIterator(x);
        for(i.init(); !i.end(); i.next()) {
            Xobject xx = i.getXobject();
            if(xx == null)
                continue;
            if(xx.isVariable()) {
                if((v = refOMPvar(b, xx.getName())) == null)
                    continue;
                i.setXobject(v.Ref());
            } else if(xx.isVarAddr() || xx.isArray() || xx.isArrayAddr()) {
                if((v = refOMPvar(b, xx.getName())) == null)
                    continue;
                if(v != null)
                    i.setXobject(v.getAddr());
            }
        }
        return i.topXobject();
    }

    void setSchedule(Xobject sched_arg)
    {
        OMP.debug("setSchedule");
        if(sched_arg == null)
            return; // for error recovery

        if(sched != OMPpragma.SCHED_NONE) {
            OMP.error(block.getLineNo(), "schedule clause is already specified");
            return;
        }
        sched = OMPpragma.valueOf(sched_arg.getArg(0));
        sched_chunk = sched_arg.getArgOrNull(1);
        if(sched == OMPpragma.SCHED_RUNTIME && sched_chunk != null)
            OMP.error(block.getLineNo(), "schedule RUNTIME has chunk argument");
        if(sched == OMPpragma.SCHED_AFFINITY)
            OMP.error(block.getLineNo(), "schedule AFFINITY is not supported in this mode");
        sched_chunk = refOMPvarExpr(block.getParentBlock(), sched_chunk);
        if(OMP.debugFlag)
            System.out.println("schedule(" + sched + "," + sched_chunk + ")");
    }

    void setIfExpr(Xobject cond)
    {
        OMP.debug("setIfExpr");
        if_expr = refOMPvarExpr(block.getParentBlock(), cond);
        if(XmOption.isLanguageC() && !if_expr.Type().isIntegral())
            if_expr = Xcons.Cast(Xtype.intType, if_expr);
        if(OMP.debugFlag)
            System.out.println("parallel if = " + if_expr);
    }

    void setFinalExpr(Xobject cond)
    {
        OMP.debug("setFinalExpr");
        final_expr = refOMPvarExpr(block.getParentBlock(), cond);
        if(XmOption.isLanguageC() && !final_expr.Type().isIntegral())
            final_expr = Xcons.Cast(Xtype.intType, final_expr);
        if(OMP.debugFlag)
            System.out.println("task final = " + final_expr);
    }

    void setFlushVars(Xobject list)
    {
        OMP.debug("setFlushVars");
        // list is list of IDENT
        if(list == null)
            return;
        flush_vars = new ArrayList<Xobject>();
        for(Xobject x  : (XobjList)list) {
            OMPvar v = refOMPvar(block, x.getName());
            if(v != null) {
                if(v.is_private) {
                    OMP.error(block.getLineNo(), "private variable in flush, '"
                        + x.getName() + "'");
                    continue;
                }
                if(!v.is_shared)
                    OMP.fatal("not shared variable in flush '" + x.getName() + "'");
                if(XmOption.isLanguageC()) {
                    flush_vars.add(Xcons.List(v.getSharedAddr(), v.getSize()));
                }
            } else { /* local or global from orphan */
                Ident id = block.findVarIdent(x.getName());
                if(id == null) {
                    OMP.error(block.getLineNo(), "undefined variable '"
                        + x.getName() + "' in flush");
                    continue;
                }
                
                switch(id.getStorageClass()) {
                case EXTERN:
                case EXTDEF:
                case STATIC:
                case FCOMMON:
                case FPARAM:
                case FSAVE:
                    break;
                default:
                    OMP.error(block.getLineNo(), "local variable '" + x.getName() + "' in flush");
                    continue;
                }

                if(XmOption.isLanguageC()) {
                    Xtype t = id.Type();
                    Xobject size;
                    if(t.isVariableArray()) {
                        size = Xcons.binaryOp(Xcode.MUL_EXPR, refOMPvarExpr(
                            block, t.getArraySizeExpr().copy()), Xcons.SizeOf(t.getRef()));
                    } else {
                        size = Xcons.SizeOf(t);
                    }
                    flush_vars.add(Xcons.List(id.getAddr(), size));
                }
            }
        }
    }

    OMPinfo findContext(OMPpragma pragma)
    {
        OMP.debug("findContext");
        for(OMPinfo i = this.parent; i != null; i = i.parent)
            if(i.pragma == pragma)
                return i;
        return null;
    }

    OMPinfo findContext(OMPpragma pragma, Xobject arg)
    {
        OMP.debug("findContext");
        for(OMPinfo i = this.parent; i != null; i = i.parent)
            if(i.pragma == pragma && i.arg == arg)
                return i;
        return null;
    }

    public List<Xobject> getRegionArgs()
    {
        OMP.debug("getRegionArgs");
        return region_args;
    }
    
    private void addRegionArg(Xobject x)
    {
        OMP.debug("getRegionArg");
        region_args.add(x);
    }
    
    public List<Ident> getRegionParams()
    {
        OMP.debug("getRegionParams");
        return region_params;
    }
    
    private void addRegionParam(Ident id)
    {
        OMP.debug("addRegionParam"+id.toString());
        region_params.add(id);
    }

    public boolean hasIfExpr()
    {
        OMP.debug("hasIfExpr");
        return if_expr != null;
    }
    
    public Xobject getIfExpr()
    {
        OMP.debug("getIfExpr");
        return if_expr;
    }

    public boolean hasFinalExpr()
    {
        OMP.debug("hasFinalExpr");
        return final_expr != null;
    }
    
    public Xobject getFinalExpr()
    {
        OMP.debug("getFinalExpr");
        return final_expr;
    }

    public Xobject getIdList()
    {
        OMP.debug("getIdList");
        return id_list;
    }
    
    public List<OMPvar> getThdPrvVarList()
    {
        OMP.debug("getThdPrvVarList");
        return thdprv_varlist;
    }
    
    private void addThdPrvVar(OMPvar var)
    {
        OMP.debug("addThdPrvVar");
        thdprv_varlist.add(var);
    }

    public OMPpragma getSched()
    {
        OMP.debug("getSched");
        return sched;
    }

    public void setSchedChunk(Xobject new_chunk)
    {
        OMP.debug("setSchedChunk");
        sched_chunk = new_chunk;
    }

    public Xobject getSchedChunk()
    {
        OMP.debug("getSchedChunk");
        return sched_chunk;
    }

    public boolean isOrdered()
    {
        OMP.debug("isOrdered");
        return ordered;
    }

    public boolean isNoWait()
    {
        OMP.debug("isNoWait");
        return no_wait;
    }
    
    public Xobject getNumThreads()
    {
        OMP.debug("getNumThreads");
        return num_threads;
    }
    
    public boolean hasCopyPrivate()
    {
        OMP.debug("hasCopyPrivate");
        for(OMPvar v : varlist) {
            if(v.is_copy_private)
                return true;
        }
        return false;
    }
    
    public Xobject getThreadNumVar()
    {
        OMP.debug("getThreadNumVar");
        if(thdnum_var == null) {
            thdnum_var = Ident.Fident(OMPtransPragma.THDNUM_VAR, Xtype.FintType).getAddr();
        }
        return thdnum_var;
    }
}
