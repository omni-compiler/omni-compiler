/* 
 * $TSUKUBA_Release: Omni XcalableMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import exc.object.*;
import exc.util.MachineDep;
import exc.block.*;

import java.io.File;
import java.util.*;

/**
 * AST transformation by XcalableMP directives.
 */
public class XMPtransPragma
{
  protected XobjectDef current_def;
  private XMPenv  xmp_env;
  protected XobjectFile env;
    
  public XMPtransPragma() { }

  // pass3: do transformation for XMP pragma
  public void run(FuncDefBlock def, XMPenv env) {
    Block b;
    Block fblock = def.getBlock();
    current_def = def.getDef();
    this.env = current_def.getFile();
    this.xmp_env = env; 

    XMP.debug("pass3:");
    // scan by bottom-up iterator
    BlockIterator i = new bottomupBlockIterator(fblock);
    for(i.init(); !i.end(); i.next()) {
      b = i.getBlock();
      if(b.Opcode() == Xcode.XMP_PRAGMA) {
	b = transPragma((PragmaBlock)b);
	if(b != null) i.setBlock(b);
      }
    }
    // constructor for XMPobjects
    fblock.getBody().getHead().
      getBody().getHead().insert(localConstructorBlock(fblock));
  }

  public Block localConstructorBlock(Block fblock){
    BlockList body = Bcons.emptyBody();
    XMPsymbolTable table = XMPenv.localXMPsymbolTable(fblock);
    if(table != null){
      for(XMPobject o: table.getXMPobjects()){
	Block b1 = o.buildConstructor(current_def);
	if(b1 != null) body.add(b1);
      }
      for(XMParray a: table.getXMParrays()){
	Block b2 = a.buildConstructor(current_def);
	if(b2 != null) body.add(b2);
      }
    }
    return Bcons.COMPOUND(body);
  }

  // pass3:
  // write pragma
  Block transPragma(PragmaBlock pb) {
    XMPinfo i = (XMPinfo)pb.getProp(XMP.prop);
    if(i == null) return null;

    switch (i.pragma){
    case NODES:
    case TEMPLATE:
    case DISTRIBUTE:
    case ALIGN:
    case SHADOW:
    case COARRAY:
      /* declaration directives, do nothing */
      return Bcons.emptyBlock();

    case LOOP:
      return translateLoop(pb,i);

    case TASK:
      return translateTask(pb,i);
    case TASKS:
      return translateTasks(pb,i);
    case REFLECT:
      return translateReflect(pb,i);
    case BARRIER:
      return translateBarrier(pb,i);
    case REDUCTION:
      return translateReduction(pb,i);	
    case BCAST:
      return translateBcast(pb,i);     
    case GMOVE:
      return translateGmove(pb,i);
    default:
      // XMP.fatal("unknown pragma");
      // ignore it
      return null;
    }
  }

  static boolean loop_opt_enable = false;

  private Block translateLoop(PragmaBlock pb, XMPinfo info){
    if(!loop_opt_enable){
      // default loop transformation (non-opt)
      // just guard body
      // DO I = lb, up, step ; if(xmp_is_loop_1(i,on_desc)) body
      // create on_desc, only use loop_on_ref
      BlockList ret_body = Bcons.emptyBody();
      XMPobjectsRef on_ref = info.getLoopOnRef();
      ret_body.add(on_ref.buildConstructor(current_def));

      ForBlock for_block = (ForBlock)pb.getBody().getHead();
      BlockList loopBody = info.getBody();
      FdoBlock for_inner = (FdoBlock)loopBody.getParent();
      Block new_body = 
	Bcons.IF(BasicBlock.Cond(on_ref.buildLoopTestFuncCall(current_def,info)),
		 loopBody,null);
      for_inner.setBody(new BlockList(new_body));
      ret_body.add((Block)for_block);
      return Bcons.COMPOUND(ret_body);
    }

    // default loop transformation (convert to local index)
    // DO I = lb, up, step ; body
    // to:----
    // call loop_on_desc(on_desc,#dim); call loop_on_disc_set(on_desc,src,dst,off); 
    // DO I = lb, up, step ; if(xmp_is_loop_1(i,on_desc)) body

    ForBlock for_block = (ForBlock)pb.getBody().getHead();
    
    BlockList loop_body = Bcons.emptyBody();
    CompoundBlock comp_block = (CompoundBlock)Bcons.COMPOUND(loop_body);
    Xobject ind_var = for_block.getInductionVar();
    BlockList ret_body = Bcons.emptyBody();

    // addDataSetupBlock(ret_body, i);
    Xtype indvarType = Xtype.FuintPtrType;
    Xtype btype = indvarType;
    Xtype step_type = Xtype.intType;

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

    vlb = lb_var.Ref();
    vub = ub_var.Ref();
    vlb_addr = lb_var.getAddr();
    vub_addr = ub_var.getAddr();
    loop_body.add((Block)for_block);
        
    BasicBlock bb = new BasicBlock();
    ret_body.add(bb);
        
    if(for_block.getInitBBlock() != null) {
      for(Statement s : for_block.getInitBBlock()) {
	if(s.getNext() == null)
	  break;
	s.remove();
	bb.add(s); // move s to bb
      }
    }
        
    bb.add(Xcons.Set(lb_var.Ref(), for_block.getLowerBound()));
    bb.add(Xcons.Set(ub_var.Ref(), for_block.getUpperBound()));
    bb.add(Xcons.Set(step_var.Ref(), for_block.getStep()));

    for_block.setLowerBound(lb_var.Ref());
    for_block.setUpperBound(ub_var.Ref());
    for_block.setStep(step_ref);

    // schedule
    bb.add(XMPfuncIdent("xmpf_sched").
	   Call(Xcons.List(vlb_addr, vub_addr, step_addr)));
    ret_body.add(comp_block);

    // bp = dataUpdateBlock(i);
    // if(bp != null)
    // ret_body.add(bp);

    return Bcons.COMPOUND(ret_body);
  }

  private Block translateReflect(PragmaBlock pb, XMPinfo i){
    XMP.fatal("translateReflect");
    return null;
  }

  private Block translateTask(PragmaBlock pb, XMPinfo i){
    XMP.fatal("translateTask");
    return null;
  }

  private Block translateTasks(PragmaBlock pb, XMPinfo i) {
    XMP.fatal("translateTasks");
    return null;
  }

  private Block translateBarrier(PragmaBlock pb, XMPinfo i) {
    XMP.fatal("translateBarrier");
    return null;
  }

  private Block translateReduction(PragmaBlock pb, XMPinfo i){
    XMP.fatal("translateReduction");
    return null;
  }

  private Block translateBcast(PragmaBlock pb, XMPinfo i){
    XMP.fatal("translateBcast");
    return null;
  }

  private Block translateGmove(PragmaBlock pb, XMPinfo i) {
    XMP.fatal("translateGmove");
    return null;
  }

// declare func and return Ident with type
  public Ident XMPfuncIdent(String name) {
    if(name == null)
      throw new IllegalArgumentException("may be generating unsupported function call");
        
    Xtype t;
//     if(name == getMaxThreadsFunc || name == getThreadNumFunc ||
//        name == getNumThreadsFunc || name == sectionIdFunc)
//       t = Xtype.FintFunctionType;
//     else if(name == isLastFunc || name == isMasterFunc ||
// 	    name == doSingleFunc ||
// 	    name == staticShedNextFunc || name == dynamicShedNextFunc ||
// 	    name == guidedShedNextFunc || name == runtimeShedNextFunc ||
// 	    name == affinityShedNextFunc)
//       t = Xtype.FlogicalFunctionType;
//     else
    t = Xtype.FsubroutineType;
    return current_def.declExternIdent(name, t);
  }
}


