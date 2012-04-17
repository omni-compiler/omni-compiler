/* 
 * $TSUKUBA_Release: Omni XcalableMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import java.util.ArrayList;
import java.util.List;

import exc.object.*;
import exc.block.*;
import java.util.*;
import xcodeml.util.XmOption;

/**
 * process analyze XcalableMP pragma
 * pass1: check directive and allocate descriptor
 */
public class XMPanalyzePragma
{
  FuncDefBlock def;
  XMPenv env;

  public XMPanalyzePragma() {}

  public void run(FuncDefBlock def, XMPenv env) {
    this.def = def;
    this.env = env;
    Block b;
    Block fblock = def.getBlock();

    // pass1: traverse to collect information about XMP pramga
    XMP.debug("pass1:");

    b = fblock.getBody().getHead();
    b.setProp(XMP.prop, new XMPinfo(XMPpragma.FUNCTION_BODY, null, b, env));

    // scan by topdown iterator
    BlockIterator i = new topdownBlockIterator(fblock);
    for(i.init(); !i.end(); i.next()) {
      b = i.getBlock();
      if(XMP.debugFlag)	System.out.println("pass1=" + b);
      if(b.Opcode() == Xcode.XMP_PRAGMA)
	analyzePragma((PragmaBlock)b);
    }
  }

  private void analyzePragma(PragmaBlock pb) {
    String pragmaName = pb.getPragma();

    // debug
    if(XMP.debugFlag){
      System.out.println("Pragma: "+pragmaName);
      System.out.println(" Clause= "+pb.getClauses());
      System.out.println(" Body= "+pb.getBody());
    }

    XMPinfo outer = null;
    for(Block bp = pb.getParentBlock(); bp != null; bp = bp.getParentBlock()) {
      if(bp.Opcode() == Xcode.XMP_PRAGMA) {
	outer = (XMPinfo)bp.getProp(XMP.prop);
      }
    }
    XMPinfo info = new XMPinfo(XMPpragma.valueOf(pb.getPragma()), outer, pb, env);
    pb.setProp(XMP.prop, info);

    XMPpragma p = info.pragma;
    switch (p){
    case NODES:
      {
	Xobject clauses = pb.getClauses();
	XMPnodes.analyzePragma(clauses, env, pb);
      }
      break;

    case TEMPLATE:
      {
	Xobject templateDecl = pb.getClauses();
	XobjList templateNameList = (XobjList)templateDecl.getArg(0);
	  
	for(Xobject xx:templateNameList){
	  XMPtemplate.analyzeTemplate(xx,templateDecl.getArg(1), env, pb);
	}
      }
      break;

    case DISTRIBUTE:
      {
	Xobject distributeDecl = pb.getClauses();
	XobjList distributeNameList = (XobjList)distributeDecl.getArg(0);
	Xobject distributeDeclCopy = distributeDecl.copy();

	for(Xobject xx:distributeNameList){
	  XMPtemplate.analyzeDistribute(xx,distributeDecl.getArg(1),
					distributeDecl.getArg(2),env, pb);
	}
      }
      break;

    case ALIGN:
      {
	Xobject alignDecl = pb.getClauses();
	XobjList alignNameList = (XobjList)alignDecl.getArg(0);

	for(Xobject xx: alignNameList){
	  XMParray.analyzeAlign(xx, alignDecl.getArg(1),
				  alignDecl.getArg(2),
				  alignDecl.getArg(3),
				  env, pb);
	  if(XMP.hasError()) break;
	}
      }
      break;

    case SHADOW:
      {
 	Xobject shadowDecl = pb.getClauses();
 	XobjList shadowNameList = (XobjList) shadowDecl.getArg(0);
	Xobject shadow_w_list = shadowDecl.getArg(1);

	for(Xobject xx: shadowNameList){
	  XMParray.analyzeShadow(xx,shadow_w_list,env,pb);
	  if(XMP.hasError()) break;
 	}
      }
      break;

    case LOOP:
      analyzeLoop(pb.getClauses(), pb.getBody(), pb,info);
      break; 

    case REFLECT:
      analyzeReflect(pb.getClauses(),info,pb);
      break;

    case BARRIER:
      analyzeBarrier(pb.getClauses(),info,pb);
      break;

    case REDUCTION:
      analyzeReduction(pb.getClauses(),info,pb);
      break;

    case BCAST:
      analyzeBcast(pb.getClauses(),info,pb);
      break;

    case TASK:
      { analyzeTask(pb);			break; }
    case TASKS:
      { analyzeTasks(pb);			break; }
    case GMOVE:
      { analyzeGmove(pb);			break; }

    case COARRAY:
      // { translateCoarrayDecl(x);   		break; }

    default:
      XMP.fatal("'" + pragmaName.toLowerCase() + 
		"' directive is not supported yet");
    }
  }

  private static boolean isEqualVar(Xobject v1, Xobject v2){
    return (v1.isVariable() && v2.isVariable() &&
	    v1.getName().equals(v2.getName()));
  }

  /* 
   * analyze Loop directive:
   *  loopDecl = (on_ref | ...)
   */
  void analyzeLoop(Xobject loopDecl, BlockList loopBody,
		   PragmaBlock pb, XMPinfo info) {
    // get block to schedule
    Vector<XMPdimInfo> dims = new Vector<XMPdimInfo>();
    
    // schedule loop
    XobjList loopIterList = (XobjList)loopDecl.getArg(0);
    if (loopIterList == null) {
      ForBlock loopBlock = getOutermostLoopBlock(loopBody);
      dims.add(XMPdimInfo.loopInfo(loopBlock));
      loopBody = loopBlock.getBody();
    } else {
      for(Xobject x: loopIterList){
	ForBlock loopBlock = getOutermostLoopBlock(loopBody);
	if(loopBlock == null) return;
	if(x.Opcode() == Xcode.LIST) x = x.getArg(0);
	if(!isEqualVar(loopBlock.getInductionVar(),x)){
	  XMP.error("loop index is different from loop varaible");
	  return;
	}
	dims.add(XMPdimInfo.loopInfo(loopBlock));
	loopBody = loopBlock.getBody();
      }
    }
    
    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(loopDecl.getArg(1),env,pb);
    if(XMP.hasError()) return;

    /* check on ref: it should be v+off */
    for(XMPdimInfo d_info: on_ref.getSubscripts()){
      if(d_info.isStar()) continue;
      if(d_info.isTriplet()){
	XMP.error("on-ref in loop must not be triplet");
      } else {
	Xobject t = d_info.getIndex();
	Xobject v = null;
	Xobject off = null;
	if(t.isVariable()){
	  v = t;
	} else {
	  switch(t.Opcode()){
	  case PLUS_EXPR:
	  case MINUS_EXPR:
	    if(!t.left().isVariable())
	      XMP.error("left hand-side in align-subscript must be a variable");
	    else {
	      v = t.left();
	      off = t.right();
	      if(t.Opcode() == Xcode.MINUS_EXPR)
		off = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR,off);
	    }
	    // check right-hand side?
	    break;
	  default:
	    XMP.error("bad expression in subsript of on-ref");
	  }
	}
	if(v == null) continue; // some error occurs
	// find varaible 
	int idx = -1;
	for(int i = 0; i < dims.size(); i++){
	  if(isEqualVar(v,dims.elementAt(i).getLoopVar())){
	    idx = i;
	    break;
	  }
	}
	if(idx < 0) XMP.error("loop variable is not found in on_ref: '"+v.getName()+"'");
	d_info.setLoopOnRefInfo(idx,off);
      }
    }

    Xobject reductionRef = loopDecl.getArg(2);
    // should check reduction clause

    info.setBody(loopBody);
    info.setLoopInfo(dims, on_ref, reductionRef);
  }

  private static ForBlock getOutermostLoopBlock(BlockList body) {
    Block b = body.getHead();
    if (b != null) {
      if (b.Opcode() == Xcode.F_DO_STATEMENT) {
        if (b.getNext() != null){
          XMP.error("only one loop statement is allowed in loop directive");
	  return null;
	}
        ForBlock forBlock = (ForBlock)b;
        forBlock.Canonicalize();
        if (!(forBlock.isCanonical())){
	  XMP.error("loop statement is not canonical");
	  return null;
	}
        return forBlock;
      }
      else if (b.Opcode() == Xcode.COMPOUND_STATEMENT)
        return getOutermostLoopBlock(b.getBody());
    } else {
      XMP.error("cannot find a loop statement");
    }
    return null;
  }

  private void analyzeReflect(Xobject reflectDecl, 
			      XMPinfo info, PragmaBlock pb){
    XobjList reflectNameList = (XobjList) reflectDecl.getArg(0);
    Xobject reflectOpt = reflectDecl.getArg(1);

    if(reflectOpt != null){
      XMP.fatal("reflect opt is not supported yet, sorry!");
      return;
    }

    Vector<XMParray> reflectArrays = new Vector<XMParray>();
    // check array
    for(Xobject x: reflectNameList){
      if(!x.isVariable()){
	XMP.error("Bad array name in reflect name list");
	continue;
      }
      String name = x.getName();
      XMParray array = env.getXMParray(name, pb);
      if(array == null){
	XMP.error("array '" + name + "'for reflect is not declared");
	continue;
      }
      reflectArrays.add(array);
    }
    info.setReflectArrays(reflectArrays);
  }

  void analyzeBarrier(Xobject barrierDecl, 
		      XMPinfo info, PragmaBlock pb) {
    Xobject barrierOnRef = barrierDecl.getArg(0);
    Xobject barrierOpt = barrierDecl.getArg(1);

    if(barrierOpt != null){
      System.out.println("opt="+barrierOpt);
      XMP.fatal("barrier opt is not supported yet, sorry!");
      return;
    }

    info.setOnRef(XMPobjectsRef.parseDecl(barrierOnRef,env,pb));
  }

  void analyzeReduction(Xobject reductionDecl, 
			XMPinfo info, PragmaBlock pb){
    Xobject reductionSpec = reductionDecl.getArg(0);
    Xobject reductionOnRef = reductionDecl.getArg(1);
    Xobject reductionOpt = reductionDecl.getArg(2);

    if(reductionOpt != null){
      XMP.fatal("redution opt is not supported yet, sorry!");
      return;
    }

    Xobject op = reductionSpec.getArg(0);
    if(!op.isIntConstant()) XMP.fatal("reduction: op is not INT");

    Vector<Ident> reduction_vars = new Vector<Ident>();
    for(Xobject v: (XobjList)reductionSpec.getArg(1)){
      if(!v.isVariable()){
	XMP.error("not variable in reduction spec list");
      }
      Ident id = env.findVarIdent(v.getName(),pb);
      if(id == null){
	XMP.error("variable '"+v.getName()+"' in reduction is not found");
      }
      reduction_vars.add(id);
    }
    info.setReductionInfo(op.getInt(),reduction_vars);
    
    info.setOnRef(XMPobjectsRef.parseDecl(reductionOnRef,env,pb));
  }

  private void analyzeBcast(Xobject bcastDecl, 
			XMPinfo info, PragmaBlock pb){
    XobjList bcastNameList = (XobjList) bcastDecl.getArg(0);
    Xobject fromRef = bcastDecl.getArg(1);
    Xobject toRef = bcastDecl.getArg(2);
    Xobject bcastOpt = bcastDecl.getArg(3);

    if(bcastOpt != null){
      XMP.fatal("bcast opt is not supported yet, sorry!");
      return;
    }

    Vector<Ident> bcast_vars = new Vector<Ident>();
    for(Xobject v: bcastNameList){
      if(!v.isVariable()){
	XMP.error("not variable in bcast variable list");
      }
      Ident id = env.findVarIdent(v.getName(),pb);
      if(id == null){
	XMP.error("variable '"+v.getName()+"' in reduction is not found");
      }
      bcast_vars.add(id);
    }
    
    info.setBcastInfo(XMPobjectsRef.parseDecl(fromRef,env,pb),
		      XMPobjectsRef.parseDecl(toRef,env,pb),
		      bcast_vars);
  }

  private void analyzeTask(PragmaBlock pb){
    XMP.fatal("analyzeTask");
  }

  private void analyzeTasks(PragmaBlock pb) {
    XMP.fatal("analyzeTasks");
  }


  private void analyzeGmove(PragmaBlock pb) {
    XMP.fatal("analyzeGmove");
  }

  private void analyzeCoarray(Xobject coarrayPragma){
    XMP.fatal("analyzeCoarray");
  }

}
