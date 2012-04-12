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
    System.out.println("Pragma: "+pragmaName);
    System.out.println(" Clause= "+pb.getClauses());
    System.out.println(" Body= "+pb.getBody());

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
	  
	Iterator<Xobject> iter = templateNameList.iterator();
	while (iter.hasNext()) {
	  Xobject xx = iter.next();
	  XMPtemplate.analyzeTemplate(xx,templateDecl.getArg(1), env, pb);
	}
      }
      break;
    case DISTRIBUTE:
      {
	Xobject distributeDecl = pb.getClauses();
	XobjList distributeNameList = (XobjList)distributeDecl.getArg(0);
	Xobject distributeDeclCopy = distributeDecl.copy();

	Iterator<Xobject> iter = distributeNameList.iterator();
	while (iter.hasNext()) {
	  Xobject xx = iter.next();
	  XMPtemplate.analyzeDistribute(xx,distributeDecl.getArg(1),
					distributeDecl.getArg(2),env, pb);
	}
      }
      break;
    case ALIGN:
      {
	Xobject alignDecl = pb.getClauses();
	XobjList alignNameList = (XobjList)alignDecl.getArg(0);

	Iterator<Xobject> iter = alignNameList.iterator();
	while (iter.hasNext()) {
	  Xobject xx = iter.next();
	  XMParray.analyzeAlign(xx, alignDecl.getArg(1),
				  alignDecl.getArg(2),
				  alignDecl.getArg(3),
				  env, pb);
	}
      }
      break;
    case SHADOW:
      {
// 	Xobject shadowDecl = pb.getClauses();
// 	XobjList shadowNameList = (XobjList) shadowDecl.getArg(0);
// 	Xobject shadowDeclCopy = shadowDecl.copy();

// 	Iterator<Xobject> iter = shadowNameList.iterator();
// 	while (iter.hasNext()) {
// 	  Xobject xx = iter.next();
// 	  shadowDeclCopy.setArg(0, xx);
// 	  XMPshadow.analyzeShadow(shadowDeclCopy, env , pb);
// 	}
      }
      break;
    case COARRAY:
      // translateCoarrayDecl(x);
      break;

    case LOOP:
      { analyzeLoop(pb,info);			break; }

    case TASK:
      { analyzeTask(pb);			break; }
    case TASKS:
      { analyzeTasks(pb);			break; }
    case REFLECT:
      { analyzeReflect(pb);			break; }
    case BARRIER:
      { analyzeBarrier(pb);			break; }
    case REDUCTION:
      { analyzeReduction(pb);		break; }
    case BCAST:
      { analyzeBcast(pb);			break; }
    case GMOVE:
      { analyzeGmove(pb);			break; }
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
  private void analyzeLoop(PragmaBlock pb, XMPinfo info) {
    XobjList loopDecl = (XobjList)pb.getClauses();
    BlockList loopBody = pb.getBody();

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

  private void analyzeCoarray(Xobject coarrayPragma){
    XMP.fatal("analyzeCoarray");
  }

  private void analyzeReflect(PragmaBlock pb){
    XMP.fatal("analyzeReflect");
  }

  private void analyzeTask(PragmaBlock pb){
    XMP.fatal("analyzeTask");
  }

  private void analyzeTasks(PragmaBlock pb) {
    XMP.fatal("analyzeTasks");
  }

  private void analyzeBarrier(PragmaBlock pb) {
    XMP.fatal("analyzeBarrier");
  }

  private void analyzeReduction(PragmaBlock pb){
    XMP.fatal("analyzeReduction");
  }

  private void analyzeBcast(PragmaBlock pb){
    XMP.fatal("analyzeBcast");
  }

  private void analyzeGmove(PragmaBlock pb) {
    XMP.fatal("analyzeGmove");
  }
}
