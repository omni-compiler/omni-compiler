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
  XMPenv env;

  public XMPanalyzePragma() {}

  public void run(FuncDefBlock def, XMPenv env) {
    this.env = env;
    env.setCurrentDef(def);

    Block b;
    Block fblock = def.getBlock();


    // pass1: traverse to collect information about XMP pramga
    XMP.debug("pass1:");

    // check module use 
    checkUseDecl(fblock.getBody().getDecls());

    b = fblock.getBody().getHead();
    if(b == null) return; // null body, do nothing
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

  private void checkUseDecl(Xobject decls){
    if(decls == null) return;
    for(Xobject decl: (XobjList)decls){
      if(decl.Opcode() != Xcode.F_USE_DECL) continue;
      String name = decl.getArg(0).getName();
      env.useModule(name);
      env.findModule(name); // import 
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
      analyzeTask(pb.getClauses(), pb.getBody(), info, pb);
      break;

    case TASKS:
      { analyzeTasks(pb);			break; }

    case GMOVE:
      analyzeGmove(pb.getClauses(),pb.getBody(), info, pb);
      break;

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
    if (loopIterList == null || loopIterList.Nargs() == 0) {
      ForBlock loopBlock = getOutermostLoopBlock(loopBody);
      if(loopBlock == null){
	XMP.errorAt(pb,"loop is not found after loop directive");
	return;
      }
      dims.add(XMPdimInfo.loopInfo(loopBlock));
      loopBody = loopBlock.getBody();
    } else {
      while(true){
	ForBlock loopBlock = getOutermostLoopBlock(loopBody);
	if(loopBlock == null) break;
	boolean is_found = false;
	for(Xobject x: loopIterList){
	  if(x.Opcode() == Xcode.LIST) x = x.getArg(0);
	  if(isEqualVar(loopBlock.getInductionVar(),x)){
	    is_found = true;
	    break;
	  }
	}
	if(is_found)
	  dims.add(XMPdimInfo.loopInfo(loopBlock));
	loopBody = loopBlock.getBody();
      }

      /* check loopIterList */
      for(Xobject x: loopIterList){
	if(x.Opcode() == Xcode.LIST){
	  if(x.getArgOrNull(1) != null ||
	     x.getArgOrNull(2) != null){
	    XMP.errorAt(pb,"bad syntax in loop directive");
	  }
	  x = x.getArg(0);
	}
	boolean is_found = false;
	for(XMPdimInfo d_info: dims){
	  if(isEqualVar(d_info.getLoopVar(),x)){
	    is_found = true;
	    break;
	  }
	}
	if(!is_found)
	  XMP.errorAt(pb,"loop index is not found in loop varaibles");
      }
    }
    
    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(loopDecl.getArg(1),env,pb);
    if(XMP.hasError()) return;

    /* check on ref: it should be v+off */
    int on_ref_idx = 0;
    Vector<XMPdimInfo> on_ref_dims = on_ref.getSubscripts();
    for(int k = 0; k < on_ref_dims.size(); k++){
      XMPdimInfo d_info = on_ref_dims.elementAt(k);
      if(d_info.isStar()) continue;
      if(d_info.isTriplet()){
	XMP.errorAt(pb,"on-ref in loop must not be triplet");
      } else {
	Xobject t = d_info.getIndex();
	if(t == null) continue;
	Xobject v = null;
	Xobject off = null;
	if(t.isVariable()){
	  v = t;
	} else {
	  switch(t.Opcode()){
	  case PLUS_EXPR:
	  case MINUS_EXPR:
	    if(!t.left().isVariable())
	      XMP.errorAt(pb,"left hand-side in align-subscript must be a variable");
	    else {
	      v = t.left();
	      off = t.right();
	      if(t.Opcode() == Xcode.MINUS_EXPR)
		off = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR,off);
	    }
	    // check right-hand side?
	    break;
	  default:
	    XMP.errorAt(pb,"bad expression in subsript of on-ref");
	  }
	}
	if(v == null) continue; // some error occurs
	// find varaible
	int idx = -1;
	for(int i = 0; i < dims.size(); i++){
	  if(isEqualVar(v,dims.elementAt(i).getLoopVar())){
	    idx = i;
	    dims.elementAt(i).setLoopOnIndex(k);
	    break;
	  }
	}
	if(idx < 0)
	  XMP.errorAt(pb,"loop variable is not found in on_ref: '"
		      +v.getName()+"'");
	d_info.setLoopOnRefInfo(idx,off);
      }
    }
    
    Xobject reductionSpec = loopDecl.getArg(2);
    if(reductionSpec != null) 
      analyzeReductionSpec(info, reductionSpec, pb);

    on_ref.setLoopDimInfo(dims);  // set back pointer
    
    checkLocalizableLoop(dims,on_ref,pb);

    info.setBody(loopBody);  // inner most body
    info.setLoopInfo(dims, on_ref);
  }

  private static ForBlock getOutermostLoopBlock(BlockList body) {
    Block b = body.getHead();
    if (b != null) {
      if (b.Opcode() == Xcode.F_DO_STATEMENT) {
        if (b.getNext() != null){
          // XMP.error("only one loop statement is allowed in loop directive");
	  return null;
	}
        ForBlock forBlock = (ForBlock)b;
        forBlock.Canonicalize();
        if (!(forBlock.isCanonical())){
	  // XMP.error("loop statement is not canonical");
	  return null;
	}
        return forBlock;
      }
      else if (b.Opcode() == Xcode.COMPOUND_STATEMENT)
        return getOutermostLoopBlock(b.getBody());
    } 
//     else {
//       XMP.error("cannot find a loop statement");
//     }
    return null;
  }

  /*
   * distributed loop is localizable if:
   * (1) step is 1
   * (2) distribution is BLOCK
   */
  private void checkLocalizableLoop(Vector<XMPdimInfo> dims,
				    XMPobjectsRef on_ref, Block b){
    for(int i = 0; i < dims.size(); i++){
      boolean localizable = false;
      XMPdimInfo d_info = dims.elementAt(i);
      if(d_info.getStride().isOneConstant()) 
	localizable = true;
      else {
	if(!(on_ref.getRefObject() instanceof XMPtemplate))
	  continue;
	XMPtemplate tmpl = (XMPtemplate)on_ref.getRefObject();
	if(tmpl.isDistributed()){
	  if(tmpl.getDistMannerAt(d_info.getLoopOnIndex()) 
	     == XMPtemplate.BLOCK)
	    localizable = true;
	} else 
	  XMP.errorAt(b,"template '"+tmpl.getName()+"' is not distributed");
      }

      if(XMP.debugFlag)
	System.out.println("localizable(i="+i+")="+localizable);

      if(localizable){
	Xobject loop_var = d_info.getLoopVar();
	/* if localiable, allocate local */
	Ident local_loop_var = 
	  env.declIdent(XMP.genSym(loop_var.getName()),
			loop_var.Type(),b);
	d_info.setLoopLocalVar(local_loop_var);
      }
    }
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
	XMP.errorAt(pb,"Bad array name in reflect name list");
	continue;
      }
      String name = x.getName();
      Ident id = env.findVarIdent(name,pb);
      if(id == null){
	XMP.errorAt(pb,"variable '" + name + "'for reflect is not declared");
	continue;
      }
      XMParray array =  XMParray.getArray(id);
      if(array == null){
	XMP.errorAt(pb,"array '" + name + "'for reflect is not declared");
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

    analyzeReductionSpec(info, reductionSpec, pb);
    info.setOnRef(XMPobjectsRef.parseDecl(reductionOnRef,env,pb));
  }

  private void analyzeReductionSpec(XMPinfo info, Xobject reductionSpec,
				    PragmaBlock pb){
    Xobject op = reductionSpec.getArg(0);
    if(!op.isIntConstant()) XMP.fatal("reduction: op is not INT");

    Vector<Ident> reduction_vars = new Vector<Ident>();
    for(Xobject v: (XobjList)reductionSpec.getArg(1)){
      if(!v.isVariable()){
	XMP.errorAt(pb,"not variable in reduction spec list");
      }
      Ident id = env.findVarIdent(v.getName(),pb);
      if(id == null){
	XMP.errorAt(pb,"variable '"+v.getName()+"' in reduction is not found");
      }
      reduction_vars.add(id);
    }
    info.setReductionInfo(op.getInt(),reduction_vars);
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
	XMP.errorAt(pb,"not variable in bcast variable list");
      }
      Ident id = env.findVarIdent(v.getName(),pb);
      if(id == null){
	XMP.errorAt(pb,"variable '"+v.getName()+"' in reduction is not found");
      }
      bcast_vars.add(id);
    }
    
    info.setBcastInfo(XMPobjectsRef.parseDecl(fromRef,env,pb),
		      XMPobjectsRef.parseDecl(toRef,env,pb),
		      bcast_vars);
  }

  void analyzeTask(Xobject taskDecl, BlockList taskBody,
		   XMPinfo info, PragmaBlock pb) {
    Xobject onRef = taskDecl.getArg(0);
    Xobject taskOpt = taskDecl.getArg(1);
    
    if(taskOpt != null){
      XMP.fatal("task opt is not supported yet, sorry!");
      return;
    }
    info.setOnRef(XMPobjectsRef.parseDecl(onRef,env,pb));
  }

  private void analyzeTasks(PragmaBlock pb) {
    XMP.fatal("analyzeTasks");
  }

  private void analyzeGmove(Xobject gmoveDecl, BlockList body, 
			    XMPinfo info, PragmaBlock pb) {
    Xobject gmoveOpt = gmoveDecl.getArg(0);
    Xobject Opt = gmoveDecl.getArg(1);

    // check body is single statement.
    Block b = body.getHead();
    if(b.getNext() != null) XMP.fatal("not single block for Gmove");
    Statement s = b.getBasicBlock().getHead();
    if(b.Opcode() != Xcode.F_STATEMENT_LIST ||s.getNext() != null)
      XMP.fatal("not single statment for Gmove");
    Xobject x = s.getExpr();
    if(x.Opcode() != Xcode.F_ASSIGN_STATEMENT)
      XMP.fatal("not assignment for Gmove");
    Xobject left = x.left();
    Xobject right = x.right();
    
    // opcode must be VAR or ARRAY_REF
    boolean left_is_global = checkGmoveOperand(left,pb);
    boolean right_is_global = checkGmoveOperand(right,pb);

    if(!left_is_global && !right_is_global)
      XMP.errorAt(pb,"local assignment for gmove");
    
    if(XMP.hasError()) return;
    
    info.setGmoveOperands(left,right);
  }

  private boolean checkGmoveOperand(Xobject x, PragmaBlock pb){
    switch(x.Opcode()){
    case F_ARRAY_REF:
      Xobject a = x.getArg(0);
      if(a.Opcode() != Xcode.F_VAR_REF)
	XMP.fatal("not F_VAR_REF for F_ARRAY_REF");
      a = a.getArg(0);
      if(a.Opcode() != Xcode.VAR)
	XMP.fatal("not VAR for F_VAR_REF");
      Ident id = env.findVarIdent(a.getName(),pb);
      if(id == null)
	XMP.fatal("array in F_ARRAY_REF is not declared");
      XMParray array = XMParray.getArray(id);
      if(array != null){
	a.setProp(XMP.RWprotected,array);
	return true;
      }
    case VAR: /* it ok */
      break;
    default:
      XMP.errorAt(pb,"gmove must be followed by simple assignment");
    }
    return false;
  }

  private void analyzeCoarray(Xobject coarrayPragma){
    XMP.fatal("analyzeCoarray");
  }
}
