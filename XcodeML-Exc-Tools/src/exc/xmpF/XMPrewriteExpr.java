package exc.xmpF;

import exc.object.*;
import exc.block.*;
import exc.xcalablemp.*;
import java.util.*;
import static xcodeml.util.XmLog.fatal;

/**
 * pass2: check and write variables
 */
public class XMPrewriteExpr
{
  private XMPenv  env;

  public XMPrewriteExpr(){ }

  public void run(FuncDefBlock def, XMPenv env) {
    this.env = env;
    env.setCurrentDef(def);

    XMP.debug("pass2:");
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    // rewrite return statements
    BlockIterator iter5 = new topdownBlockIterator(fb);
    for (iter5.init(); !iter5.end(); iter5.next()){
      if (iter5.getBlock().Opcode() == Xcode.RETURN_STATEMENT){
	Block b = Bcons.GOTO(Xcons.StringConstant(XMP.epilog_label_f));
	iter5.setBlock(b);
      }
    }

    // rewrite pointer assignment, allocate, deallocate, and stop statements
    BasicBlockIterator iter3 = new BasicBlockIterator(fb);
    for (iter3.init(); !iter3.end(); iter3.next()){
      Block b = iter3.getBasicBlock().getParent();
      StatementIterator iter4 = iter3.getBasicBlock().statements();
      while (iter4.hasNext()){
	Statement st = iter4.next();
	Xobject x = st.getExpr();
	if (x == null)  continue;
	switch (x.Opcode()) {
	case F_POINTER_ASSIGN_STATEMENT:
	  {
	    // must be performed before rewriteExpr

	    Xobject lhs_obj = x.getArg(0);

	    while (lhs_obj.Opcode() == Xcode.MEMBER_REF ||
		   lhs_obj.Opcode() == Xcode.F_ARRAY_REF ||
		   lhs_obj.Opcode() == Xcode.F_VAR_REF)
	      lhs_obj = lhs_obj.getArg(0).getArg(0);
	      
	    Ident lhs_id = env.findVarIdent(lhs_obj.getName(), b);
	    if (lhs_id == null) break;
	    XMParray lhs_array = XMParray.getArray(lhs_id);
	    if (lhs_array == null) break;

	    Xobject rhs_obj = x.getArg(1);

	    while (rhs_obj.Opcode() == Xcode.MEMBER_REF ||
		   rhs_obj.Opcode() == Xcode.F_ARRAY_REF ||
		   rhs_obj.Opcode() == Xcode.F_VAR_REF)
	      rhs_obj = rhs_obj.getArg(0).getArg(0);
	      
	    Ident rhs_id = env.findVarIdent(rhs_obj.getName(), b);
	    if (rhs_id == null) break;
	    XMParray rhs_array = XMParray.getArray(rhs_id);
	    if (rhs_array == null) break;

	    lhs_array.rewritePointerAssign(x, rhs_array, st, fb, env);

	    break;
	  }
	case F_ALLOCATE_STATEMENT:
	  {
            // must be performed before rewriteExpr
	    Iterator<Xobject> y = ((XobjList)x.getArg(1)).iterator();
	    while (y.hasNext()){
	      XobjList alloc = (XobjList)y.next();
	      Xobject obj = alloc.getArg(0);

	      while (obj.Opcode() == Xcode.MEMBER_REF ||
		     obj.Opcode() == Xcode.F_ARRAY_REF ||
		     obj.Opcode() == Xcode.F_VAR_REF)
		  obj = obj.getArg(0).getArg(0);

	      Ident id = env.findVarIdent(obj.getName(), b);
	      if (id == null) break;
	      XMParray array = XMParray.getArray(id);
	      if (array == null) break;

	      array.rewriteAllocate(alloc, st, fb, env);
	    }
	    break;
	  }
	case F_DEALLOCATE_STATEMENT:
	  {
            // must be performed before rewriteExpr
	    Iterator<Xobject> y = ((XobjList)x.getArg(1)).iterator();
	    while (y.hasNext()){
	      XobjList dealloc = (XobjList)y.next();
	      Xobject obj = dealloc.getArg(0);
	      while (obj.Opcode() == Xcode.MEMBER_REF ||
		     obj.Opcode() == Xcode.F_ARRAY_REF ||
		     obj.Opcode() == Xcode.F_VAR_REF)
		obj = obj.getArg(0).getArg(0);
	      Ident id = env.findVarIdent(obj.getName(), b);
	      if (id == null) break;
	      XMParray array = XMParray.getArray(id);
	      if (array == null) break;

	      array.rewriteDeallocate(dealloc, st, fb, env);
	    }
	    break;
	  }

	case F_STOP_STATEMENT:
	  {
	    Ident f = env.declInternIdent(XMP.finalize_all_f, Xtype.FsubroutineType);
	    Xobject call = f.callSubroutine();
	    st.insert(call);
	    break;
	  }

	case EXPR_STATEMENT: // subroutine call
	  insertSizeArray(st, fb);
	  break;

	}
      }
    }

    // rewrite id_list, decles, parameters
    rewriteDecls(fb);
    
    // rewrite expr
    BasicBlockExprIterator iter = new BasicBlockExprIterator(fb);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      if(expr != null) rewriteExpr(expr,iter.getBasicBlock(),fb);
    }
    
    // rewrite OMP pragma
    topdownBlockIterator iter2 = new topdownBlockIterator(fb);
    for (iter2.init(); !iter2.end(); iter2.next()){
      Block block = iter2.getBlock();
      if (block.Opcode() == Xcode.OMP_PRAGMA){
	Xobject clauses = ((PragmaBlock)block).getClauses();
	rewriteOmpClauses(clauses, (PragmaBlock)block, fb);
      }
    }

    // rewrite ACC pragma
    for (iter2.init(); !iter2.end(); iter2.next()){
      Block block = iter2.getBlock();
      if (block.Opcode() == Xcode.ACC_PRAGMA){
        rewriteAccClauses((PragmaBlock)block, fb);
      }
    }
  }

  private void rewriteDecls(FunctionBlock funcBlock){
    BlockIterator biter = new topdownBlockIterator(funcBlock);
    for (biter.init(); !biter.end(); biter.next()){
      if (biter.getBlock().Opcode() == Xcode.F_BLOCK_STATEMENT ||
          biter.getBlock().Opcode() == Xcode.FUNCTION_DEFINITION ||
	  biter.getBlock().Opcode() == Xcode.F_MODULE_DEFINITION){
        Xobject decl_list = biter.getBlock().getBody().getDecls();
        Xobject id_list = biter.getBlock().getBody().getIdentList();
        Xtype f_type = biter.getBlock().Opcode() == Xcode.FUNCTION_DEFINITION ?
                         funcBlock.getNameObj().Type() : null;
        Xobject f_params = null;
        if(f_type != null && f_type.isFunction())
          f_params = f_type.getFuncParam();

        for(Xobject i: (XobjList) id_list){
          Ident id = (Ident)i;
	  if(id.Type() == null) continue;  // COMMON Block
	  boolean isStructure = (id.Type().getKind() == Xtype.STRUCT);
	  StorageClass sclass = id.getStorageClass();
	  if(isStructure && (sclass == StorageClass.FLOCAL || sclass == StorageClass.FSAVE)){
	    XobjList memberList = env.findVarIdent(id.getName(), null).Type().getMemberList();
	    for(Xobject x: memberList){
	      Ident memberId = (Ident)x;
	      if(memberId.isMemberAligned()){
		Ident structVarId    = id;
		String memberName    = memberId.getName();
		String structVarName = structVarId.getName();
		Block structVarBlock = env.findVarIdentBlock(structVarName, funcBlock);
		XMParray arrayObject = new XMParray();
		memberId.setProp(XMP.StructId,id);
		XMPenv env2 = (XMPenv)memberId.getProp(XMP.Env);
		arrayObject.parseAlignForStructure(structVarName, structVarId, structVarBlock,
						   memberName, memberId, env2);
		env2.declXMParray(arrayObject, structVarBlock);
		memberId.setProp(XMP.RWprotected, arrayObject);
		
		if(memberId.getProp(XMP.Shadow_w_list) != null)
		  arrayObject.analyzeShadowForStructure(funcBlock);
		
		XMPinfo info = (XMPinfo)memberId.getProp(XMP.HasShadow);
		if(info != null)
		  info.addReflectArray(arrayObject);
	      }
	    }
	  } // end if
	
	  XMParray array = XMParray.getArray(id);
          if(array == null) continue;
      
          // write id
          String a_name = id.getName();
          String xmp_name = array.getLocalName();
          Xtype xmp_type = array.getLocalType();

          for(Xobject decl: (XobjList) decl_list){
            if(decl.Opcode() != Xcode.VAR_DECL) continue;
            Xobject decl_id = decl.getArg(0);

            if(decl_id.Opcode() == Xcode.IDENT && 
               decl_id.getName().equals(a_name)){
              decl.setArg(0,Xcons.Symbol(Xcode.IDENT,xmp_type,xmp_name));
              break;
            }
          }
          if(f_params != null && id.getStorageClass() == StorageClass.FPARAM){
            // rewrite parameter
            for(Xobject param: (XobjList)f_params){
              if(param.Opcode() == Xcode.IDENT &&
                 param.getName().equals(a_name)){
                param.setName(xmp_name);
                param.setType(xmp_type);
              }
            }
          }
        }
      }
    }
  }

  private final static String XMP_REPLACED_GLOBAL = "XMP_REPLACED_GLOBAL";
    
  /*
   * rewrite expression
   */
  private void rewriteExpr(Xobject expr, BasicBlock bb, Block block){
    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    for(iter.init(); !iter.end();iter.next()){
      Xobject x = iter.getXobject();
      if (x == null)  continue;
      if (x.Opcode() == null) continue;      // #060  see [Xmp-dev:4675]
      switch (x.Opcode()) {
      case MEMBER_REF:
	{
	  if(x.getArg(0).getArg(0).Opcode() != Xcode.VAR) continue;
	  String structName = x.getArg(0).getName();
	  Ident id = env.findVarIdent(structName, bb.getParent());
	  if (id.Type().getKind() != Xtype.STRUCT) continue;
	  String memberName = x.getArg(1).getName();
	  Ident memberId = id.Type().getMemberList().getIdent(XMP.PREFIX_ + memberName);
	  if(memberId == null || ! memberId.isMemberAligned()) continue;
	  XMParray array = XMParray.getArray(memberId.getOrigId());
	  Xobject memberObj = x.getArg(1);
	  if(memberObj.getProp(XMP.RWprotected) != null) break;
	  memberObj.setName(array.getLocalName());
	  memberObj.setType(array.getLocalType());
	  memberObj.setProp(XMP.arrayProp, array);
	  memberObj.setProp(XMP_REPLACED_GLOBAL, true);
	  break;
	}
      case VAR:
	{
	  if(x.getProp(XMP.RWprotected) != null) break;

	  Ident id = env.findVarIdent(x.getName(),bb.getParent());
	  if(id == null) break;
	  XMParray array = XMParray.getArray(id);
	  if(array == null) break;
	  
	  // replace with local decl
          ((XobjString)x).setName(array.getLocalName());
	  if (array.getLocalId().getFdeclaredModule() != null &&
	      array.getLocalId().getProp(XMP.globalAlias) != null){
	    x.setType(array.getLocalType().copy());
	  }
	  else {
	    x.setType(array.getLocalType());
	  }
          x.setProp(XMP.arrayProp,array);
	  x.setProp(XMP_REPLACED_GLOBAL, true);
	  break;
	}
      case F_ARRAY_REF:
	{
	  Xobject a = x.getArg(0);
	  if (a.Opcode() != Xcode.F_VAR_REF)
	    XMP.fatal("not F_VAR_REF for F_ARRAY_REF");
	  a = a.getArg(0);
	  if (a.Opcode() != Xcode.VAR && a.Opcode() != Xcode.MEMBER_REF) break;
	  if (a.Opcode() != Xcode.VAR && a.getArg(0).getArg(0).Opcode() != Xcode.VAR) break;
	  String varName = (a.Opcode() == Xcode.VAR)? a.getName() : a.getArg(0).getName();
	  Ident id = env.findVarIdent(varName, bb.getParent());
	  XMParray globalAlias    = null;
	  Object isReplacedGlobal = null;
	  if (id != null){
	    if (a.Opcode() == Xcode.VAR){
	      globalAlias = (XMParray)id.getProp(XMP.globalAlias);
	      isReplacedGlobal = a.getProp(XMP_REPLACED_GLOBAL);
	    }
	    else{
	      Xobject memberObj = a.getArg(1);
	      globalAlias = (XMParray)memberObj.getProp(XMP.globalAlias);
	      isReplacedGlobal = memberObj.getProp(XMP_REPLACED_GLOBAL);
	    }
	  }

	  if (globalAlias != null &&
	      (isReplacedGlobal == null || !(boolean)isReplacedGlobal)){ // local alias

	    if (globalAlias == null) break;

	    FindexRange origIndexRange = (FindexRange)id.getProp(XMP.origIndexRange);
	    
	    int i = 0;
	    for (XobjArgs args=x.getArg(1).getArgs(); args!=null; args=args.nextArgs()){
	      Xobject origLB = origIndexRange.getLbound(i);
	      Xobject index  = null;
	      switch (args.getArg().Opcode()){
	      case F_INDEX_RANGE:
		  index = args.getArg();
		  Xobject lb = index.getArg(0);
		  if (lb != null){
		    lb = Xcons.binaryOp(Xcode.MINUS_EXPR, lb, origLB);
		    if (!globalAlias.isDistributed(i)){
		      lb = Xcons.binaryOp(Xcode.PLUS_EXPR, lb, globalAlias.getLowerAt(i));
		    }
		    index.setArg(0, lb);
		  }
		  Xobject ub = index.getArg(1);
		  if (ub != null){
		    ub = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, origLB);
		    if (!globalAlias.isDistributed(i)){
		      ub = Xcons.binaryOp(Xcode.PLUS_EXPR, ub, globalAlias.getLowerAt(i));
		    }
		    index.setArg(1, ub);
		  }
		  args.setArg(index);
		  break;
	      case F_ARRAY_INDEX:
	      default:
		  index = Xcons.binaryOp(Xcode.MINUS_EXPR,
					 args.getArg().getArg(0), origLB);
		  if (!globalAlias.isDistributed(i)){
		      index = Xcons.binaryOp(Xcode.PLUS_EXPR,
					     index, globalAlias.getLowerAt(i));
		  }
		  args.getArg().setArg(0, index);
		  break;
	      }
	      i++;
	    }
	  }
	  else {
	    XMParray array;
	    if (a.Opcode() == Xcode.VAR){
	      array = (XMParray)a.getProp(XMP.arrayProp);
	    }
	    else{
	      Xobject memberObj = a.getArg(1);
	      array = (XMParray)memberObj.getProp(XMP.arrayProp);
	    }
	    if(array == null) break;

	    int dim_i = 0;
	    boolean no_leading_scalar_in_subscripts = true;
	    for(XobjArgs args=x.getArg(1).getArgs();args!= null;args=args.nextArgs()){
	      Xobject index_calc = arrayIndexCalc(array,dim_i++,args.getArg(),bb,block);
	      if(index_calc != null) args.setArg(index_calc);
	    }
	  
	    if(array.isLinearized())
	      x.setArg(1,array.convertLinearIndex(x.getArg(1)));
	  }

	  break;
	}

        // XXX delete this
      case CO_ARRAY_REF:
	{
	  break;
	}

      case FUNCTION_CALL:
	{
	  if (x.getArg(0).Opcode() == Xcode.MEMBER_REF) break;
	  String fname = x.getArg(0).getString();
	  if (fname.equalsIgnoreCase("xmp_desc_of")){

	    env.removeIdent(fname, block);

	    XMParray array = (XMParray) x.getArg(1).getArg(0).getProp(XMP.arrayProp);
	    if (array != null){
	      Xobject desc = array.getDescId();
	      iter.setXobject(desc);
	      break;
	    }

	    String objName = x.getArg(1).getArg(0).getString();
	    XMPobject obj = env.findXMPobject(objName, block);
	    if (obj != null) {
	      Xobject desc = obj.getDescId();
	      iter.setXobject(desc);
	      env.removeIdent(objName, block);
	      break;
	    }

	    XMP.errorAt(block, "xmp_desc_of applied to non-XMP object");
	    XMP.exitByError();

	    break;
	  }
	  else if (fname.equalsIgnoreCase("xmp_transpose")){
	    XMParray array0 = (XMParray) x.getArg(1).getArg(0).getProp(XMP.arrayProp);
	    XMParray array1 = (XMParray) x.getArg(1).getArg(1).getProp(XMP.arrayProp);
	    if (array0 == null || array1 == null){
	      XMP.errorAt(block,"wrong argument of xmp_transpose");
	      XMP.exitByError();
	    }
	    x.getArg(1).setArg(0, array0.getDescId());
	    x.getArg(1).setArg(1, array1.getDescId());
	  }
	  break;
	}
      }
    }
  }

  /*
   * rewrite Pragma
   */
  private void rewriteOmpClauses(Xobject expr, PragmaBlock pragmaBlock, Block block){
    boolean private_done = false;
    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    
    for (iter.init(); !iter.end();iter.next()){
      Xobject x = iter.getXobject();
      if (x == null)  continue;
      
      if (x.Opcode() == Xcode.VAR){
	  if (x.getProp(XMP.RWprotected) != null) break;

	  Ident id = env.findVarIdent(x.getName(),block);
	  if (id == null) break;
	  XMParray array = XMParray.getArray(id);

	  if (array != null){
	      // replace with local decl
	      Xobject var = Xcons.Symbol(Xcode.VAR,array.getLocalType(),
					 array.getLocalName());
	      var.setProp(XMP.arrayProp,array);
	      iter.setXobject(var);
	  }
      }
      else if (x.Opcode() == Xcode.LIST){
	  if (x.left() != null && x.left().Opcode() == Xcode.STRING &&
	      x.left().getString().equals("DATA_PRIVATE")){

	      private_done = true;

	      if (!pragmaBlock.getPragma().equals("FOR")) continue;

	      XobjList itemList = (XobjList)x.right();

	      // find loop variable
	      Xobject loop_var = null;
	      BasicBlockIterator i = new BasicBlockIterator(pragmaBlock.getBody());
	      for (Block b = pragmaBlock.getBody().getHead();
		   b != null;
		   b = b.getNext()){
		  if (b.Opcode() == Xcode.F_DO_STATEMENT){
		      loop_var = ((FdoBlock)b).getInductionVar();
		  }
	      }
	      if (loop_var == null) continue;

	      // check if the clause has contained the loop variable
	      boolean flag = false;
	      Iterator<Xobject> j = itemList.iterator();
	      while (j.hasNext()){
		  Xobject item = j.next();
		  if (item.getName().equals(loop_var.getName())){
		      flag = true;
		  }
	      }

	      // add the loop variable to the clause
	      if (!flag){
		  itemList.add(loop_var);
	      }
	  }
      }
    }

    if (!private_done){
      if (!pragmaBlock.getPragma().equals("FOR")) return;

      // find loop variable
      Xobject loop_var = null;
      BasicBlockIterator i = new BasicBlockIterator(pragmaBlock.getBody());
      for (Block b = pragmaBlock.getBody().getHead();
	   b != null;
	   b = b.getNext()){
	if (b.Opcode() == Xcode.F_DO_STATEMENT){
	  loop_var = ((FdoBlock)b).getInductionVar();
	}
      }
      if (loop_var == null) return;
      
      // add the loop variable to the clause
      Xobject thread_private = Xcons.List(Xcons.StringConstant("DATA_PRIVATE"),
					  Xcons.List(loop_var));
      pragmaBlock.addClauses(thread_private);
    }
  }

  private void rewriteAccClauses(PragmaBlock pragmaBlock, Block block) {
    Xobject clauses = pragmaBlock.getClauses();

    XobjectIterator iter = new bottomupXobjectIterator(clauses);
    for (iter.init(); !iter.end();iter.next()){
      Xobject x = iter.getXobject();
      if(x == null) continue;

      switch(x.Opcode()) {
      case VAR: {
        if (x.getProp(XMP.RWprotected) != null) break;

        Ident id = env.findVarIdent(x.getName(), block);
        if (id == null) break;

        XMParray array = XMParray.getArray(id);

        if (array == null) break;

        // replace with local decl
        Xobject var = Xcons.Symbol(Xcode.VAR, array.getLocalType(), array.getLocalName());
        var.setProp(XMP.arrayProp, array);
        iter.setXobject(var);
      }
      break;
      case F_ARRAY_REF: {
        Xobject a = x.getArg(0);
        if(a.Opcode() != Xcode.F_VAR_REF)
          XMP.fatal("not F_VAR_REF for F_ARRAY_REF");
        a = a.getArg(0);
        XMParray array = (XMParray) a.getProp(XMP.arrayProp);
        if(array == null) break;

        Xobject indexRanges = x.getArg(1);
        Fshape arrayShape = new Fshape((FarrayType)array.getType(), block);

        int dim = array.getDim();
        Xobject subscripts[] = new Xobject[dim];
        for(int i = 0; i < dim; i++) subscripts[i] = indexRanges.getArg(i);
        Fshape refShape = new Fshape(new FindexRange(subscripts, block));
        FarrayType farrayType = (FarrayType)array.getType();
        Xobject[] arraySizes = farrayType.getFarraySizeExpr();

        for(int i = 0; i < dim; i++){
          Xobject refIndexRange = indexRanges.getArg(i);
          Xobject refIndexRangeAssumedShape = refIndexRange.getArgOrNull(3);

          if((refIndexRangeAssumedShape != null) && (refIndexRangeAssumedShape.getInt() != 0)) continue; //refIndexRange is assumed shape

          Xobject arraySizeAssumedShape = arraySizes[i].getArgOrNull(3);
          if((arraySizeAssumedShape != null) && (arraySizeAssumedShape.getInt() != 0)){
            XMP.fatal("subarray shape must be assumed shape for distributed assumed shape array in OpenACC pragma");
          }

          Xobject refLbound = refShape.lbound(i);
          Xobject refUbound = refShape.ubound(i);
          if(! arrayShape.lbound(i).equals(refLbound)
                  || ! arrayShape.ubound(i).equals(refUbound)){
            XMP.fatal("subarray shape must be same to the distributed array shape in OpenACC pragma");
          }

          //set is assumed shape
          refIndexRange.setArg(0, null);
          refIndexRange.setArg(1, null);
          refIndexRange.setArg(3, Xcons.IntConstant(1)); //set to assumed shape
        }
      } break;
      case LIST: {
        if (x.left() == null || x.left().Opcode() != Xcode.STRING) continue;

        String clauseName = x.left().getString();
        if(clauseName.equals("PRIVATE")){
          //need to rename induction variable
        }
      } break;
      }
    }
  }

  Xobject localIndexOffset;

  Xobject convertLocalIndex(Xobject v, int dim_i, XMParray a, 
			    BasicBlock bb, Block block){
    localIndexOffset = null;
    int loop_idx = -1;
    XMPobjectsRef on_ref = null;
    Ident local_loop_var = null;

    // find the loop variable v
    for(Block b = bb.getParent(); b != null; b = b.getParentBlock()){
      XMPinfo info = (XMPinfo)b.getProp(XMP.prop);
      if(info == null) continue;
      if(info.pragma != XMPpragma.LOOP) continue;

      for(int k = 0; k < info.getLoopDim(); k++){
	if(v.equals(info.getLoopVar(k))){
	  loop_idx = k;
	  on_ref = info.getOnRef();
	  local_loop_var = info.getLoopDimInfo(k).getLoopLocalVar();
	  break;
	}
      }
      if(loop_idx >= 0) break;
    }

    if(XMP.debugFlag) 
      System.out.println("convertLocalIndex v="+v+" loop_idx="+loop_idx);

    if(loop_idx < 0 || local_loop_var == null) return null; // not found

    if(on_ref.getRefObject().getKind() != XMPobject.TEMPLATE) 
      return null;
    
    XMPtemplate on_tmpl = on_ref.getTemplate();
    XMPtemplate a_tmpl = a.getAlignTemplate();
    if(on_tmpl != a_tmpl && !on_tmpl.getName().equals(a_tmpl.getName())){
	return null; // different template
    }
    
    if(XMP.debugFlag) System.out.println("same template");

    int a_tmpl_idx = a.getAlignSubscriptIndexAt(dim_i);
    int on_tmpl_idx = on_ref.getLoopOnIndex(loop_idx);

    if(XMP.debugFlag) 
      System.out.println("template index a_tmpl_idx="+a_tmpl_idx+
			 ", on_tmp_idx="+on_tmpl_idx);

    if(a_tmpl_idx != on_tmpl_idx) return null;

    Xobject off1;
    int lshadow = a.getShadowLeft(dim_i);
    if (lshadow != 0){
	off1 = Xcons.IntConstant(lshadow);
    }
    else {
	off1 = null;
    }

    Xobject off2 = a.getAlignSubscriptOffsetAt(dim_i);
    Xobject off3 = on_ref.getLoopOffset(on_ref.getLoopOnIndex(loop_idx));
    Xobject off4 = null;
    if (off3 != null){
      if (off2 != null)
	off4 = Xcons.binaryOp(Xcode.MINUS_EXPR, off2, off3);
      else 
	off4 = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR, off3);
    }
    else if (off2 != null)
      off4 = off2;
    else
      off4 = Xcons.IntConstant(0);
	
    if (off1 == null)
      localIndexOffset = off4;
    else
      localIndexOffset = Xcons.binaryOp(Xcode.PLUS_EXPR, off1, off4);

    if(XMP.debugFlag) 
      System.out.println("check template v="+local_loop_var
			 +" off="+localIndexOffset);

    return local_loop_var.Ref();
  }

  Xobject arrayIndexCalc(XMParray a, int dim_i, Xobject i, 
			 BasicBlock bb, Block block){
    switch(i.Opcode()){
    case F_ARRAY_INDEX:      // if not distributed, do nothing
      if(!a.isDistributed(dim_i)){
	Xobject x = a.convertOffset(dim_i);
	if (x != null)
	  i.setArg(0,Xcons.binaryOp(Xcode.MINUS_EXPR, i.getArg(0), x));
	return i;
      }
      
      if (!a.isFullShadow(dim_i)){
	Xobject e = i.getArg(0);
	Xobject x = null;
	
	if (e.isVariable()){
	  x = convertLocalIndex(e, dim_i, a, bb, block);
	  if (localIndexOffset != null){
	    x = Xcons.binaryOp(Xcode.PLUS_EXPR, x, localIndexOffset);
	  }
	}
	else {
	  x = e.copy();
	  int cnt = convertLocalIndexInExpression(x, dim_i, a, bb, block, 0);
	  if (cnt != 1) x = null;
	}
	
	if (x != null){
	  Xobject y = a.convertOffset(dim_i);
	  if (y != null)
	    x = Xcons.binaryOp(Xcode.MINUS_EXPR, x, y);
	  i.setArg(0, x);
	  return i;
	}
      }
      
      Xobject x = null;
      switch (a.getDistMannerAt(dim_i)){
      case XMPtemplate.BLOCK:
      case XMPtemplate.GBLOCK:
	{
	  x = Xcons.binaryOp(Xcode.MINUS_EXPR, i.getArg(0), a.getBlkOffsetVarAt(dim_i).Ref());
	  break;
	}
      default:
	{
	  Ident f = env.declInternIdent("xmpf_local_idx_",
					Xtype.Function(Xtype.intType));
	  x = f.Call(Xcons.List(a.getDescId().Ref(),
				Xcons.IntConstant(dim_i),
				i.getArg(0)));
	  break;
	}
      }
      i.setArg(0,x);
      return i;
    case F_INDEX_RANGE:
      if (!a.isDistributed(dim_i)) return i;
      if (is_colon(i, a, dim_i)){ // NOTE: this check is not strict.
	i.setArg(0, null); i.setArg(1, null); i.setArg(2, null);
      	return i;
      }
    default:
      XMP.errorAt(block,"bad expression in XMP array index");
      return null;
    }
  }

  private int convertLocalIndexInExpression(Xobject x, int dim_i, XMParray a,
					    BasicBlock bb, Block block, int cnt){
    Xobject v;
    switch (x.Opcode()){
    case PLUS_EXPR:

      if (x.right().isVariable()){
	v = convertLocalIndex(x.right(), dim_i, a, bb, block);
	if (v != null){
	  if (localIndexOffset != null)
	    v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, localIndexOffset);
	  x.setRight(v);
	  cnt++;
	}
      }
      else {
	cnt = convertLocalIndexInExpression(x.right(), dim_i, a, bb, block, cnt);
      }
      // fall through
    case MINUS_EXPR:
      // how to deal with the case for a(1 - (2 - i)) ???
      if (x.left().isVariable()){
	v = convertLocalIndex(x.left(), dim_i, a, bb, block);
	if (v != null){
	  if (localIndexOffset != null)
	    v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, localIndexOffset);
	  x.setLeft(v);
	  cnt++;
	}
      }
      else {
	cnt = convertLocalIndexInExpression(x.left(), dim_i, a, bb, block, cnt);
      }

      break;
    }
    return cnt;
  }

  private void insertSizeArray(Statement st, FunctionBlock fb){
    Xobject x = st.getExpr().getArg(0);
    if (x.Opcode() != Xcode.FUNCTION_CALL) return;

    Ident sizeArray = env.declOrGetSizeArray(fb);
    String fname = x.getArg(0).Opcode() != Xcode.MEMBER_REF ?
                   x.getArg(0).getString() : null;
    Xtype ftype = x.getArg(0).Type();
    while((ftype != null) && (ftype.copied != null)/*ftype.isFprocedure()*/)
      ftype = ftype.copied;
    XobjList arg_list = (XobjList)x.getArg(1);

    //
    // get interface of each argument
    //
    XobjList param_list = null;
    if ((ftype != null) && ftype instanceof FunctionType/*not VOID type*/){
      // internal or module procedures
      param_list = (XobjList)ftype.getFuncParam();
    }

    if (param_list == null && fname != null){
	    
      XobjList decl_list = (XobjList)fb.getBody().getDecls();

      // retrieve interface block
    DECLLOOP: for (Xobject decl: decl_list){
	if (decl.Opcode() == Xcode.F_INTERFACE_DECL){
	  XobjList func_list = (XobjList)decl.getArg(3);
	  for (Xobject func: func_list){
	    if (func.Opcode() == Xcode.FUNCTION_DECL &&
		func.getArg(0).getString().equals(fname)){
	      ftype = func.getArg(0).Type();
	      param_list = (XobjList)ftype.getFuncParam();
	      break DECLLOOP;
	    }
	  }
	}
      }
    }

    if (param_list == null || arg_list == null) return;

    int k = 0;
    int nargs = Math.min(arg_list.Nargs(), param_list.Nargs());
    for (int i = 0; i < nargs; i++){
      if (!param_list.getArg(i).Type().isFassumedShape()) continue;
	
      Xobject arg = arg_list.getArg(i);

      if (arg.Opcode() == Xcode.VAR ||
	  arg.Opcode() == Xcode.MEMBER_REF){ // just array name
	  
	Xtype atype;
	int arrayDim;
	Ident sizeFunc;
	Xobject arg0;
	XMParray array = null;

	if (arg.Opcode() != Xcode.MEMBER_REF){
	  Ident id = env.findVarIdent(arg.getName(), fb);
	  array = XMParray.getArray(id);
	}

	if (array != null){
	  atype = array.getType();
	  arrayDim = array.getDim();
	  sizeFunc = env.declExternIdent("xmp_array_gsize", Xtype.FintFunctionType);
	  arg0 = array.getDescId().Ref();
	}
	else {
	  atype = arg.Type();
	  arrayDim = atype.getNumDimensions();
	  sizeFunc = env.declIntrinsicIdent("size", Xtype.FintFunctionType);
	  arg0 = arg;
	}

	if (atype.isFallocatable() || atype.isFassumedShape()){
	  for (int j = 0; j < arrayDim; j++){
	    Xobject lhs = Xcons.FarrayRef(sizeArray.Ref(), Xcons.IntConstant(k), Xcons.IntConstant(j));
	    Xobject rhs = sizeFunc.Call(Xcons.List(arg0, Xcons.IntConstant(j+1)));
	    st.insert(Xcons.Set(lhs, rhs));
	  }
	}
	else {
	  Xobject declSize[] = atype.getFarraySizeExpr();
	  for (int j = 0; j < arrayDim; j++){
	    Xobject lhs = Xcons.FarrayRef(sizeArray.Ref(), Xcons.IntConstant(k), Xcons.IntConstant(j));
	    Xobject rhs = Xcons.binaryOp(Xcode.PLUS_EXPR,
					 Xcons.binaryOp(Xcode.MINUS_EXPR,
							declSize[j].getArg(1),
							declSize[j].getArg(0)),
					 Xcons.IntConstant(1));
	    st.insert(Xcons.Set(lhs, rhs));
	  }
	}
      }
      else if (arg.Opcode() == Xcode.F_ARRAY_REF){ // array section
	Xtype atype;
	int arrayDim;
	Ident lbFunc, ubFunc;
	Xobject arg0;
	XMParray array = null;

	if (arg.getArg(0).getArg(0).Opcode() != Xcode.MEMBER_REF){
	  Ident id = env.findVarIdent(arg.getArg(0).getArg(0).getName(), fb);
	  array = XMParray.getArray(id);
	}

	if (array != null){
	  atype = array.getType();
	  arrayDim = array.getDim();
	  lbFunc = env.declExternIdent("xmp_array_lbound", Xtype.FintFunctionType);
	  ubFunc = env.declExternIdent("xmp_array_ubound", Xtype.FintFunctionType);
	  arg0 = array.getDescId().Ref();
	}
	else {
	  atype = arg.getArg(0).getArg(0).Type();
	  arrayDim = atype.getNumDimensions();
	  lbFunc = env.declIntrinsicIdent("lbound", Xtype.FintFunctionType);
	  ubFunc = env.declIntrinsicIdent("ubound", Xtype.FintFunctionType);
	  arg0 = arg.getArg(0).getArg(0);
	}

	XobjList subList = (XobjList)arg.getArg(1);
	for (int j = 0; j < arrayDim; j++){
	  Xobject sub = subList.getArg(j);
	  Xobject rhs = null;

	  if (sub.Opcode() == Xcode.F_ARRAY_INDEX){
	    //rhs = Xcons.IntConstant(1);
	    continue;
	  }
	  else if (sub.Opcode() == Xcode.F_INDEX_RANGE){
	    Xobject lb = sub.getArg(0);
	    if (lb == null){
	      lb = lbFunc.Call(Xcons.List(arg0, Xcons.IntConstant(j+1)));
	    }
	    Xobject ub = sub.getArg(1);
	    if (ub == null){
	      ub = ubFunc.Call(Xcons.List(arg0, Xcons.IntConstant(j+1)));
	    }

	    Xobject stride = sub.getArg(2);
	    if (stride == null){
	      rhs = Xcons.binaryOp(Xcode.PLUS_EXPR,
				   Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb),
				   Xcons.IntConstant(1));
	    }
	    else {
	      rhs = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb);
	      rhs = Xcons.binaryOp(Xcode.PLUS_EXPR, rhs, stride);
	      rhs = Xcons.binaryOp(Xcode.DIV_EXPR, rhs, stride);
	    }
	  }

	  Xobject lhs = Xcons.FarrayRef(sizeArray.Ref(), Xcons.IntConstant(k), Xcons.IntConstant(j));
	  st.insert(Xcons.Set(lhs, rhs));
	}
      }
      else {
	continue;
      }
      k++;
      if (k > XMP.MAX_ASSUMED_SHAPE){
	XMP.fatal("too many assumed-shape arguments (MAX = " + XMP.MAX_ASSUMED_SHAPE + ").");
      }
    }
  }

  private boolean is_colon(Xobject i, XMParray a, int dim_i){
    // check lower
    if (i.getArg(0) != null){
      Xobject lb = a.getType().getFarraySizeExpr()[dim_i].getArg(0);
      if (!lb.equals(i.getArg(0))) return false;
    }

    // check upper
    if (i.getArg(1) != null){
      Xobject ub = a.getType().getFarraySizeExpr()[dim_i].getArg(1);
      if (!ub.equals(i.getArg(1))) return false;
    }

    // check stride
    if (i.getArg(2) != null){
      if (i.getArg(2).Opcode() != Xcode.INT_CONSTANT ||
  	  i.getArg(2).getInt() != 1) return false;
    }

    return true;
  }
}
