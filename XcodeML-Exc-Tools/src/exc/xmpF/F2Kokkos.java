package exc.xmpF;

import exc.xcalablemp.XMPexception;
import exc.object.*;
import exc.util.MachineDep;
import exc.block.*;
import xcodeml.util.XmOption;
import xcodeml.util.XmLanguage;
import java.io.*;
import java.util.*;
import xcodeml.util.IXobject;

/**
 * F2Kokkos
 */
public class F2Kokkos {

  private KKSdecompileWriter out = null;
  private static final int BUFFER_SIZE = 4096;
  private static final String KKS_SRC_EXTENSION = "_kks.cc";
  private XobjList _KKSFuncParams = null;
  
  // constructor
  public F2Kokkos(XobjectFile env){
    try {
      Writer w = new BufferedWriter(new FileWriter(getSrcName(env.getSourceFileName()) + KKS_SRC_EXTENSION),
				    BUFFER_SIZE);
      out = new KKSdecompileWriter(w, env);

      // add header include line
      out.println("# include <Kokkos_Core.hpp>");
      out.println("# include \"flcl-cxx.hpp\"");
      out.println();
      out.flush();
    } catch (IOException e) {
      //throw new XMPexception("error in gpu decompiler: " + e.getMessage());
    }

  }

  
  private static String getSrcName(String srcName) {
    String name = "";
    String[] buffer = srcName.split("\\.");
    for (int i = 0; i < buffer.length - 1; i++) {
      name += buffer[i];
    }

    return name;
  }
  

  //
  // generate Kokkos code
  //
  
  public XobjectDef generateKKSFunc(XMPpragma kind, BlockList loopBody,
				    XobjList dataList, XobjList onList, XobjList tileList, XobjList reductionList,
				    int num_kernels){

    XmOption.setLanguage(XmLanguage.C);

    Vector<Integer> ndims = new Vector<>();
    XobjList lbList = Xcons.List();
    XobjList ubList = Xcons.List();
    XobjList tList = Xcons.List();
    
    // The wrapper
    
    Ident KKSFuncId = Ident.Local(XMP.kokkos_sub_f + String.valueOf(num_kernels), Xtype.Function(Xtype.voidType));

    //
    // The parameters
    //
    
    Xobject KKSFuncParams = Xcons.List(Xcode.ID_LIST);
    Xobject view_from_ndarray_args = Xcons.List();

    // "00000" is a dummy.
    Xtype flcl_ndarray_t = new StructType("00000", true, Xcons.String("flcl_ndarray_t"),
				      null, 0L, null);

    for (Xobject a: dataList){
      if (a.Type().isFarray()){
	Ident nd_array_param = Ident.Param("nd_array_" + a.getName(), Xtype.Pointer(flcl_ndarray_t));
	KKSFuncParams.add(nd_array_param);
	Ident array_lb_param = Ident.Param(a.getName() + "_lb", Xtype.Pointer(Xtype.intType));
	KKSFuncParams.add(array_lb_param);
	view_from_ndarray_args.add(nd_array_param.Ref());
	ndims.add(((FarrayType)a.Type()).getNumDimensions());
      }
      else {
	Ident param = Ident.Param(a.getName(), Xtype.Pointer(F2CPP_type(a.Type())));
	KKSFuncParams.add(param);
      }
    }      

    for (Xobject a: onList){
      Ident param_lower = Ident.Param(a.getName() + "_lb", Xtype.intType);
      Ident param_upper = Ident.Param(a.getName() + "_ub", Xtype.intType);
      KKSFuncParams.add(param_lower);
      KKSFuncParams.add(param_upper);
      lbList.add(param_lower.Ref());
      ubList.add(param_upper.Ref());
    }

    int tdim = 0;
    for (Xobject t: tileList){
      Ident param_t = Ident.Param("t" + String.valueOf(tdim++), Xtype.intType);
      KKSFuncParams.add(param_t);
      tList.add(param_t.Ref());
    }
    
    Ident param_reducer = null;
    if (reductionList != null){
      Xobject reducer = reductionList.getArg(1).getArg(0).getArg(0);
      param_reducer = Ident.Param(reducer.getName(), Xtype.Pointer(F2CPP_type(reducer.Type())));
      KKSFuncParams.add(param_reducer);
    }

    // Ident nd_array_x = Ident.Param("nd_array_x", Xtype.Pointer(Xtype.intType));
    // Ident nd_array_y = Ident.Param("nd_array_y", Xtype.Pointer(Xtype.intType));
    // Ident a = Ident.Param("a", Xtype.floatType);
    // Ident ilen = Ident.Param("i_len", Xtype.intType);
    // Ident jlen = Ident.Param("j_len", Xtype.intType);
    
    // Xobject KKSFuncParams = Xcons.List(nd_array_x, nd_array_y, a, ilen, jlen);

    //
    // The body
    //
    
    // Convert Fortran arrays to Kokkos view

    //auto x = flcl::view_from_ndarray<float**>(*nd_array_x);
    //auto y = flcl::view_from_ndarray<float**>(*nd_array_y);

    //Ident view_from_ndarray = Ident.Local("flcl::view_from_ndarray<float**>", Xtype.Function(Xtype.charType));
    
    Xobject id_list = Xcons.List();
    Xobject decls = Xcons.List();

    int i = 0;
    for (Xobject d: dataList){
      if (d.Type().isFarray()){
	Ident x = Ident.Local(d.getName(), Xtype.autoType);
	Xobject args_x = Xcons.List(Xcons.PointerRef(view_from_ndarray_args.getArg(i)));

	String view_from_ndarray_name = "flcl::view_from_ndarray<float";
	for (int j = 0; j < ndims.get(i); j++){
	  view_from_ndarray_name += "*";
	}
	view_from_ndarray_name += ">";
	// charType is a dummy.
	Ident view_from_ndarray = Ident.Local(view_from_ndarray_name, Xtype.Function(Xtype.charType));

	Xobject decl_x = Xcons.List(Xcode.VAR_DECL, x.getValue(), Xcons.functionCall(view_from_ndarray, args_x));
	id_list.add(x);
	decls.add(decl_x);

	i++;
      }
    }      

    // Ident x = Ident.Local("x", Xtype.autoType);
    // Xobject args_x = Xcons.List(Xcons.PointerRef(nd_array_x.Ref()));
    // Xobject decl_x = Xcons.List(Xcode.VAR_DECL, x.getValue(), Xcons.functionCall(view_from_ndarray, args_x));

    // Ident y = Ident.Local("y", Xtype.autoType);
    // Xobject args_y = Xcons.List(Xcons.PointerRef(nd_array_y.Ref()));
    // Xobject decl_y = Xcons.List(Xcode.VAR_DECL, y.getValue(), Xcons.functionCall(view_from_ndarray, args_y));

    // Xobject id_list = Xcons.List(x, y);
    // Xobject decls = Xcons.List(decl_x, decl_y);

    BlockList KKSBlockList = new BlockList(id_list, decls);

    //
    // The kernel
    //

    Ident kernelId = null;
    if (kind == XMPpragma.PARALLEL_FOR){
      kernelId = Ident.Local("parallel_for", Xtype.Function(Xtype.voidType));
    }
    else {
      kernelId = Ident.Local("parallel_reduce", Xtype.Function(Xtype.voidType));
    }

    Xobject kernelArgs = Xcons.List();
    kernelArgs.add(Xcons.StringConstant(XMP.kokkos_sub_f + String.valueOf(num_kernels)));

    // MDRangePolicy

    String nnests = String.valueOf(onList.Nargs());
    Ident MDRangePolicyId = Ident.Local("Kokkos::MDRangePolicy<Kokkos::Rank<"+nnests+">>",
					Xtype.Function(Xtype.voidType)); // voidType is a dummy.
    Xobject MDRangePolicyArgs = Xcons.List();
    // kernelArgs.add(Xcons.List(il, jl),
    // 	      Xcons.List(Xcons.binaryOp(Xcode.PLUS_EXPR, iu, Xcons.IntConstant(1)),
    // 			 Xcons.binaryOp(Xcode.PLUS_EXPR, ju, Xcons.IntConstant(1))),
    // 	      Xcons.List(t1, t2));

    //MDRangePolicyArgs.add(Xcons.List(Xcons.IntConstant(0), Xcons.IntConstant(0)));
    //MDRangePolicyArgs.add(Xcons.List(ilen.Ref(), jlen.Ref()));

    MDRangePolicyArgs.add(lbList);
    MDRangePolicyArgs.add(ubList);
    if (tList.Nargs() > 0) MDRangePolicyArgs.add(tList);

    kernelArgs.add(Xcons.functionCall(MDRangePolicyId, MDRangePolicyArgs));

    // KOKKOS_LAMBDA

    //args.add(KOKKOS_LAMBDA(int i, int j) "{y(i-yil,j-yjl) = y(i-yil,j-yjl) + a * x(i-xil,j-xjl)");

    Ident kokkosLambdaId = Ident.FidentNotExternal("KOKKOS_LAMBDA", Xtype.Function(Xtype.voidType));
    Xobject kokkosLambdaParams = Xcons.List();

    for (Xobject a: onList){
      kokkosLambdaParams.add(Ident.Local(a.getName(), Xtype.intType));
    }

    if (reductionList != null){
      Xobject x = reductionList.getArg(1).getArg(0).getArg(0);
      //Ident id = Ident.Local("l_" + x.getSym(), Xtype.Pointer(x.Type()));
      Xtype type = x.Type().copy(); type.setIsReference(true);
      Ident id = Ident.Local("l_" + x.getSym(), type);
      kokkosLambdaParams.add(id);
    }
    
    // translate loop body

    _KKSFuncParams = (XobjList)KKSFuncParams;
    loopBody = F2CPP_loopBody(loopBody, reductionList);
    Block kernelBlock = Bcons.COMPOUND(loopBody);

    Xobject kokkos_lambda = Xcons.List(Xcode.FUNCTION_DEFINITION, kokkosLambdaId, kokkosLambdaParams,
				       null, kernelBlock.toXobject());
    
    kernelArgs.add(kokkos_lambda);

    if (kind == XMPpragma.PARALLEL_REDUCE){
      //Xobject result = reductionList.getArg(1).getArg(0).getArg(0);
      kernelArgs.add(Xcons.PointerRef(param_reducer.Ref()));
    }
    
    Xobject kernel = Xcons.functionCall(kernelId, kernelArgs);
    KKSBlockList.add(Bcons.Statement(kernel));

    // generate Kokkos::fence

    Ident fenceId = Ident.Local("Kokkos::fence", Xtype.Function(Xtype.voidType));
    Xobject fence = Xcons.functionCall(fenceId, null);
    KKSBlockList.add(Bcons.Statement(fence));

    Block externBlock = Bcons.COMPOUND(KKSBlockList);
    
    // Finish
    
    XobjectDef kksFunc = XobjectDef.Func(KKSFuncId, KKSFuncParams, null, externBlock.toXobject());
      
    XmOption.setLanguage(XmLanguage.F);

    return kksFunc;
  }

  
  public BlockList F2CPP_loopBody(BlockList loopBody, XobjList reductionList){
    Xobject loopObject = loopBody.toXobject();
    loopObject = F2CPP_Xobject(loopObject, reductionList);
    return Bcons.buildList(loopObject);
  }

  
  public Xobject F2CPP_Xobject(Xobject x, XobjList reductionList){

    Xobject xx = null;

    if (x != null){
      switch (x.Opcode()){

      case F_STATEMENT_LIST:
      case LIST:
	xx = Xcons.List();
	for (Xobject s : (XobjList)x){
	  xx.add(F2CPP_Xobject(s, reductionList));
	}
	break;
	  
      case F_ASSIGN_STATEMENT: {
	xx = Xcons.List(Xcode.EXPR_STATEMENT, Xcons.Set(F2CPP_Xobject(x.getArg(0), reductionList),
							F2CPP_Xobject(x.getArg(1), reductionList)));
	break;
      }
      
      case PLUS_EXPR:
      case MINUS_EXPR:
      case MUL_EXPR:
      case DIV_EXPR:
	xx = Xcons.binaryOp(x.Opcode(), F2CPP_Xobject(x.left(), reductionList),
			                F2CPP_Xobject(x.right(), reductionList));
	break;
	
      case INT_CONSTANT:
      case FLOAT_CONSTANT:
      case LONG_CONSTANT:
	xx = x.copy();
	break;
	  
      case F_VAR_REF: {
	Xobject var = x.getArg(0);
	Ident id = Ident.Local(var.getSym(), x.Type());
	// if (id.getName().equals("i") || id.getName().equals("j")){
	//   i.setXobject(Xcons.binaryOp(Xcode.MINUS_EXPR, Xcons.SymbolRef(id), Xcons.IntConstant(1)));
	// }
	// else {
	xx = Xcons.SymbolRef(id);
	// }
	break;
      }
	  
      case F_ARRAY_REF: {
	XobjList orig_indices = (XobjList)x.getArg(1);
	XobjList new_indices = Xcons.List();
	FarrayType array_type = (FarrayType)x.getArg(0).Type();
	for (int i = 0; i < array_type.getNumDimensions(); i++){
	  Xobject idx = orig_indices.getArg(i).getArg(0);

	  Xobject lb = null;
	  if (array_type.isFfixedShape()){
	    lb = array_type.getLbound(i, null);
	  }
	  else {
	    Ident id_lb = _KKSFuncParams.find(x.getArg(0).getArg(0).getSym() + "_lb", IXobject.FINDKIND_VAR);
	    Xobject array_lb = Xcons.SymbolRef(id_lb);
	    lb = Xcons.arrayRef(Xtype.intType, array_lb, Xcons.List(Xcons.IntConstant(i)));
	  }

	  idx = Xcons.binaryOp(Xcode.MINUS_EXPR, F2CPP_Xobject(idx, reductionList), lb);
	  new_indices.add(idx);
	}

	xx = Xcons.arrayRef(x.Type(), F2CPP_Xobject(x.getArg(0), reductionList), new_indices);

	break;
      }

      case F_ARRAY_INDEX:
	xx = F2CPP_Xobject(x.getArg(0), reductionList);
	break;

      case VAR:

	// must be fixed.
	if (reductionList != null &&
	    x.getSym().equals(reductionList.getArg(1).getArg(0).getArg(0).getSym())){
	  //Ident id = Ident.Local("l_" + x.getSym(), Xtype.Pointer(x.Type()));
	  //xx = Xcons.PointerRef(Xcons.SymbolRef(id));
	  Ident id = Ident.Local("l_" + x.getSym(), x.Type());
	  xx = Xcons.SymbolRef(id);
	}
	// must be fixed.
	else if (x.Type().equals(Xtype.intType)){
	  xx = x.copy();
	}
	else {
	  Ident id = Ident.Local(x.getSym(), Xtype.Pointer(x.Type()));
	  xx = Xcons.PointerRef(Xcons.SymbolRef(id));
	}

	break;

      case FUNCTION_CALL: {
	String fname = x.left().getSym();
	switch (fname){
	case "dble":
	  xx = Xcons.Cast(Xtype.floatType, F2CPP_Xobject(x.right(), reductionList));
	  return xx;
	case "dabs":
	  fname = "Kokkos::fabs";
	  break;
	case "sin":
	  fname = "Kokkos::sin";
	  break;
	case "cos":
	  fname = "Kokkos::cos";
	  break;
	}
	Xobject args = F2CPP_Xobject(x.right(), reductionList);
	xx = Xcons.functionCall(Ident.Local(fname, Xtype.Function(F2CPP_type(x.Type()))), args);
	break;
      }
	  
      default:
	XMP.fatal("not supported by F2Kokkos: " + x);
	break;
	  
	//case MOD_EXPR:
      }
    }
    
    return xx;
    
  }


  public Xtype F2CPP_type(Xtype t){

    if (t != null){
      switch (t.getBasicType()){

      case BasicType.FLOAT:
	return Xtype.floatType;
	  
      default:
	XMP.fatal("not supported type by F2Kokkos: " + t);
	break;
      }
    }
    
    return t;
    
  }


  public void decompile(XobjectDef kksFunc){
    
    out.println("extern \"C\" {");
    out.println();
    
    // decompile the Kokkos function.
    out.printKKSfunc(kksFunc, (Ident)kksFunc.getNameObj());
    out.println();

    out.println("}"); // end of extern "C"
    out.println();
    
    out.flush();
    
  }

  // private static void addLocalVar(Ident id, XMPpair<XobjList, XobjList> vars) {
  //   vars.getFirst().add(id);
  //   vars.getSecond().add(Xcons.List(Xcode.VAR_DECL, id, null, null));
  // }

  // private static Block createFuncCallBlock(String funcName, XobjList funcArgs) {
  //   Ident funcId = XMP.getMacroId(funcName);
  //   return Bcons.Statement(funcId.Call(funcArgs));
  // }

  // private static XMPpair<XobjList, XobjList> genDeviceFuncParamArgs(XobjList paramIdList, XobjList mapThreads) {
  //   XobjList funcParams = Xcons.List();
  //   XobjList funcArgs = Xcons.List();
  //   for (XobjArgs i = paramIdList.getArgs(); i != null; i = i.nextArgs()) {
  //     Ident id = (Ident)i.getArg();

  //     funcParams.add(id);

  //     if (id.Type().isArray()) {
  //       funcArgs.add(id.getValue());
  //     } else {
  //       funcArgs.add(id.Ref());
  //     }
  //   }

  //   if (mapThreads == null) {
  //     Ident totalIterId = Ident.Param("_XMP_GPU_TOTAL_ITER", Xtype.unsignedlonglongType);
  //     funcParams.add(totalIterId);
  //     funcArgs.add(totalIterId.Ref());
  //   }

  //   return new XMPpair<XobjList, XobjList>(funcParams, funcArgs);
  // }

  // private static XobjList getNumThreads(XobjList gpuClause) {
  //   XobjList mapThreads = null;

  //   for (Xobject c : gpuClause) {
  //     XMPpragma p = XMPpragma.valueOf(c.getArg(0));
  //     switch (p) {
  //       case GPU_MAP_THREADS:
  //         mapThreads = (XobjList)c.getArg(1);
  //       default:
  //     }
  //   }

  //   return mapThreads;
  // }

  // private static boolean hasVarRef(String varName, CforBlock loopBlock) throws XMPexception {
  //   BasicBlockExprIterator iter = new BasicBlockExprIterator(loopBlock.getBody());
  //   for (iter.init(); !iter.end(); iter.next()) {
  //     Xobject expr = iter.getExpr();
  //     if (expr == null) {
  //       continue;
  //     }

  //     bottomupXobjectIterator myIter = new bottomupXobjectIterator(expr);
  //     for (myIter.init(); !myIter.end(); myIter.next()) {
  //       Xobject myExpr = myIter.getXobject();
  //       if (myExpr == null) {
  //         continue;
  //       }

  //       if (myExpr.Opcode() == Xcode.VAR) {
  //         if (myExpr.getName().equals(varName)) {
  //           return true;
  //         }
  //       }
  //     }
  //   }

  //   return false;
  // }

  // // FIXME localVars, localDecls delete
  // private static void rewriteLoopBody(CforBlock loopBlock) throws XMPexception {
  //   // rewrite declarations
  //   rewriteDecls(loopBlock);

  //   // rewrite loop
  //   BasicBlockExprIterator iter = new BasicBlockExprIterator(loopBlock.getBody());
  //   for (iter.init(); !iter.end(); iter.next()) {
  //     rewriteExpr(iter.getExpr(), loopBlock);
  //   }
  // }

  // private static void rewriteDecls(CforBlock loopBlock) {
  //   topdownBlockIterator iter = new topdownBlockIterator(loopBlock);
  //   for (iter.init(); !iter.end(); iter.next()) {
  //     Block b = iter.getBlock();
  //     BlockList bl = b.getBody();

  //     if (bl != null) {
  //       XobjList decls = (XobjList)bl.getDecls();
  //       if (decls != null) {
  //         try {
  //           for (Xobject x : decls) {
  //             rewriteExpr(x.getArg(1), loopBlock);
  //           }
  //         } catch (XMPexception e) {
  //           XMP.error(b.getLineNo(), e.getMessage());
  //         }
  //       }
  //     }
  //   }
  // }

  // private static void rewriteExpr(Xobject expr, CforBlock loopBlock) throws XMPexception {
  //   if (expr == null) return;

  //   topdownXobjectIterator iter = new topdownXobjectIterator(expr);
  //   for (iter.init(); !iter.end(); iter.next()) {
  //     Xobject myExpr = iter.getXobject();
  //     if (myExpr == null) {
  //       continue;
  //     }

  //     switch (myExpr.Opcode()) {
  //       case ARRAY_REF:
  //         {
  //           String varName = myExpr.getArg(0).getSym();
  //           XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(varName, loopBlock);
  //           if (gpuData == null) {
  //             throw new XMPexception("array '" + varName + "' is not allocated on the device memory");
  //           }

  //           XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
  //           if (alignedArray != null) {
  //             if (alignedArray.realloc()) {
  //               iter.setXobject(rewriteAlignedArrayExpr((XobjList)myExpr.getArg(1), gpuData));
  //             }
  //           }
  //         } break;
  //       default:
  //     }
  //   }
  // }

  // private static Xobject rewriteAlignedArrayExpr(XobjList refExprList,
  //                                                XMPgpuData gpuData) throws XMPexception {
  //   int arrayDimCount = 0;
  //   XobjList args = Xcons.List(gpuData.getHostId().getAddr());
  //   if (refExprList != null) {
  //     for (Xobject x : refExprList) {
  //       args.add(x); //args.add(getCalcIndexFuncRef(gpuData, arrayDimCount, x));
  //       arrayDimCount++;
  //     }
  //   }

  //   return createRewriteAlignedArrayFunc(gpuData, arrayDimCount, args);
  // }

  // private static Xobject createRewriteAlignedArrayFunc(XMPgpuData gpuData, int arrayDimCount,
  //                                                      XobjList getAddrFuncArgs) throws XMPexception {
  //   XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
  //   int arrayDim = alignedArray.getDim();
  //   XobjList accIdList = _accIdHash.get(alignedArray.getName());
  //   Ident getAddrFuncId = null;

  //   if (arrayDim < arrayDimCount) {
  //     throw new XMPexception("wrong array ref");
  //   } else if (arrayDim == arrayDimCount) {
  //     getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_E_" + arrayDim, Xtype.Pointer(alignedArray.getType()));
  //     for (int i = 0; i < arrayDim - 1; i++) {
  //       getAddrFuncArgs.add(((Ident)(accIdList.getArg(i))).Ref());
  //     }
  //   } else {
  //     getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_" + arrayDimCount, Xtype.Pointer(alignedArray.getType()));
  //     for (int i = 0; i < arrayDimCount; i++) {
  //       getAddrFuncArgs.add(((Ident)(accIdList.getArg(i))).Ref());
  //     }
  //   }

  //   Xobject retObj = getAddrFuncId.Call(getAddrFuncArgs);
  //   if (arrayDim == arrayDimCount) {
  //     return Xcons.PointerRef(retObj);
  //   } else {
  //     return retObj;
  //   }
  // }

  // private static Xobject getCalcIndexFuncRef(XMPgpuData gpuData, int index, Xobject indexRef) throws XMPexception {
  //   XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
  //   XobjList gtolIdList = _gtolIdHash.get(alignedArray.getName());

  //   switch (alignedArray.getAlignMannerAt(index)) {
  //     case XMPalignedArray.NOT_ALIGNED:
  //     case XMPalignedArray.DUPLICATION:
  //       return indexRef;
  //     case XMPalignedArray.BLOCK:
  //       if (alignedArray.hasShadow()) {
  //         XMPshadow shadow = alignedArray.getShadowAt(index);
  //         switch (shadow.getType()) {
  //           case XMPshadow.SHADOW_NONE:
  //           case XMPshadow.SHADOW_NORMAL:
  //             {
  //               XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
  //               return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
  //             }
  //           case XMPshadow.SHADOW_FULL:
  //             return indexRef;
  //           default:
  //             throw new XMPexception("unknown shadow type");
  //         }
  //       } else {
  //         XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
  //         return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
  //       }
  //     case XMPalignedArray.CYCLIC:
  //       if (alignedArray.hasShadow()) {
  //         XMPshadow shadow = alignedArray.getShadowAt(index);
  //         switch (shadow.getType()) {
  //           case XMPshadow.SHADOW_NONE:
  //             {
  //               XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
  //               return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
  //             }
  //           case XMPshadow.SHADOW_FULL:
  //             return indexRef;
  //           case XMPshadow.SHADOW_NORMAL:
  //             throw new XMPexception("only block distribution allows shadow");
  //           default:
  //             throw new XMPexception("unknown shadow type");
  //         }
  //       } else {
  //         XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
  //         return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
  //       }
  //     default:
  //       throw new XMPexception("unknown align manner for array '" + alignedArray.getName()  + "'");
  //   }
  // }

}
