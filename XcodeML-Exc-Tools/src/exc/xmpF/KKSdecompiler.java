package exc.xmpF;

import exc.xcalablemp.XMPexception;
import exc.block.*;
import exc.object.*;
import java.io.*;
import java.util.*;

public class KKSdecompiler {
    
  public static final String GPU_FUNC_CONF = "XCALABLEMP_GPU_FUNC_CONF_PROP";
  public static final String GPU_INDEX_TABLE = "XCALABLEMP_GPU_INDEX_TABLE_PROP";

  private static KKSdecompileWriter out = null;
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = "_kks.cc";

  private static HashMap<String, XobjList> _gtolIdHash = null;
  private static HashMap<String, XobjList> _accIdHash = null;

  public static void decompile(XobjectDef kksFunc, XobjectFile env){
    
    try {
      if (out == null) {
        Writer w = new BufferedWriter(new FileWriter(getSrcName(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
        out = new KKSdecompileWriter(w, env);

	// add header include line
	out.println("# include <Kokkos_Core.hpp>");
	out.println("# include \"flcl-cxx.hpp\"");
	out.println();
      }

      out.println("extern \"C\" {");
      out.println();
    
      // decompile the Kokkos function.
      out.printKKSfunc(kksFunc, (Ident)kksFunc.getNameObj());
      out.println();

      out.println("}"); // end of extern "C"
      out.println();
    
      out.flush();
    } catch (IOException e) {
      //throw new XMPexception("error in gpu decompiler: " + e.getMessage());
    }
    
  }

  // private static void addLocalVar(Ident id, XMPpair<XobjList, XobjList> vars) {
  //   vars.getFirst().add(id);
  //   vars.getSecond().add(Xcons.List(Xcode.VAR_DECL, id, null, null));
  // }

  // private static Block createFuncCallBlock(String funcName, XobjList funcArgs) {
  //   Ident funcId = XMP.getMacroId(funcName);
  //   return Bcons.Statement(funcId.Call(funcArgs));
  // }

  private static String getSrcName(String srcName) {
    String name = "";
    String[] buffer = srcName.split("\\.");
    for (int i = 0; i < buffer.length - 1; i++) {
      name += buffer[i];
    }

    return name;
  }

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
