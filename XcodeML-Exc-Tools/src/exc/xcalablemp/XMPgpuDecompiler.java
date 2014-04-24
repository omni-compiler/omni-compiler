/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.io.*;
import java.util.*;

public class XMPgpuDecompiler {
  public static final String GPU_FUNC_CONF = "XCALABLEMP_GPU_FUNC_CONF_PROP";
  public static final String GPU_INDEX_TABLE = "XCALABLEMP_GPU_INDEX_TABLE_PROP";

  private static XMPgpuDecompileWriter out = null;
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = ".cu";

  private static HashMap<String, XobjList> _gtolIdHash = null;
  private static HashMap<String, XobjList> _accIdHash = null;

  public static void decompile(Ident hostFuncId, XobjList paramIdList, ArrayList<XMPalignedArray> alignedArrayList,
                               CforBlock loopBlock, XobjList loopIndexList, PragmaBlock pb, XobjectFile env) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();
    XobjList gpuClause = (XobjList)loopDecl.getArg(3).getArg(1);

    XMPpair<XobjList, XobjList> localVars = new XMPpair<XobjList, XobjList>(Xcons.List(), Xcons.List());

    _gtolIdHash = new HashMap<String, XobjList>();
    _accIdHash = new HashMap<String, XobjList>();
    for (XMPalignedArray alignedArray : alignedArrayList) {
      int dim = alignedArray.getDim();
      String alignedArrayName = alignedArray.getName();

      XobjList gtolIdList = Xcons.List();
      XobjList accIdList = Xcons.List();

      for (int i = 0; i < dim; i++) {
        Ident gtolId = Ident.Local(new String("_XMP_GPU_" + alignedArrayName + "_GTOL_" + i), Xtype.intType);
        gtolIdList.add(gtolId);

        if (i != (dim - 1)) {
          Ident accId = Ident.Local(new String("_XMP_GPU_" + alignedArrayName + "_ACC_" + i), Xtype.unsignedlonglongType);
          accIdList.add(accId);
        }
      }

      _gtolIdHash.put(alignedArrayName, gtolIdList);
      _accIdHash.put(alignedArrayName, accIdList);
    }

    // schedule iteration
    XobjList loopIndexRefList = Xcons.List();
    XobjList loopIndexAddrList = Xcons.List();
    XobjList loopIterRefList = Xcons.List();
    Iterator<Xobject> iter = loopIndexList.iterator();
    while(iter.hasNext()) {
      Ident loopVarId = (Ident)iter.next();
      String loopVarName = loopVarId.getSym();

      addLocalVar(loopVarId, localVars);

      XobjList loopIter = XMPutil.getLoopIter(loopBlock, loopVarName);

      Ident initId = (Ident)loopIter.getArg(0);
      Ident condId = (Ident)loopIter.getArg(1);
      Ident stepId = (Ident)loopIter.getArg(2);

      loopIndexRefList.add(loopVarId.Ref());
      loopIndexAddrList.add(loopVarId.getAddr());
      loopIterRefList.add(initId.Ref());
      loopIterRefList.add(condId.Ref());
      loopIterRefList.add(stepId.Ref());
    }

    // generate wrapping function
    Ident blockXid = Ident.Local("_XMP_GPU_DIM3_block_x", Xtype.intType);
    Ident blockYid = Ident.Local("_XMP_GPU_DIM3_block_y", Xtype.intType);
    Ident blockZid = Ident.Local("_XMP_GPU_DIM3_block_z", Xtype.intType);
    Ident threadXid = Ident.Local("_XMP_GPU_DIM3_thread_x", Xtype.intType);
    Ident threadYid = Ident.Local("_XMP_GPU_DIM3_thread_y", Xtype.intType);
    Ident threadZid = Ident.Local("_XMP_GPU_DIM3_thread_z", Xtype.intType);
    Ident totalIterId = null;

    XobjList hostBodyIdList = Xcons.List(blockXid, blockYid, blockZid, threadXid, threadYid, threadZid);
    XobjList hostBodyDecls = Xcons.List(Xcons.List(Xcode.VAR_DECL, blockXid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, blockYid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, blockZid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, threadXid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, threadYid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, threadZid, null, null));

    XobjList mapThreads = getNumThreads(gpuClause);
    Xobject confParamFuncCall = null;
    { // create confParamFuncCall, totalIterId (if mapThreads == null)
      String confParamFuncName = null;
      if (mapThreads == null) {
        totalIterId = Ident.Local("_XMP_GPU_TOTAL_ITER", Xtype.unsignedlonglongType);
        hostBodyIdList.add(totalIterId);
        hostBodyDecls.add(Xcons.List(Xcode.VAR_DECL, totalIterId, null, null));

        confParamFuncName = new String("_XMP_gpu_calc_config_params");
        Ident confParamFuncId = XMP.getMacroId(confParamFuncName);
        XobjList confParamFuncArgs = Xcons.List(totalIterId.getAddr(),
                                                blockXid.getAddr(), blockYid.getAddr(), blockZid.getAddr(),
                                                threadXid.getAddr(), threadYid.getAddr(), threadZid.getAddr());
        confParamFuncArgs.mergeList(loopIterRefList);

        confParamFuncCall = Xcons.List(Xcode.EXPR_STATEMENT, confParamFuncId.Call(confParamFuncArgs));
      } else {
        confParamFuncName = new String("_XMP_gpu_calc_config_params_MAP_THREADS");
        Ident confParamFuncId = XMP.getMacroId(confParamFuncName);
        XobjList confParamFuncArgs = Xcons.List(blockXid.getAddr(), blockYid.getAddr(), blockZid.getAddr(),
                                                threadXid.getAddr(), threadYid.getAddr(), threadZid.getAddr());
        for (Xobject mapList : mapThreads) {
          if (mapList != null) {
            confParamFuncArgs.add(((XobjList)mapList).getArg(1));
          }
        }

        confParamFuncArgs.mergeList(loopIterRefList);

        confParamFuncCall = Xcons.List(Xcode.EXPR_STATEMENT, confParamFuncId.Call(confParamFuncArgs));
      }
    }

    // create device function
    XMPpair<XobjList, XobjList> deviceFuncParamArgs = genDeviceFuncParamArgs(paramIdList, mapThreads);
    XobjList deviceFuncParams = deviceFuncParamArgs.getFirst();
    XobjList deviceFuncArgs = deviceFuncParamArgs.getSecond();

    Ident deviceFuncId = XMP.getMacroId(hostFuncId.getName() + "_DEVICE");
    ((FunctionType)deviceFuncId.Type()).setFuncParamIdList(deviceFuncParams);
    Xobject deviceFuncCall = deviceFuncId.Call(deviceFuncArgs);
    deviceFuncCall.setProp(GPU_FUNC_CONF,
                           (Object)Xcons.List(blockXid, blockYid, blockZid,
                                              threadXid, threadYid, threadZid));

    // rewrite loop statement
    rewriteLoopBody(loopBlock);

    ArrayList<XobjList> gtolFuncArgsList = new ArrayList<XobjList>();
    ArrayList<XobjList> accFuncArgsList = new ArrayList<XobjList>();
    for (XMPalignedArray alignedArray : alignedArrayList) {
      int index;
      String alignedArrayName = alignedArray.getName();

      XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(alignedArrayName, loopBlock);
      Xobject deviceDescRef = gpuData.getDeviceDescId().Ref();

      XobjList gtolIdList = _gtolIdHash.get(alignedArrayName);
      index = 0;
      for (Xobject x : gtolIdList) {
        Ident id = (Ident)x;
        if (hasVarRef(id.getName(), loopBlock)) {
          addLocalVar(id, localVars);
          gtolFuncArgsList.add(Xcons.List(id.Ref(), deviceDescRef, Xcons.IntConstant(index)));
        }

        index++;
      }

      XobjList accIdList = _accIdHash.get(alignedArrayName);
      index = 0;
      for (Xobject x : accIdList) {
        Ident id = (Ident)x;
        if (hasVarRef(id.getName(), loopBlock)) {
          addLocalVar(id, localVars);
          accFuncArgsList.add(Xcons.List(id.Ref(), deviceDescRef, Xcons.IntConstant(index)));
        }

        index++;
      }
    }

    // create Defs
    Xobject hostBodyObj = Xcons.CompoundStatement(hostBodyIdList, hostBodyDecls, Xcons.List(confParamFuncCall, deviceFuncCall));
    ((FunctionType)hostFuncId.Type()).setFuncParamIdList(paramIdList);
    XobjectDef hostDef = XobjectDef.Func(hostFuncId, paramIdList, null, hostBodyObj);

    Ident threadNumId = null;
    if (mapThreads == null) {
      threadNumId = Ident.Local("_XMP_GPU_THREAD_ID", Xtype.unsignedlonglongType);
      addLocalVar(threadNumId, localVars);
    }

    BlockList newLoopBlockList = Bcons.emptyBody(localVars.getFirst(), localVars.getSecond());
    for (XobjList args : gtolFuncArgsList) {
      newLoopBlockList.add(createFuncCallBlock("_XMP_GPU_M_GET_ARRAY_GTOL", args));
    }
    for (XobjList args : accFuncArgsList) {
      newLoopBlockList.add(createFuncCallBlock("_XMP_GPU_M_GET_ARRAY_ACC", args));
    }

    // add: function mapping iteration space, create: kernel func body
    Xobject kernelFuncObj = null;
    if (mapThreads == null) {
      newLoopBlockList.add(createFuncCallBlock("_XMP_gpu_calc_thread_id", Xcons.List(threadNumId.getAddr())));

      XobjList calcIterFuncArgs = Xcons.List(threadNumId.Ref());
      calcIterFuncArgs.mergeList(loopIterRefList);
      calcIterFuncArgs.mergeList(loopIndexAddrList);
      newLoopBlockList.add(createFuncCallBlock("_XMP_gpu_calc_iter", calcIterFuncArgs));

      kernelFuncObj = Xcons.List(Xcode.IF_STATEMENT,
                                 Xcons.binaryOp(Xcode.LOG_LT_EXPR, threadNumId.Ref(), totalIterId.Ref()),
                                                loopBlock.getBody().toXobject(), null);
    } else {
      int mapDim = 0;
      for (Xobject mapList : mapThreads) {
        if (mapList != null) {
          mapDim++;
        }
      }

      XobjList calcIterFuncArgs = Xcons.List();
      calcIterFuncArgs.mergeList(loopIterRefList);
      calcIterFuncArgs.mergeList(loopIndexRefList);
      newLoopBlockList.add(createFuncCallBlock(new String("_XMP_gpu_calc_iter_MAP_THREADS_" + mapDim),
                                               calcIterFuncArgs));

      kernelFuncObj = loopBlock.getBody().toXobject();
    }

    // add: kernel func body
    newLoopBlockList.add(kernelFuncObj);

    Xobject deviceBodyObj = newLoopBlockList.toXobject();
    XobjectDef deviceDef = XobjectDef.Func(deviceFuncId, deviceFuncParams, null, deviceBodyObj);

    try {
      if (out == null) {
        Writer w = new BufferedWriter(new FileWriter(getSrcName(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
        out = new XMPgpuDecompileWriter(w, env);
      }

      // add header include line
      out.println("#include \"xmp_gpu_func.hpp\"");
      out.println("#include \"xmp_index_macro.h\"");
      out.println();

      // decompile device function
      out.printDeviceFunc(deviceDef, deviceFuncId);
      out.println();

      // decompie wrapping function
      out.printHostFunc(hostDef);
      out.println();

      out.flush();
    } catch (IOException e) {
      throw new XMPexception("error in gpu decompiler: " + e.getMessage());
    }
  }

  private static void addLocalVar(Ident id, XMPpair<XobjList, XobjList> vars) {
    vars.getFirst().add(id);
    vars.getSecond().add(Xcons.List(Xcode.VAR_DECL, id, null, null));
  }

  private static Block createFuncCallBlock(String funcName, XobjList funcArgs) {
    Ident funcId = XMP.getMacroId(funcName);
    return Bcons.Statement(funcId.Call(funcArgs));
  }

  private static String getSrcName(String srcName) {
    String name = "";
    String[] buffer = srcName.split("\\.");
    for (int i = 0; i < buffer.length - 1; i++) {
      name += buffer[i];
    }

    return name;
  }

  private static XMPpair<XobjList, XobjList> genDeviceFuncParamArgs(XobjList paramIdList, XobjList mapThreads) {
    XobjList funcParams = Xcons.List();
    XobjList funcArgs = Xcons.List();
    for (XobjArgs i = paramIdList.getArgs(); i != null; i = i.nextArgs()) {
      Ident id = (Ident)i.getArg();

      funcParams.add(id);

      if (id.Type().isArray()) {
        funcArgs.add(id.getValue());
      } else {
        funcArgs.add(id.Ref());
      }
    }

    if (mapThreads == null) {
      Ident totalIterId = Ident.Param("_XMP_GPU_TOTAL_ITER", Xtype.unsignedlonglongType);
      funcParams.add(totalIterId);
      funcArgs.add(totalIterId.Ref());
    }

    return new XMPpair<XobjList, XobjList>(funcParams, funcArgs);
  }

  private static XobjList getNumThreads(XobjList gpuClause) {
    XobjList mapThreads = null;

    for (Xobject c : gpuClause) {
      XMPpragma p = XMPpragma.valueOf(c.getArg(0));
      switch (p) {
        case GPU_MAP_THREADS:
          mapThreads = (XobjList)c.getArg(1);
        default:
      }
    }

    return mapThreads;
  }

  private static boolean hasVarRef(String varName, CforBlock loopBlock) throws XMPexception {
    BasicBlockExprIterator iter = new BasicBlockExprIterator(loopBlock.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      if (expr == null) {
        continue;
      }

      bottomupXobjectIterator myIter = new bottomupXobjectIterator(expr);
      for (myIter.init(); !myIter.end(); myIter.next()) {
        Xobject myExpr = myIter.getXobject();
        if (myExpr == null) {
          continue;
        }

        if (myExpr.Opcode() == Xcode.VAR) {
          if (myExpr.getName().equals(varName)) {
            return true;
          }
        }
      }
    }

    return false;
  }

  // FIXME localVars, localDecls delete
  private static void rewriteLoopBody(CforBlock loopBlock) throws XMPexception {
    // rewrite declarations
    rewriteDecls(loopBlock);

    // rewrite loop
    BasicBlockExprIterator iter = new BasicBlockExprIterator(loopBlock.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      rewriteExpr(iter.getExpr(), loopBlock);
    }
  }

  private static void rewriteDecls(CforBlock loopBlock) {
    topdownBlockIterator iter = new topdownBlockIterator(loopBlock);
    for (iter.init(); !iter.end(); iter.next()) {
      Block b = iter.getBlock();
      BlockList bl = b.getBody();

      if (bl != null) {
        XobjList decls = (XobjList)bl.getDecls();
        if (decls != null) {
          try {
            for (Xobject x : decls) {
              rewriteExpr(x.getArg(1), loopBlock);
            }
          } catch (XMPexception e) {
            XMP.error(b.getLineNo(), e.getMessage());
          }
        }
      }
    }
  }

  private static void rewriteExpr(Xobject expr, CforBlock loopBlock) throws XMPexception {
    if (expr == null) return;

    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            String varName = myExpr.getArg(0).getSym();
            XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(varName, loopBlock);
            if (gpuData == null) {
              throw new XMPexception("array '" + varName + "' is not allocated on the device memory");
            }

            XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
            if (alignedArray != null) {
              if (alignedArray.realloc()) {
                iter.setXobject(rewriteAlignedArrayExpr((XobjList)myExpr.getArg(1), gpuData));
              }
            }
          } break;
        default:
      }
    }
  }

  private static Xobject rewriteAlignedArrayExpr(XobjList refExprList,
                                                 XMPgpuData gpuData) throws XMPexception {
    int arrayDimCount = 0;
    XobjList args = Xcons.List(gpuData.getHostId().getAddr());
    if (refExprList != null) {
      for (Xobject x : refExprList) {
        args.add(x); //args.add(getCalcIndexFuncRef(gpuData, arrayDimCount, x));
        arrayDimCount++;
      }
    }

    return createRewriteAlignedArrayFunc(gpuData, arrayDimCount, args);
  }

  private static Xobject createRewriteAlignedArrayFunc(XMPgpuData gpuData, int arrayDimCount,
                                                       XobjList getAddrFuncArgs) throws XMPexception {
    XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
    int arrayDim = alignedArray.getDim();
    XobjList accIdList = _accIdHash.get(alignedArray.getName());
    Ident getAddrFuncId = null;

    if (arrayDim < arrayDimCount) {
      throw new XMPexception("wrong array ref");
    } else if (arrayDim == arrayDimCount) {
      getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_E_" + arrayDim, Xtype.Pointer(alignedArray.getType()));
      for (int i = 0; i < arrayDim - 1; i++) {
        getAddrFuncArgs.add(((Ident)(accIdList.getArg(i))).Ref());
      }
    } else {
      getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_" + arrayDimCount, Xtype.Pointer(alignedArray.getType()));
      for (int i = 0; i < arrayDimCount; i++) {
        getAddrFuncArgs.add(((Ident)(accIdList.getArg(i))).Ref());
      }
    }

    Xobject retObj = getAddrFuncId.Call(getAddrFuncArgs);
    if (arrayDim == arrayDimCount) {
      return Xcons.PointerRef(retObj);
    } else {
      return retObj;
    }
  }

  private static Xobject getCalcIndexFuncRef(XMPgpuData gpuData, int index, Xobject indexRef) throws XMPexception {
    XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
    XobjList gtolIdList = _gtolIdHash.get(alignedArray.getName());

    switch (alignedArray.getAlignMannerAt(index)) {
      case XMPalignedArray.NOT_ALIGNED:
      case XMPalignedArray.DUPLICATION:
        return indexRef;
      case XMPalignedArray.BLOCK:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
            case XMPshadow.SHADOW_NORMAL:
              {
                XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
                return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            default:
              throw new XMPexception("unknown shadow type");
          }
        } else {
          XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
          return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
        }
      case XMPalignedArray.CYCLIC:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
              {
                XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
                return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            case XMPshadow.SHADOW_NORMAL:
              throw new XMPexception("only block distribution allows shadow");
            default:
              throw new XMPexception("unknown shadow type");
          }
        } else {
          XobjList args = Xcons.List(indexRef, ((Ident)(gtolIdList.getArg(index))).Ref());
          return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
        }
      default:
        throw new XMPexception("unknown align manner for array '" + alignedArray.getName()  + "'");
    }
  }
}
