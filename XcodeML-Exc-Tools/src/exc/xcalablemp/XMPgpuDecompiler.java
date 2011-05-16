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

  public static void decompile(Ident hostFuncId, XobjList paramIdList,
                               CforBlock loopBlock, PragmaBlock pb, XobjectFile env) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();
    XobjList gpuClause = (XobjList)loopDecl.getArg(3).getArg(1);

    // FIXME decl & use thread id (local var)
    XobjList localVars = Xcons.List();
    XobjList localDecls = Xcons.List();

    // schedule iteration
    XobjList loopIndexList = (XobjList)loopDecl.getArg(0);
    XobjList loopIndexAddrList = Xcons.List();
    XobjList loopIterRefList = Xcons.List();
    Iterator<Xobject> iter = loopIndexList.iterator();
    while(iter.hasNext()) {
      String loopVarName = iter.next().getString();
      Ident loopVarId = loopBlock.findVarIdent(loopVarName);

      localVars.add(loopVarId);
      localDecls.add(Xcons.List(Xcode.VAR_DECL, loopVarId, null, null));

      XobjList loopIter = XMPutil.getLoopIter(loopBlock, loopVarName);

      Ident initId = (Ident)loopIter.getArg(0);
      Ident condId = (Ident)loopIter.getArg(1);
      Ident stepId = (Ident)loopIter.getArg(2);

      // FIXME consider array dimension
      loopIndexAddrList.add(loopVarId.getAddr());
      loopIterRefList.add(initId.Ref());
      loopIterRefList.add(condId.Ref());
      loopIterRefList.add(stepId.Ref());
    }

    XMPpair<XobjList, XobjList> deviceFuncParamArgs = genDeviceFuncParamArgs(paramIdList);
    XobjList deviceFuncParams = deviceFuncParamArgs.getFirst();
    XobjList deviceFuncArgs = deviceFuncParamArgs.getSecond();

    // generate wrapping function
    Ident blockXid = Ident.Local("_XMP_GPU_DIM3_block_x", Xtype.intType);
    Ident blockYid = Ident.Local("_XMP_GPU_DIM3_block_y", Xtype.intType);
    Ident blockZid = Ident.Local("_XMP_GPU_DIM3_block_z", Xtype.intType);
    Ident threadXid = Ident.Local("_XMP_GPU_DIM3_thread_x", Xtype.intType);
    Ident threadYid = Ident.Local("_XMP_GPU_DIM3_thread_y", Xtype.intType);
    Ident threadZid = Ident.Local("_XMP_GPU_DIM3_thread_z", Xtype.intType);

    String confParamFuncName = null;
    XobjList numThreads = getNumThreads(gpuClause);
    if (numThreads == null) {
      confParamFuncName = new String("_XMP_gpu_calc_config_params");
    } else {
      confParamFuncName = new String("_XMP_gpu_calc_config_params_NUM_THREADS");
    }

    Ident totalIterId = Ident.Local("_XMP_GPU_TOTAL_ITER", Xtype.unsignedlonglongType);
    Ident confParamFuncId = XMP.getMacroId(confParamFuncName);
    XobjList confParamFuncArgs = Xcons.List(totalIterId.getAddr(),
                                            blockXid.getAddr(), blockYid.getAddr(), blockZid.getAddr(),
                                            threadXid.getAddr(), threadYid.getAddr(), threadZid.getAddr());
    XMPutil.mergeLists(confParamFuncArgs, numThreads);
    XMPutil.mergeLists(confParamFuncArgs, loopIterRefList);
    Xobject confParamFuncCall = confParamFuncId.Call(confParamFuncArgs);

    Ident deviceFuncId = XMP.getMacroId(hostFuncId.getName() + "_DEVICE");
    ((FunctionType)deviceFuncId.Type()).setFuncParamIdList(deviceFuncParams);
    Xobject deviceFuncCall = deviceFuncId.Call(deviceFuncArgs);
    deviceFuncCall.setProp(GPU_FUNC_CONF,
                           (Object)Xcons.List(blockXid, blockYid, blockZid,
                                              threadXid, threadYid, threadZid));

    XobjList hostBodyIdList = Xcons.List(totalIterId, blockXid, blockYid, blockZid, threadXid, threadYid, threadZid);
    XobjList hostBodyDecls = Xcons.List(Xcons.List(Xcode.VAR_DECL, totalIterId, null, null),
                                        Xcons.List(Xcode.VAR_DECL, blockXid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, blockYid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, blockZid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, threadXid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, threadYid, null, null),
                                        Xcons.List(Xcode.VAR_DECL, threadZid, null, null));

    // create Defs
    Xobject hostBodyObj = Xcons.CompoundStatement(hostBodyIdList, hostBodyDecls, Xcons.List(confParamFuncCall, deviceFuncCall));
    ((FunctionType)hostFuncId.Type()).setFuncParamIdList(paramIdList);
    XobjectDef hostDef = XobjectDef.Func(hostFuncId, paramIdList, null, hostBodyObj);

    Ident threadNumId = Ident.Local("_XMP_GPU_THREAD_ID", Xtype.unsignedlonglongType);
    localVars.add(threadNumId);
    localDecls.add(Xcons.List(Xcode.VAR_DECL, threadNumId, null, null));
    BlockList newLoopBlockList = Bcons.emptyBody(localVars, localDecls);

    XobjList calcIterFuncArgs = Xcons.List(threadNumId.Ref());
    XMPutil.mergeLists(calcIterFuncArgs, loopIterRefList);
    XMPutil.mergeLists(calcIterFuncArgs, loopIndexAddrList);
    newLoopBlockList.insert(createFuncCallBlock("_XMP_gpu_calc_iter", calcIterFuncArgs));
    newLoopBlockList.insert(createFuncCallBlock("_XMP_gpu_calc_thread_id", Xcons.List(threadNumId.getAddr())));

    rewriteLoopBody(loopBlock, newLoopBlockList);
    newLoopBlockList.add(Xcons.List(Xcode.IF_STATEMENT, Xcons.binaryOp(Xcode.LOG_LT_EXPR, threadNumId.Ref(), totalIterId.Ref()), loopBlock.getBody().toXobject(), null));

    Xobject deviceBodyObj = newLoopBlockList.toXobject();
    XobjectDef deviceDef = XobjectDef.Func(deviceFuncId, deviceFuncParams, null, deviceBodyObj);

    try {
      if (out == null) {
        Writer w = new BufferedWriter(new FileWriter(getSrcName(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
        out = new XMPgpuDecompileWriter(w, env);
      }

      // add header include line
      out.println("#include \"xmp_gpu_func.hpp\"");
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

  private static XMPpair<XobjList, XobjList> genDeviceFuncParamArgs(XobjList paramIdList) {
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

    Ident totalIterId = Ident.Param("_XMP_GPU_TOTAL_ITER", Xtype.unsignedlonglongType);
    funcParams.add(totalIterId);
    funcArgs.add(totalIterId.Ref());

    return new XMPpair<XobjList, XobjList>(funcParams, funcArgs);
  }

  private static XobjList getNumThreads(XobjList gpuClause) {
    XobjList numThreads = null;

    for (Xobject c : gpuClause) {
      XMPpragma p = XMPpragma.valueOf(c.getArg(0));
      switch (p) {
        case GPU_NUM_THREADS:
          numThreads = (XobjList)c.getArg(1);
        default:
      }
    }

    return numThreads;
  }

  // FIXME localVars, localDecls delete
  private static void rewriteLoopBody(CforBlock loopBlock, BlockList bl) throws XMPexception {
    XMPgenSym symGen = new XMPgenSym();

    BasicBlockExprIterator iter = new BasicBlockExprIterator(loopBlock.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      rewriteExpr(iter.getExpr(), loopBlock, bl, symGen);
    }
  }

  // FIXME only for one dimension
  private static void rewriteExpr(Xobject expr, CforBlock loopBlock, BlockList bl, XMPgenSym symGen) throws XMPexception {
    if (expr == null) return;

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            String varName = myExpr.getString();
            XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(varName, loopBlock);
            XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
            if (alignedArray != null) {
              if (alignedArray.realloc()) {
                parseArrayExpr(iter, bl, symGen, loopBlock);
              }
            }
          } break;
        default:
      }
    }
  }

  // FIXME only for one dimension
  private static void parseArrayExpr(bottomupXobjectIterator iter, BlockList bl, XMPgenSym symGen,
                                     CforBlock loopBlock) throws XMPexception {
    Xobject myExpr = iter.getXobject();
    switch (myExpr.Opcode()) {
      case ARRAY_REF:
        iter.next();
        parseArrayExpr(iter, bl, symGen, loopBlock);
        return;
      case PLUS_EXPR:
        {
          String varName = null;
          Xobject indexRef = null;
          Xobject left = myExpr.left();
          if (left.Opcode() == Xcode.ARRAY_REF) {
            varName = myExpr.left().getString();
            indexRef = myExpr.right();
          } else {
            Xobject right = myExpr.right();
            if (right.Opcode() == Xcode.ARRAY_REF) {
              varName = myExpr.right().getString();
              indexRef = myExpr.left();
            } else {
              throw new XMPexception("wrong array expr");
            }
          }

          HashMap<String, Ident> gpuIndexTable = (HashMap<String, Ident>)loopBlock.getProp(GPU_INDEX_TABLE);
          if (gpuIndexTable == null) {
            gpuIndexTable = new HashMap<String, Ident>();
            loopBlock.setProp(GPU_INDEX_TABLE, (Object)gpuIndexTable);
          }

          //Ident indexId = gpuIndexTable.get(alignedArray.getName());
          Ident indexId = gpuIndexTable.get("XMP_DUMMY");
          if (indexId == null) {
            indexId = Ident.Local(symGen.getStr("_XMP_gpu_idx_"), Xtype.unsignedlonglongType);
            bl.getIdentList().add(indexId);
            bl.getDecls().add(Xcons.List(Xcode.VAR_DECL, indexId, null, null));

            //gpuIndexTable.put(alignedArray.getName(), indexId);
            gpuIndexTable.put("XMP_DUMMY", indexId);

            XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(varName, loopBlock);
            bl.insert(createFuncCallBlock("_XMP_gpu_calc_index",
                                          Xcons.List(indexId.getAddr(), indexRef, gpuData.getDeviceDescId().Ref())));
          }

          left = myExpr.left();
          if (left.Opcode() == Xcode.ARRAY_REF) {
            myExpr.setRight(indexId.Ref());
          } else {
            Xobject right = myExpr.right();
            if (right.Opcode() == Xcode.ARRAY_REF) {
              myExpr.setLeft(indexId.Ref());
            } else {
              throw new XMPexception("wrong array expr");
            }
          }

          iter.setXobject(myExpr);
        } return;
      default:
        throw new XMPexception("wrong array expr");
    }
  }
}
