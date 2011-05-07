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

  private static XMPgpuDecompileWriter out = null;
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = ".cu";

  public static void decompile(Ident id, XobjList paramIdList, XobjList localVarIdList,
                               CforBlock loopBlock, XobjList gpuClause, XobjectFile env) throws XMPexception {
    BlockList loopBody = loopBlock.getBody();

    // schedule iteration
    XobjList loopIterRefList = Xcons.List();
    Iterator<Xobject> iter = localVarIdList.iterator();
    while(iter.hasNext()) {
      Ident localVarId = (Ident)iter.next();
      loopBody.addIdent(localVarId);
      Xobject loopDecls = loopBody.getDecls();
      if (loopDecls == null) {
        loopDecls = Xcons.List();
        loopBody.setDecls(loopDecls);
      }

      loopDecls.add(Xcons.List(Xcode.VAR_DECL, localVarId, null, null));

      XobjList loopIter = XMPutil.getLoopIter(loopBlock, localVarId.getName());
      if (loopIter != null) {
        // FIXME consider array dimension
        loopIterRefList.add(((Ident)(loopIter.getArg(0))).Ref());
        loopIterRefList.add(((Ident)(loopIter.getArg(1))).Ref());
        loopIterRefList.add(((Ident)(loopIter.getArg(2))).Ref());
        loopBody.insert(createFuncCallBlock("_XMP_gpu_calc_thread_id", Xcons.List(localVarId.getAddr())));
      }
    }

    Xobject deviceBodyObj = loopBody.toXobject();

    XMPpair<XobjList, XobjList> deviceFuncParamArgs = genDeviceFuncParamArgs(paramIdList);
    XobjList deviceFuncParams = deviceFuncParamArgs.getFirst();
    XobjList deviceFuncArgs = deviceFuncParamArgs.getSecond();

    try {
      if (out == null) {
        Writer w = new BufferedWriter(new FileWriter(getSrcName(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
        out = new XMPgpuDecompileWriter(w, env);
      }

      // add header include line
      out.println("#include \"xmp_gpu_func.hpp\"");
      out.println();

      // decompile device function
      XobjectDef deviceDef = XobjectDef.Func(id, deviceFuncParams, null, deviceBodyObj);
      out.printDeviceFunc(deviceDef, id);
      out.println();

      // generate wrapping function
      // FIXME add configuration parameters
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

      Ident confParamFuncId = XMP.getMacroId(confParamFuncName);
      XobjList confParamFuncArgs = Xcons.List(blockXid.getAddr(), blockYid.getAddr(), blockZid.getAddr(),
                                              threadXid.getAddr(), threadYid.getAddr(), threadZid.getAddr());
      XMPutil.mergeLists(confParamFuncArgs, numThreads);
      XMPutil.mergeLists(confParamFuncArgs, loopIterRefList);
      Xobject confParamFuncCall = confParamFuncId.Call(confParamFuncArgs);

      Ident deviceFuncId = XMP.getMacroId(id.getName() + "_DEVICE");
      Xobject deviceFuncCall = deviceFuncId.Call(deviceFuncArgs);
      deviceFuncCall.setProp(GPU_FUNC_CONF,
                             (Object)Xcons.List(blockXid, blockYid, blockZid,
                                                threadXid, threadYid, threadZid));

      XobjList hostBodyIdList = Xcons.List(blockXid, blockYid, blockZid, threadXid, threadYid, threadZid);
      XobjList hostBodyDecls = Xcons.List(Xcons.List(Xcode.VAR_DECL, blockXid, null, null),
                                          Xcons.List(Xcode.VAR_DECL, blockYid, null, null),
                                          Xcons.List(Xcode.VAR_DECL, blockZid, null, null),
                                          Xcons.List(Xcode.VAR_DECL, threadXid, null, null),
                                          Xcons.List(Xcode.VAR_DECL, threadYid, null, null),
                                          Xcons.List(Xcode.VAR_DECL, threadZid, null, null));
      Xobject hostBodyObj = Xcons.CompoundStatement(hostBodyIdList, hostBodyDecls, Xcons.List(confParamFuncCall, deviceFuncCall));

      // decompie wrapping function
      XobjectDef hostDef = XobjectDef.Func(id, paramIdList, null, hostBodyObj);
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
      if (!id.getName().startsWith("_XMP_loop_")) {
        funcParams.add(id);
        funcArgs.add(id.Ref());
      }
    }

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
}
