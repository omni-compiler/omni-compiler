/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.io.*;
import java.util.*;

public class XMPgpuDecompiler {
  public static void decompile(LinkedList<XobjectDef> defList, XobjectFile env) {
    XMPgpuDecompileWriter out = new XMPgpuDecompileWriter(env);

    Iterator<XobjectDef> it = defList.iterator();
    while (it.hasNext()) {
      out.print(it.next());
      out.println();
    }
  }
}
