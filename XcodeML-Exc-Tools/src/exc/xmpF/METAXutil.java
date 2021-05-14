package exc.xmpF;

import java.io.*;
import java.util.Vector;
import exc.object.*;
import exc.block.*;
import xcodeml.util.XmOption;

public class METAXutil {

  static int gensym_num = 0;
  
  public static String genSym(String name) {
    String newString = new String("mtx_" + name + String.valueOf(gensym_num));
    gensym_num++;
    return newString;
  }

}
