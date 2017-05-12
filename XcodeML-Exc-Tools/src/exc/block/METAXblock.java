package exc.block;

import exc.object.*;

import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

public class METAXblock extends PragmaBlock {

  Class<?> metaxClass;
  Object metaxObj;

  Method run;
  Method runDecl;

  METAXblock(Xcode code, String pragma, Xobject args, BlockList body){
    super(code, pragma, args, body);
    try {
      metaxClass = Class.forName(pragma);
    } catch (ClassNotFoundException e){
      e.printStackTrace();
    }
    try {
      metaxObj = metaxClass.newInstance();
    } catch (InstantiationException e){
      e.printStackTrace();
    } catch (IllegalAccessException e){
      e.printStackTrace();
    }
    try {
      run = metaxClass.getMethod("run", BlockList.class, XobjList.class);
    } catch (NoSuchMethodException e){
      e.printStackTrace();
    }
    try {
      runDecl = metaxClass.getMethod("runDecl", BlockList.class, XobjList.class);
    } catch (NoSuchMethodException e){
      runDecl = null;
    }

  }

  public void run(){
    try {
      run.invoke(metaxObj, body, args);
    } catch (IllegalAccessException e){
      e.printStackTrace();
    } catch (InvocationTargetException e){
      e.printStackTrace();
    }
  }

  public void runDecl(){
    if (runDecl == null) return;
    try {
      runDecl.invoke(metaxObj, body, args);
    } catch (IllegalAccessException e){
      e.printStackTrace();
    } catch (InvocationTargetException e){
      e.printStackTrace();
    }
  }

}
