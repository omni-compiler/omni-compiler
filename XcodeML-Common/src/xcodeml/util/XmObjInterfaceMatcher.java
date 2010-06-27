/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmObj;

/**
 * The matcher execute mathing about object's interfaces.
 */
public class XmObjInterfaceMatcher implements XmObjMatcher
{
    private Class<?> cls;

    /**
     * Creates XmObjInterfaceMatcher.
     *
     * @param name class name to match wich.
     */
    public XmObjInterfaceMatcher(String name)
    {
        try {
            cls = Class.forName(name);
        } catch(Exception e) {
            cls = null;
        }
    }

    /**
     * Creates XmObjClassMatcher
     *
     * @param c class to match wich.
     */
    public XmObjInterfaceMatcher(Class<?> c)
    {
        this.cls = c;
    }

    @Override
    public boolean match(XmObj n)
    {
        if(n == null) {
            return false;
        } else {
            Class<?> c = n.getClass();
            do {
                for(Class<?> i : c.getInterfaces()) {
                    if(i.equals(cls)) {
                        return true;
                    }
                }
                c = c.getSuperclass();
            } while(!c.equals(Object.class));
            return false;
        }
    }

}
