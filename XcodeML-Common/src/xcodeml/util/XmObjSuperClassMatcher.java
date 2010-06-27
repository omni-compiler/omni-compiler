/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmObj;

/**
 * The matcher execute mathing about object's super class.
 */
public class XmObjSuperClassMatcher implements XmObjMatcher
{
    private Class<?> cls;

    /**
     * Creates XmObjClassMatcher
     *
     * @param name super class name to match wich.
     */
    public XmObjSuperClassMatcher(String name)
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
     * @param c super class to match wich.
     */
    public XmObjSuperClassMatcher(Class<?> c)
    {
        this.cls = c;
    }

    @Override
    public boolean match(XmObj n)
    {
        if(n == null) {
            return false;
        } else {
            Class<?> c;
            c = n.getClass();
            while(!c.equals(Object.class)) {
                if(c.equals(cls)) {
                    return true;
                }
                c = c.getSuperclass();
            }
            return Object.class.equals(cls);
        }
    }
}
