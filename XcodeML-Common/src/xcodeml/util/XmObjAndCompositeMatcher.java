/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.util.ArrayList;

import xcodeml.XmObj;

/**
 * The matcher execute AND matchings.
 */
public class XmObjAndCompositeMatcher implements XmObjMatcher
{
    private ArrayList<XmObjMatcher> list;

    /**
     * Creates XmObjAndCompositeMatcher
     */
    public XmObjAndCompositeMatcher()
    {
        list = new ArrayList<XmObjMatcher>();
    }

    @Override
    public boolean match(XmObj n)
    {
        for(XmObjMatcher m : list) {
            if(!m.match(n)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Adds matcher be executed by this matcher.
     *
     * @param matcher provides condition.
     */
    public void addMatcher(XmObjMatcher matcher)
    {
        list.add(matcher);
    }
}
